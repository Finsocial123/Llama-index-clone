import os
import PyPDF2
from docx import Document
import pandas as pd
import csv
import json
import xml.etree.ElementTree as ET
import html2text
import subprocess
import logging
import mimetypes
import magic  # This requires python-magic library
import whisper
import tempfile
import torch
from pathlib import Path
import ffmpeg
import time
import shutil

logger = logging.getLogger(__name__)

# Global variable for Whisper model to avoid reloading it on each request
_whisper_model = None

def get_whisper_model():
    """Load whisper model (lazily to save memory if not used)"""
    global _whisper_model
    if _whisper_model is None:
        # Check if CUDA (GPU) is available for faster processing
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading Whisper model on {device}...")
        try:
            # Use 'base' model for a balance of accuracy and speed
            # Other options: 'tiny', 'small', 'medium', 'large'
            _whisper_model = whisper.load_model('base', device=device)
            logger.info("Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading Whisper model: {str(e)}")
            raise
    return _whisper_model

def process_file(filepath):
    """Process different file types and extract content"""
    # First try to detect file type by content (more reliable than extension)
    try:
        mime = magic.Magic(mime=True)
        content_type = mime.from_file(filepath)
        logger.info(f"Detected content type: {content_type}")
    except Exception as e:
        logger.warning(f"Could not detect MIME type: {str(e)}")
        content_type = None
    
    # Fall back to extension if content type detection failed
    file_extension = os.path.splitext(filepath)[1].lower()
    logger.info(f"File extension: {file_extension}")
    
    try:
        # Image files
        if (content_type and content_type.startswith('image/')) or \
           file_extension in ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.tiff']:
            logger.info(f"Detected image file: {filepath}")
            return f"IMAGE_FILE:{filepath}"
        
        # Audio/Video files - check both mime type and extension
        if (content_type and ('audio' in content_type or 'video' in content_type)) or \
           file_extension in ['.mp3', '.mp4', '.wav', '.avi', '.mkv', '.m4a', '.flac', '.ogg', '.webm', '.mov']:
            logger.info(f"Detected audio/video file: {filepath}")
            return extract_from_media(filepath)
            
        # PDF files
        if content_type == 'application/pdf' or file_extension == '.pdf':
            return extract_from_pdf(filepath)
        
        # HTML files
        if content_type and ('html' in content_type or 'xml' in content_type):
            return extract_from_html_xml(filepath, '.html')
        
        # Word documents
        elif file_extension in ['.docx', '.doc']:
            return extract_from_docx(filepath)
        
        # Excel files
        elif file_extension in ['.xlsx', '.xls']:
            return extract_from_excel(filepath)
        
        # CSV files
        elif file_extension == '.csv' or (content_type and 'csv' in content_type):
            return extract_from_csv(filepath)
        
        # JSON files
        elif file_extension == '.json' or (content_type and 'json' in content_type):
            return extract_from_json(filepath)
        
        # XML/HTML files
        elif file_extension in ['.xml', '.html', '.htm']:
            return extract_from_html_xml(filepath, file_extension)
        
        # Text and code files - this should be the fallback for any text-like content
        elif any([
            file_extension in ['.txt', '.py', '.js', '.java', '.c', '.cpp', '.cs', '.php', '.rb', '.go', '.rs', '.ts'],
            content_type and ('text/' in content_type)
        ]):
            return extract_from_text(filepath)
        
        # If we don't recognize it by content type or extension, try to read it as text
        else:
            try:
                return extract_from_text(filepath)
            except:
                return f"Unsupported file type: {file_extension or content_type or 'unknown'}"
            
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        return f"Error processing file: {str(e)}"

def extract_from_pdf(filepath):
    """Extract text from PDF with robust error handling"""
    try:
        text = ""
        with open(filepath, 'rb') as f:
            # First verify this is actually a PDF
            header = f.read(5)
            f.seek(0)  # Reset file pointer to beginning
            
            if header != b'%PDF-':
                raise ValueError("File is not a valid PDF (incorrect header)")
            
            try:
                pdf_reader = PyPDF2.PdfReader(f)
                
                # Check if PDF is encrypted
                if pdf_reader.is_encrypted:
                    raise ValueError("PDF is encrypted/password protected")
                
                # Extract text from each page
                for page_num in range(len(pdf_reader.pages)):
                    page_text = pdf_reader.pages[page_num].extract_text() or ""
                    text += page_text + "\n"
                    
                if not text.strip():
                    logger.warning("PDF appears to contain no extractable text (might be scanned)")
                    return "This PDF appears to be scanned or contains no extractable text."
                    
                return text
            except PyPDF2.errors.PdfReadError as e:
                raise ValueError(f"Invalid PDF structure: {str(e)}")
    except Exception as e:
        logger.error(f"PDF extraction error: {str(e)}")
        return f"Could not extract text from PDF: {str(e)}"

def extract_from_docx(filepath):
    doc = Document(filepath)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_from_excel(filepath):
    df = pd.read_excel(filepath)
    return df.to_string()

def extract_from_csv(filepath):
    result = []
    with open(filepath, 'r', encoding='utf-8') as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            result.append(",".join(row))
    return "\n".join(result)

def extract_from_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return json.dumps(data, indent=2)

def extract_from_html_xml(filepath, extension):
    try:
        if extension in ['.html', '.htm']:
            with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                converter = html2text.HTML2Text()
                converter.ignore_links = False
                return converter.handle(f.read())
        else:  # XML
            try:
                tree = ET.parse(filepath)
                return ET.tostring(tree.getroot(), encoding='utf-8').decode('utf-8')
            except:
                # If XML parsing fails, try as plain text
                return extract_from_text(filepath)
    except Exception as e:
        logger.error(f"HTML/XML extraction error: {str(e)}")
        try:
            # Fallback to plain text
            return extract_from_text(filepath)
        except:
            return f"Could not extract text from HTML/XML: {str(e)}"

def extract_audio_from_video(video_path, output_audio_path=None):
    """Extract audio track from a video file with improved error handling"""
    try:
        if output_audio_path is None:
            # Create a temporary file with .wav extension if no output path is provided
            temp_dir = tempfile.gettempdir()
            output_audio_path = os.path.join(temp_dir, f"audio_{int(time.time())}.wav")
        
        logger.info(f"Extracting audio from video: {video_path} to {output_audio_path}")
        
        # Check if FFmpeg is installed
        try:
            version = ffmpeg.probe(video_path)
            logger.info(f"FFmpeg probe result: {version.get('format', {}).get('format_name', 'unknown')}")
        except ffmpeg.Error as e:
            logger.error(f"FFmpeg probe error: {e.stderr.decode() if hasattr(e, 'stderr') else str(e)}")
            # Try command-line fallback if ffmpeg-python fails
            try:
                import subprocess
                subprocess.run(['ffmpeg', '-i', video_path, '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', output_audio_path], 
                               check=True, capture_output=True)
                logger.info("Used subprocess fallback for audio extraction")
                return output_audio_path
            except Exception as sub_e:
                logger.error(f"Subprocess fallback also failed: {str(sub_e)}")
                raise
            
        # Use ffmpeg to extract audio
        try:
            (
                ffmpeg
                .input(video_path)
                .output(output_audio_path, acodec='pcm_s16le', ac=1, ar='16000')
                .run(quiet=True, overwrite_output=True)
            )
        except ffmpeg.Error as e:
            logger.error(f"FFmpeg error: {e.stderr.decode() if hasattr(e, 'stderr') else str(e)}")
            raise
            
        logger.info(f"Audio extraction successful: {output_audio_path}")
        return output_audio_path
    except Exception as e:
        logger.error(f"Error extracting audio from video: {str(e)}")
        raise

def extract_from_media(filepath):
    """Extract text from audio/video files using Whisper with improved handling"""
    try:
        logger.info(f"Processing audio/video file: {filepath}")
        file_extension = os.path.splitext(filepath)[1].lower()
        file_size = os.path.getsize(filepath)
        
        # Copy to a temporary file to avoid any permission issues
        temp_dir = tempfile.gettempdir()
        temp_filepath = os.path.join(temp_dir, f"media_file_{int(time.time())}{file_extension}")
        shutil.copy2(filepath, temp_filepath)
        logger.info(f"Copied to temporary file: {temp_filepath}")
        
        # Check if we need to extract audio from video
        is_video = file_extension in ['.mp4', '.avi', '.mkv', '.mov', '.flv', '.wmv', '.webm']
        audio_path = temp_filepath
        temp_audio_file = None
        
        if is_video:
            logger.info("Detected video file, extracting audio track")
            try:
                temp_audio_file = extract_audio_from_video(temp_filepath)
                audio_path = temp_audio_file
                logger.info(f"Using extracted audio: {audio_path}")
            except Exception as e:
                logger.error(f"Failed to extract audio from video: {str(e)}")
                return f"Error extracting audio from video: {str(e)}. Please ensure ffmpeg is installed correctly."
        
        # Load Whisper model
        try:
            model = get_whisper_model()
        except Exception as e:
            logger.error(f"Error loading Whisper model: {str(e)}")
            # Clean up temporary files
            try:
                os.remove(temp_filepath)
                if temp_audio_file and os.path.exists(temp_audio_file):
                    os.remove(temp_audio_file)
            except:
                pass
            return f"Error loading speech recognition model: {str(e)}"
        
        # Transcribe the audio
        try:
            logger.info(f"Starting transcription of: {audio_path}")
            
            # For all media files, use the full transcribe method for better results
            try:
                result = model.transcribe(audio_path, verbose=False)
                transcript = result['text']
                detected_language = result.get('language', 'unknown')
                logger.info(f"Transcription complete, language: {detected_language}")
            except Exception as e:
                logger.error(f"Full transcription failed: {str(e)}. Trying segmented approach.")
                
                # If full transcription fails, try processing in chunks
                try:
                    # Load audio
                    audio = whisper.load_audio(audio_path)
                    # Process in 30-second chunks
                    transcript_parts = []
                    chunk_size = 30 * 16000  # 30 seconds at 16kHz
                    
                    for i in range(0, len(audio), chunk_size):
                        chunk = audio[i:i+chunk_size]
                        chunk_pad = whisper.pad_or_trim(chunk)
                        mel = whisper.log_mel_spectrogram(chunk_pad).to(model.device)
                        result = model.decode(mel)
                        transcript_parts.append(result.text)
                    
                    transcript = " ".join(transcript_parts)
                    detected_language = "unknown (chunked processing)"
                except Exception as chunk_err:
                    logger.error(f"Chunked processing also failed: {str(chunk_err)}")
                    raise Exception(f"All transcription methods failed: {str(e)}, then: {str(chunk_err)}")
            
            # Add header to the transcript
            file_name = os.path.basename(filepath)
            transcription_header = f"Transcription of {file_name}\n"
            
            if is_video:
                transcription_header += "Content Type: Video with Audio\n"
            else:
                transcription_header += "Content Type: Audio\n"
                
            transcription_header += f"Detected language: {detected_language}\n"
            transcription_header += f"Duration: {result.get('duration', 'Unknown')} seconds\n"
            transcription_header += "=" * 50 + "\n\n"
            
            final_transcript = transcription_header + transcript
            logger.info(f"Transcription complete, length: {len(final_transcript)} characters")
            
            # Make sure we have a meaningful transcript
            if len(transcript.strip()) < 10:
                logger.warning("Transcription produced minimal or no text")
                return "The audio/video file could not be transcribed properly. It may contain no speech or very low quality audio."
            
            # Clean up temporary files
            try:
                os.remove(temp_filepath)
                if temp_audio_file and os.path.exists(temp_audio_file):
                    os.remove(temp_audio_file)
            except Exception as clean_err:
                logger.warning(f"Error cleaning up temp files: {str(clean_err)}")
                
            return final_transcript
            
        except Exception as e:
            logger.error(f"Transcription error: {str(e)}")
            # Clean up
            try:
                os.remove(temp_filepath)
                if temp_audio_file and os.path.exists(temp_audio_file):
                    os.remove(temp_audio_file)
            except:
                pass
            return f"Error during transcription: {str(e)}"
    
    except Exception as e:
        logger.error(f"Media extraction error: {str(e)}")
        return f"Error processing audio/video file: {str(e)}"

def extract_from_text(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            return f.read()
    except Exception as e:
        # Try with different encodings if UTF-8 fails
        try:
            with open(filepath, 'r', encoding='latin-1', errors='replace') as f:
                return f.read()
        except:
            logger.error(f"Text extraction error: {str(e)}")
            return f"Could not read text file: {str(e)}"

def extract_from_image(filepath):
    """
    Prepare image file for vision model processing.
    We don't extract text here - just return the file path prefixed
    so the Ollama client knows to use the vision model.
    """
    try:
        # Verify it's a valid image file
        from PIL import Image
        img = Image.open(filepath)
        img.verify()  # Verify it's an image
        
        width, height = img.size
        logger.info(f"Valid image file: {filepath}, dimensions: {width}x{height}")
        
        # Return specially formatted string that signals this is an image file
        return f"IMAGE_FILE:{filepath}"
    except Exception as e:
        logger.error(f"Invalid image file: {str(e)}")
        return f"Error: Could not process image file - {str(e)}"
