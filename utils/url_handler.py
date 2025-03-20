import os
import re
import aiohttp
import logging
from urllib.parse import urlparse, unquote
import tempfile
import mimetypes
from .dropbox_handler import download_file_from_dropbox
from .youtube_handler import extract_video_id, get_youtube_transcript

logger = logging.getLogger(__name__)

async def detect_url_type(url):
    """Detect the type of URL (dropbox, youtube, or generic)"""
    if "dropbox.com" in url.lower():
        return "dropbox"
    elif "youtube.com" in url.lower() or "youtu.be" in url.lower():
        return "youtube"
    else:
        return "generic"

def is_image_url(url, content_type=None):
    """Check if a URL points to an image file"""
    # Check content type if provided
    if (content_type and content_type.startswith('image/')):
        return True
    
    # Check URL extension
    parsed_url = urlparse(url)
    path = parsed_url.path.lower()
    
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.tiff']
    return any(path.endswith(ext) for ext in image_extensions)

def get_filename_from_generic_url(url):
    """Extract filename from any URL"""
    try:
        # Parse the URL and extract the path
        parsed_url = urlparse(url)
        path = parsed_url.path
        
        # Try to get filename from the path
        filename = os.path.basename(path)
        
        # URL decode the filename
        filename = unquote(filename)
        
        # If no filename found or it's empty, generate a generic one
        if not filename or filename == "/" or "." not in filename:
            # Try to use the domain name as part of the filename
            domain = parsed_url.netloc.replace("www.", "")
            domain = domain.split(".")[0] if "." in domain else domain
            filename = f"document_from_{domain}.txt"
        
        logger.info(f"Extracted filename: {filename}")
        return filename
    except Exception as e:
        logger.error(f"Error extracting filename from URL: {str(e)}")
        # Return a generic filename if extraction fails
        return "downloaded_document.txt"

async def download_from_generic_url(url, output_path):
    """Download file from any generic URL with improved media handling"""
    try:
        logger.info(f"Downloading from generic URL: {url}")
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml,application/pdf,*/*",
            "Range": "bytes=0-"  # Support for resumable downloads
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, allow_redirects=True) as response:
                response.raise_for_status()
                
                # Check content type
                content_type = response.headers.get("content-type", "").lower()
                logger.info(f"Content-Type from URL: {content_type}")
                is_media = 'audio' in content_type or 'video' in content_type
                
                # Get content-disposition for filename if available
                content_disp = response.headers.get("content-disposition", "")
                if "filename=" in content_disp:
                    filename_match = re.findall('filename="?([^"]+)"?', content_disp)
                    if filename_match:
                        filename = filename_match[0]
                        logger.info(f"Found filename in headers: {filename}")
                        
                        # If we have a media file without proper extension, try to fix it
                        if is_media:
                            ext = os.path.splitext(filename)[1].lower()
                            if not ext or ext not in ['.mp3', '.mp4', '.wav', '.avi', '.mkv']:
                                guessed_ext = mimetypes.guess_extension(content_type)
                                if guessed_ext:
                                    new_filename = f"{os.path.splitext(output_path)[0]}{guessed_ext}"
                                    logger.info(f"Renamed media file with proper extension: {new_filename}")
                                    output_path = new_filename
                
                # Save the file to disk
                with open(output_path, 'wb') as f:
                    total_size = 0
                    while True:
                        chunk = await response.content.read(8192)
                        if not chunk:
                            break
                        f.write(chunk)
                        total_size += len(chunk)
                        
                        # For very large files, display progress
                        if total_size % (10 * 1024 * 1024) == 0:  # Every 10MB
                            logger.info(f"Downloaded {total_size / (1024*1024):.2f} MB so far")
                            
                        if total_size > 100 * 1024 * 1024:  # 100 MB limit
                            logger.warning("File too large, stopping download")
                            break
        
        logger.info(f"File downloaded successfully to: {output_path}")
        
        # Verify file size
        file_size = os.path.getsize(output_path)
        if file_size == 0:
            raise ValueError("Downloaded file is empty")
        logger.info(f"File size: {file_size} bytes")
        
        # For media files, add additional verification
        if is_media:
            logger.info(f"Downloaded media file: {output_path}")
            # Get file extension and check if it matches content type
            ext = os.path.splitext(output_path)[1].lower()
            expected_ext = mimetypes.guess_extension(content_type)
            if expected_ext and ext != expected_ext and ext not in ['.mp3', '.mp4', '.wav', '.avi', '.mkv']:
                logger.warning(f"Media file extension mismatch: {ext} vs expected {expected_ext}")
        
        # Check if this is an image URL
        is_image = is_image_url(url)
        if is_image:
            logger.info(f"URL appears to be an image: {url}")
        
        # Return information about the content type for processing hints
        return {
            "success": True, 
            "content_type": content_type,
            "file_size": file_size,
            "is_media": is_media,
            "is_image": is_image or (content_type and content_type.startswith('image/')),
            "output_path": output_path  # Return the possibly updated output path
        }
    except Exception as e:
        logger.error(f"Error downloading from URL: {str(e)}")
        # Clean up partial download if file exists
        if os.path.exists(output_path):
            os.remove(output_path)
        raise

async def process_any_url(url, upload_folder):
    """Process any URL and download its content with improved media handling"""
    try:
        # Detect URL type
        url_type = await detect_url_type(url)
        logger.info(f"Detected URL type: {url_type}")
        
        if url_type == "youtube":
            # Handle YouTube URLs
            video_id = extract_video_id(url)
            if not video_id:
                raise ValueError("Could not extract video ID from YouTube URL")
                
            # YouTube processing will be handled by the dedicated endpoint
            return {"type": "youtube", "url": url}
            
        elif url_type == "dropbox":
            # Handle Dropbox URLs
            # Get filename from URL
            from .dropbox_handler import get_filename_from_url
            original_filename = get_filename_from_url(url)
            if not original_filename:
                raise ValueError("Could not extract filename from Dropbox URL")
                
            # Dropbox processing will be handled by the dedicated endpoint
            return {"type": "dropbox", "url": url}
            
        else:
            # Handle generic URLs
            original_filename = get_filename_from_generic_url(url)
            file_id = os.urandom(4).hex()  # Generate a short unique ID
            filename = f"{file_id}_{original_filename}"
            filepath = os.path.join(upload_folder, filename)
            
            # Download the file
            result = await download_from_generic_url(url, filepath)
            
            # If output path was changed (e.g., due to extension correction), update filepath
            if "output_path" in result and result["output_path"] != filepath:
                filepath = result["output_path"]
                filename = os.path.basename(filepath)
            
            return {
                "type": "generic",
                "filepath": filepath,
                "original_filename": original_filename,
                "content_type": result.get("content_type", "application/octet-stream"),
                "is_media": result.get("is_media", False),
                "is_image": result.get("is_image", False)
            }
            
    except Exception as e:
        logger.error(f"Error processing URL: {str(e)}")
        raise ValueError(f"Error processing URL: {str(e)}")
