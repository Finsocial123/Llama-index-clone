import requests
import json
import logging
import os
import base64
import mimetypes
from PIL import Image
import io
import time

logger = logging.getLogger(__name__)

# Default to the RunPod URL but allow for environment variable override
OLLAMA_API_URL = 'https://10kko5o9i8ec9w-11434.proxy.runpod.net'
TEXT_MODEL = os.environ.get("OLLAMA_TEXT_MODEL", "llama3.2:1b")
VISION_MODEL = os.environ.get("OLLAMA_VISION_MODEL", "llama3.2-vision")

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds

def is_image_file(filepath_or_content):
    """
    Determine if the file is an image based on extension or content
    """
    if isinstance(filepath_or_content, str) and os.path.isfile(filepath_or_content):
        # Check file extension
        ext = os.path.splitext(filepath_or_content)[1].lower()
        if ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.tiff']:
            return True
        
        # Double check with mimetype
        mime_type, _ = mimetypes.guess_type(filepath_or_content)
        return mime_type is not None and mime_type.startswith('image/')
    return False

def encode_image_to_base64(image_path):
    """
    Encode an image file to base64 string
    """
    try:
        # Resize image if it's too large to avoid token limits
        with Image.open(image_path) as img:
            # Resize if the image is too large (keeping aspect ratio)
            max_size = 800
            if max(img.size) > max_size:
                ratio = max_size / max(img.size)
                new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            
            # Convert to RGB if it's not (e.g., RGBA)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Save to a buffer
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=85)
            buffer.seek(0)
            image_data = buffer.read()
    
        # Encode as base64
        encoded = base64.b64encode(image_data).decode('utf-8')
        return encoded
    except Exception as e:
        logger.error(f"Error encoding image: {str(e)}")
        raise

def get_ollama_response(question, context):
    """
    Get a response from the Ollama API, supporting both text and images
    """
    # Check if we're dealing with an image file
    is_image = False
    image_path = None
    
    # Check if context contains an image file path
    if isinstance(context, str) and context.startswith("IMAGE_FILE:"):
        is_image = True
        image_path = context.replace("IMAGE_FILE:", "").strip()
        logger.info(f"Detected image processing request for: {image_path}")
    
    # Check if this is a transcript (from audio/video)
    is_transcript = not is_image and isinstance(context, str) and "Transcription of " in context[:100] and "Content Type: " in context[:200]
    
    # Prepare the messages based on content type
    try:
        if is_image:
            # Use the vision model for images
            logger.info(f"Using vision model for image analysis: {image_path}")
            
            # Encode the image to base64
            try:
                image_base64 = encode_image_to_base64(image_path)
                logger.info(f"Successfully encoded image, base64 length: {len(image_base64)}")
            except Exception as e:
                logger.error(f"Error encoding image: {e}")
                return f"Error preparing image for analysis: {str(e)}"
            
            # Format specifically for Llama3.2-vision model based on Ollama documentation
            # Note: This is different from OpenAI's vision API format
            system_prompt = "You are an assistant capable of analyzing images. Be detailed but concise."
            
            # Modified JSON payload for vision API
            payload = {
                "model": VISION_MODEL,
                "stream": False,
                "prompt": f"{system_prompt}\n\nUser: <image>\n{question}\nAssistant: ",
                "images": [image_base64]
            }
            
            logger.info(f"Sending question to Ollama vision model: {question[:100]}...")
            
            # Implement retry logic for API calls
            retry_count = 0
            last_error = None
            
            while retry_count < MAX_RETRIES:
                try:
                    # Add timeout to prevent hanging requests
                    response = requests.post(
                        f"{OLLAMA_API_URL}/api/generate",  # Note: use /generate endpoint
                        headers={"Content-Type": "application/json"},
                        json=payload,
                        timeout=120  # 2 minutes timeout for API calls
                    )
                    
                    # If request is successful, return the response
                    if response.status_code == 200:
                        result = response.json()
                        if "response" in result:
                            return result["response"]
                        else:
                            return "Error: Unexpected response format from Ollama API for image analysis"
                    else:
                        # Handle specific HTTP errors
                        if response.status_code == 404:
                            # Model not found error
                            return f"Error: Model '{model}' not found. Please check if the model is installed on the server."
                        elif response.status_code == 500:
                            # Server error
                            error_msg = "Server error occurred"
                            try:
                                error_data = response.json()
                                if 'error' in error_data:
                                    error_msg = error_data['error']
                            except:
                                pass
                            raise Exception(f"Ollama API server error: {error_msg}")
                        else:
                            # Other HTTP errors
                            raise Exception(f"HTTP error {response.status_code}: {response.text}")
                        
                except requests.exceptions.Timeout:
                    logger.warning(f"Request timed out (attempt {retry_count + 1}/{MAX_RETRIES})")
                    last_error = "Request timed out. The server took too long to respond."
                except requests.exceptions.ConnectionError:
                    logger.warning(f"Connection error (attempt {retry_count + 1}/{MAX_RETRIES})")
                    last_error = "Connection error. Could not connect to the Ollama API. Please check if the service is running."
                except Exception as e:
                    logger.error(f"Vision API call error: {str(e)} (attempt {retry_count + 1}/{MAX_RETRIES})")
                    last_error = str(e)
                
                # Increment retry count and wait before retrying
                retry_count += 1
                if retry_count < MAX_RETRIES:
                    time.sleep(RETRY_DELAY)
            
            # If we've exhausted retries, use fallback text
            return f"I couldn't analyze the image due to technical difficulties: {last_error}. Please try again later."
        
        else:
            # Use text model for documents and transcripts
            if is_transcript:
                prompt = f"""
                You are a helpful assistant that answers questions based ONLY on the provided transcript from an audio or video file.
                The transcript may contain errors or unclear parts due to the automatic transcription process.
                If you don't know the answer based on the transcript, say "I don't have enough information in the transcript to answer this question."
                Do not use any knowledge outside of what is provided in the transcript.
                
                Transcript:
                {context}
                
                Question: {question}
                
                Answer:
                """
            else:
                prompt = f"""
                You are a helpful assistant that answers questions based ONLY on the provided context.
                If you don't know the answer based on the context, say "I don't have enough information to answer this question."
                Do not use any knowledge outside of what is provided in the context.
                
                Context:
                {context}
                
                Question: {question}
                
                Answer:
                """
            
            messages = [
                {"role": "user", "content": prompt}
            ]
            
            model = TEXT_MODEL
            
        logger.info(f"Sending question to Ollama using model {model}: {question[:100]}...")
        
        # Implement retry logic for API calls
        retry_count = 0
        last_error = None
        
        while retry_count < MAX_RETRIES:
            try:
                # Add timeout to prevent hanging requests
                response = requests.post(
                    f"{OLLAMA_API_URL}/api/chat",
                    headers={"Content-Type": "application/json"},
                    json={
                        "model": model,
                        "messages": messages,
                        "stream": False
                    },
                    timeout=120  # 2 minutes timeout for API calls
                )
                
                # If request is successful, return the response
                if response.status_code == 200:
                    result = response.json()
                    if "message" in result and "content" in result["message"]:
                        return result["message"]["content"]
                    else:
                        return "Error: Unexpected response format from Ollama API"
                else:
                    # Handle specific HTTP errors
                    if response.status_code == 404:
                        # Model not found error
                        return f"Error: Model '{model}' not found. Please check if the model is installed on the server."
                    elif response.status_code == 500:
                        # Server error
                        error_msg = "Server error occurred"
                        try:
                            error_data = response.json()
                            if 'error' in error_data:
                                error_msg = error_data['error']
                        except:
                            pass
                        raise Exception(f"Ollama API server error: {error_msg}")
                    else:
                        # Other HTTP errors
                        raise Exception(f"HTTP error {response.status_code}: {response.text}")
                        
            except requests.exceptions.Timeout:
                logger.warning(f"Request timed out (attempt {retry_count + 1}/{MAX_RETRIES})")
                last_error = "Request timed out. The server took too long to respond."
            except requests.exceptions.ConnectionError:
                logger.warning(f"Connection error (attempt {retry_count + 1}/{MAX_RETRIES})")
                last_error = "Connection error. Could not connect to the Ollama API. Please check if the service is running."
            except Exception as e:
                logger.error(f"API call error: {str(e)} (attempt {retry_count + 1}/{MAX_RETRIES})")
                last_error = str(e)
            
            # Increment retry count and wait before retrying
            retry_count += 1
            if retry_count < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
            
        # If we've exhausted retries, use fallback text
        if model == VISION_MODEL:
            return f"I couldn't process the image due to technical difficulties: {last_error}. Please try again later."
        else:
            return f"I couldn't process your question due to technical difficulties: {last_error}. Please try again later."
    
    except Exception as e:
        logger.error(f"Unexpected error in Ollama client: {str(e)}")
        return f"Unexpected error processing your request: {str(e)}. Please try again later."
