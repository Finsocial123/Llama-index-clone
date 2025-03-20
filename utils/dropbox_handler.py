import re
import aiohttp
import os
import logging
import mimetypes

logger = logging.getLogger(__name__)

def get_filename_from_url(url):
    """Extract filename from Dropbox URL"""
    try:
        # Extract filename from URL pattern
        # Example: https://www.dropbox.com/scl/fi/6qwt7nkpg9pe6zkfahru7/285_OOPS-lecture-notes-Complete.pdf?rlkey=...
        pattern = r'\/([^\/\?]+)(?:\?|$)'
        match = re.search(pattern, url)
        
        if match:
            filename = match.group(1)
            # URL decode if needed
            filename = filename.replace("%20", " ")
            logger.info(f"Extracted filename: {filename}")
            return filename
        else:
            logger.error(f"Could not extract filename from URL: {url}")
            return None
    except Exception as e:
        logger.error(f"Error extracting filename: {str(e)}")
        return None

def convert_to_direct_download(url):
    """Convert Dropbox sharing URL to direct download URL"""
    # If the URL already has dl=0 or dl=1, replace it with dl=1
    if "dl=0" in url:
        return url.replace("dl=0", "dl=1")
    elif "dl=1" in url:
        return url
    # If the URL has no dl parameter, add it
    elif "?" in url:
        return url + "&dl=1"
    else:
        return url + "?dl=1"

async def download_file_from_dropbox(url, output_path):
    """Download file from Dropbox URL with improved media file handling"""
    try:
        # Convert to direct download URL
        download_url = convert_to_direct_download(url)
        logger.info(f"Downloading from: {download_url}")
        
        # Download the file
        async with aiohttp.ClientSession() as session:
            async with session.get(download_url) as response:
                response.raise_for_status()
                
                # Check if this is a media file based on content-type
                content_type = response.headers.get('content-type', '').lower()
                is_media = 'audio' in content_type or 'video' in content_type
                
                if is_media:
                    logger.info(f"Detected media file from Dropbox (content-type: {content_type})")
                
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
        
        logger.info(f"File downloaded successfully to: {output_path}")
        
        # Verify file size
        file_size = os.path.getsize(output_path)
        if file_size == 0:
            raise ValueError("Downloaded file is empty")
        
        # For media files, verify the file is playable
        if is_media:
            extension = os.path.splitext(output_path)[1].lower()
            expected_media_extensions = ['.mp3', '.mp4', '.wav', '.avi', '.mkv', '.m4a', '.flac', '.ogg']
            
            if not extension or extension not in expected_media_extensions:
                # Try to guess correct extension from content-type
                guessed_ext = mimetypes.guess_extension(content_type) or ".bin"
                logger.info(f"Media file has incorrect or missing extension. Guessed: {guessed_ext}")
            
        logger.info(f"File size: {file_size} bytes")
        return True
    except Exception as e:
        logger.error(f"Error downloading from Dropbox: {str(e)}")
        # Clean up partial download if file exists
        if os.path.exists(output_path):
            os.remove(output_path)
        raise
