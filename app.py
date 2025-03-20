from fastapi import FastAPI, Request, File, UploadFile, HTTPException, Form, BackgroundTasks, Security, Depends
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import uuid
import uvicorn
from utils.file_processor import process_file
from utils.dropbox_handler import download_file_from_dropbox, get_filename_from_url
from utils.youtube_handler import get_youtube_transcript, extract_video_id
from utils.url_handler import process_any_url
from utils.ollama_client import get_ollama_response
from typing import Dict, Optional, Any
import shutil
import logging
import json
from starlette.middleware.sessions import SessionMiddleware
from pydantic import BaseModel
import asyncio
import concurrent.futures
import time
import requests
from dotenv import load_dotenv
from fastapi.security import APIKeyHeader

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Document Q&A Tool")
# Use a longer maxAge for sessions to ensure they don't expire too quickly
app.add_middleware(
    SessionMiddleware, 
    secret_key="your-very-long-and-secure-secret-key-here-12345",
    max_age=3600,  # 1 hour
    same_site="lax",  # Less restrictive same-site policy
    https_only=False  # Set to True in production with HTTPS
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Create a folder for temporary session storage
SESSION_STORAGE = "session_storage"
os.makedirs(SESSION_STORAGE, exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Define request model for URL
class UrlRequest(BaseModel):
    url: str

# Define API key header for session management
API_KEY_HEADER = APIKeyHeader(name="X-Session-ID", auto_error=False)

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    # Clear any existing session data when loading the main page
    request.session.clear()
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
async def upload_file(file: UploadFile = File(...), request: Request = None):
    try:
        logger.info(f"Received file: {file.filename}")
        
        if not file.filename:
            raise HTTPException(status_code=400, detail="No selected file")
        
        # Create a secure filename
        filename = f"{uuid.uuid4()}_{file.filename}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        
        # Save the file
        try:
            with open(filepath, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            logger.info(f"File saved to {filepath}")
        except Exception as e:
            logger.error(f"File save error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Could not save file: {str(e)}")
        
        return await process_saved_file(filepath, file.filename, request)
    
    except HTTPException as he:
        logger.error(f"HTTP Exception: {he.detail}")
        raise he
    except Exception as e:
        logger.error(f"Unexpected error in upload: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.post("/process_dropbox_url")
async def process_dropbox_url(dropbox_request: UrlRequest, request: Request):
    try:
        url = dropbox_request.url
        logger.info(f"Processing Dropbox URL: {url}")
        
        # Validate URL
        if not url or "dropbox.com" not in url:
            raise HTTPException(status_code=400, detail="Invalid Dropbox URL")

        # Get filename from URL
        original_filename = get_filename_from_url(url)
        if not original_filename:
            raise HTTPException(status_code=400, detail="Could not extract filename from URL")
        
        # Create a secure filename for local storage
        filename = f"{uuid.uuid4()}_{original_filename}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        
        # Download the file from Dropbox
        try:
            await download_file_from_dropbox(url, filepath)
            logger.info(f"File downloaded from Dropbox and saved to {filepath}")
        except Exception as e:
            logger.error(f"Dropbox download error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Could not download file from Dropbox: {str(e)}")
        
        return await process_saved_file(filepath, original_filename, request)
        
    except HTTPException as he:
        logger.error(f"HTTP Exception: {he.detail}")
        raise he
    except Exception as e:
        logger.error(f"Unexpected error processing Dropbox URL: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.post("/process_youtube_url")
async def process_youtube_url(youtube_request: UrlRequest, request: Request):
    try:
        url = youtube_request.url
        logger.info(f"Processing YouTube URL: {url}")
        
        # Validate URL
        if not url or "youtube.com" not in url and "youtu.be" not in url:
            raise HTTPException(status_code=400, detail="Invalid YouTube URL")

        # Extract video ID from URL
        video_id = extract_video_id(url)
        if not video_id:
            raise HTTPException(status_code=400, detail="Could not extract video ID from URL")
        
        # Get video transcript
        try:
            transcript, video_title = await get_youtube_transcript(video_id)
            if not transcript:
                raise HTTPException(status_code=404, detail="No transcript available for this video")
                
            logger.info(f"Successfully retrieved transcript for video: {video_title}")
            
            # Create a filename for the transcript
            filename = f"youtube_{video_id}_{video_title}.txt"
            safe_filename = "".join([c if c.isalnum() or c in [' ', '.', '_', '-'] else '_' for c in filename])
            safe_filename = safe_filename[:100]  # Limit filename length
            
            # Save transcript to a file
            filepath = os.path.join(UPLOAD_FOLDER, safe_filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(transcript)
            logger.info(f"Transcript saved to: {filepath}")
            
            return await process_saved_file(filepath, f"YouTube: {video_title}", request)
        except Exception as e:
            logger.error(f"Transcript extraction error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Could not extract transcript: {str(e)}")
        
    except HTTPException as he:
        logger.error(f"HTTP Exception: {he.detail}")
        raise he
    except Exception as e:
        logger.error(f"Unexpected error processing YouTube URL: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.post("/process_generic_url")
async def process_generic_url(url_request: UrlRequest, request: Request):
    try:
        url = url_request.url
        logger.info(f"Processing generic URL: {url}")
        
        # Process the URL
        try:
            start_time = time.time()
            result = await process_any_url(url, UPLOAD_FOLDER)
            logger.info(f"URL processing result: {result}")
            
            # Route to the appropriate handler based on URL type
            if result["type"] == "youtube":
                # Redirect to YouTube handler
                return await process_youtube_url(url_request, request)
            elif result["type"] == "dropbox":
                # Redirect to Dropbox handler
                return await process_dropbox_url(url_request, request)
            elif result["type"] == "generic":
                # Process the downloaded generic file
                filepath = result["filepath"]
                original_filename = result["original_filename"]
                logger.info(f"Processing generic media file: {filepath}")
                # Special handling for media files
                if result.get("is_media", False):
                    logger.info(f"Processing generic media file: {filepath}")
                return await process_saved_file(filepath, f"URL: {original_filename}", request)
            else:
                raise HTTPException(status_code=400, detail="Unknown URL type")
        except Exception as e:
            logger.error(f"URL processing error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Could not process URL: {str(e)}")
        
    except HTTPException as he:
        logger.error(f"HTTP Exception: {he.detail}")
        raise he
    except Exception as e:
        logger.error(f"Unexpected error processing generic URL: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

async def process_saved_file(filepath: str, original_filename: str, request: Request):
    """Common function to process a saved file and update session"""
    try:
        # Check if this is an audio/video file for special handling
        file_extension = os.path.splitext(filepath)[1].lower()
        is_media_file = file_extension in ['.mp3', '.mp4', '.wav', '.avi', '.mkv', '.m4a', '.flac', '.ogg', '.webm']
        
        start_time = time.time()
        logger.info(f"Starting to process file: {filepath}, is_media: {is_media_file}")
        
        if is_media_file:
            logger.info(f"Detected media file: {filepath}, processing asynchronously")
            # Use ThreadPoolExecutor to avoid blocking the event loop with CPU-intensive tasks
            with concurrent.futures.ThreadPoolExecutor() as pool:
                content = await asyncio.get_event_loop().run_in_executor(
                    pool, process_file, filepath
                )
        else:
            # Process other file types normally
            content = process_file(filepath)
        
        logger.info(f"File processed successfully in {time.time() - start_time:.2f}s, content length: {len(content)}")
        
        # Save the content to a session-specific file to avoid session size limitations
        session_id = str(uuid.uuid4())
        session_file = os.path.join(SESSION_STORAGE, f"{session_id}.txt")
        with open(session_file, "w", encoding="utf-8") as f:
            f.write(content)
        
        # Store only the session ID and file info in the session
        request.session["session_id"] = session_id
        request.session["filename"] = original_filename
        logger.info(f"Session data stored with ID: {session_id}")
        
        # Log the entire session to debug
        logger.info(f"Session contents: {dict(request.session)}")
        
        return JSONResponse(
            content={
                "message": f"File {original_filename} processed successfully",
                "content_preview": content[:200] + "..." if len(content) > 200 else content,
                "session_id": session_id
            },
            headers={"Set-Cookie": "session_active=true; Path=/; SameSite=Lax"}
        )
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"File processing error: {str(e)}")

@app.get("/check_session", response_class=JSONResponse)
async def check_session(request: Request):
    session_data = dict(request.session)
    logger.info(f"Checking session: {session_data}")
    
    if "session_id" in request.session:
        return {"status": "active", "session_id": request.session["session_id"]}
    else:
        return {"status": "inactive"}

# Helper function to get session ID from either cookie or header
async def get_session_id(
    request: Request,
    api_key: str = Security(API_KEY_HEADER)
) -> str:
    """
    Get session ID from either the session cookie or the X-Session-ID header.
    Returns None if no session ID is found.
    """
    # First check for API key in header
    if api_key:
        return api_key
    
    # Then check for session in cookie
    if "session_id" in request.session:
        return request.session["session_id"]
    
    return None

@app.post("/query")
async def query(
    request: Request,
    session_id: str = Depends(get_session_id)
):
    try:
        # Log the request
        logger.info(f"Received query request with session ID: {session_id}")
        
        # Check if session_id exists
        if not session_id:
            logger.error("No session_id provided")
            raise HTTPException(status_code=400, detail="No session ID provided. Please upload a file first or provide a valid session ID.")
        
        session_file = os.path.join(SESSION_STORAGE, f"{session_id}.txt")
        
        # Check if session file exists
        if not os.path.exists(session_file):
            logger.error(f"Session file not found: {session_file}")
            raise HTTPException(status_code=400, detail="Session data not found, please upload file again")
        
        # Load file content from session file
        with open(session_file, "r", encoding="utf-8") as f:
            file_content = f.read()
            
        # Parse the request body
        body = await request.json()
        logger.info(f"Request body: {body}")
        
        if not body or "question" not in body:
            logger.error("No question in request body")
            raise HTTPException(status_code=400, detail="No question provided")
        
        user_question = body["question"]
        
        # Get filename from session if it exists
        filename = request.session.get("filename", "uploaded file") if hasattr(request, "session") else "uploaded file"
        
        # Check if this is an image file
        is_image = file_content.startswith("IMAGE_FILE:")
        if is_image:
            logger.info(f"Processing image question for file in session: {session_id}")
            # For images, the file_content contains the path to the image file
            image_path = file_content.replace("IMAGE_FILE:", "").strip()
            
            # Verify the image file exists
            if not os.path.exists(image_path):
                logger.error(f"Image file not found: {image_path}")
                raise HTTPException(status_code=404, detail="Image file not found")
        else:
            logger.info(f"Processing question about file in session: {session_id}")
        
        # Get response from Ollama
        try:
            response = get_ollama_response(user_question, file_content)
            logger.info("Got response from Ollama successfully")
            return JSONResponse(content={"response": response})
        except Exception as e:
            logger.error(f"Ollama API error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error getting response: {str(e)}")
    
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Unexpected error in query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.get("/api/health")
async def health_check():
    """Health check endpoint to verify API connectivity"""
    try:
        # Check if Ollama API is reachable
        from utils.ollama_client import OLLAMA_API_URL
        
        try:
            # Try to connect to Ollama API
            response = requests.get(f"{OLLAMA_API_URL}/api/version", timeout=500)
            ollama_status = "up" if response.status_code == 200 else "down"
            ollama_version = response.json().get("version", "unknown") if response.status_code == 200 else "unknown"
        except Exception as e:
            logger.error(f"Error connecting to Ollama API: {str(e)}")
            ollama_status = "down"
            ollama_version = str(e)
        
        return {
            "status": "healthy",
            "version": "1.0.0",
            "ollama_api": {
                "url": OLLAMA_API_URL,
                "status": ollama_status,
                "version": ollama_version
            },
            "environment": os.environ.get("ENVIRONMENT", "development")
        }
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"status": "unhealthy", "error": str(e)}
        )

# Add new endpoints for session management
@app.post("/api/create_session", response_class=JSONResponse)
async def create_session():
    """Create a new session and return its ID"""
    session_id = str(uuid.uuid4())
    return {"session_id": session_id, "message": "New session created successfully"}

@app.get("/api/session/{session_id}/status")
async def get_session_status(session_id: str):
    """Check if a specific session exists"""
    session_file = os.path.join(SESSION_STORAGE, f"{session_id}.txt")
    if os.path.exists(session_file):
        # Get basic info about the session file
        file_stats = os.stat(session_file)
        creation_time = file_stats.st_ctime
        size = file_stats.st_size
        
        return {
            "status": "active",
            "session_id": session_id,
            "created": creation_time,
            "size_bytes": size
        }
    else:
        return {"status": "inactive", "message": "Session not found"}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)