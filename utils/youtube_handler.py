import re
import aiohttp
import logging
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound

logger = logging.getLogger(__name__)

def extract_video_id(url):
    """Extract YouTube video ID from URL"""
    try:
        # Try to extract from standard YouTube URL format
        pattern = r'(?:youtube\.com\/(?:[^\/]+\/.+\/|(?:v|e(?:mbed)?)\/|.*[?&]v=)|youtu\.be\/)([^"&?\/\s]{11})'
        match = re.search(pattern, url)
        
        if match:
            return match.group(1)
        else:
            logger.error(f"Could not extract video ID from URL: {url}")
            return None
    except Exception as e:
        logger.error(f"Error extracting video ID: {str(e)}")
        return None

async def get_youtube_title(video_id):
    """Get the title of a YouTube video"""
    try:
        # Use oEmbed API to get video information without API key
        oembed_url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(oembed_url) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("title", f"YouTube Video {video_id}")
                else:
                    logger.warning(f"Could not get video title, status code: {response.status}")
                    return f"YouTube Video {video_id}"
    except Exception as e:
        logger.error(f"Error getting video title: {str(e)}")
        return f"YouTube Video {video_id}"

async def get_youtube_transcript(video_id):
    """Get transcript from YouTube video"""
    try:
        # Get transcript using youtube_transcript_api
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        
        # Format the transcript with timestamps
        formatted_transcript = []
        for item in transcript_list:
            start_time = item['start']
            minutes = int(start_time // 60)
            seconds = int(start_time % 60)
            formatted_time = f"[{minutes:02d}:{seconds:02d}]"
            formatted_transcript.append(f"{formatted_time} {item['text']}")
        
        transcript_text = "\n".join(formatted_transcript)
        
        # Get the video title
        video_title = await get_youtube_title(video_id)
        
        return transcript_text, video_title
        
    except (TranscriptsDisabled, NoTranscriptFound) as e:
        logger.error(f"No transcript available: {str(e)}")
        raise ValueError(f"No transcript available for this video: {str(e)}")
    except Exception as e:
        logger.error(f"Error getting transcript: {str(e)}")
        raise ValueError(f"Error extracting transcript: {str(e)}")
