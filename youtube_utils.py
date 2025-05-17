import os
import yt_dlp as youtube_dl  # Using yt-dlp instead of youtube-dl
import librosa
import numpy as np

def download_youtube_audio(video_url, output_dir='downloads'):
    """
    Download audio from a YouTube video and extract metadata
    
    Args:
        video_url: URL of the YouTube video
        output_dir: Directory to store downloaded files
        
    Returns:
        Tuple containing:
        - Path to the downloaded audio file
        - Dictionary with video metadata (title, artist, duration, upload_date)
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Options for youtube-dl
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),
        'prefer_ffmpeg': True,
        'quiet': True,
        'no_warnings': True
    }
    
    try:
        # Download the video
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(video_url, download=True)
            video_title = info_dict['title']
            audio_file = os.path.join(output_dir, f"{video_title}.wav")
            
            # Extract metadata
            metadata = {
                'title': info_dict.get('title', 'Unknown'),
                'artist': info_dict.get('artist', 'Unknown'),
                'uploader': info_dict.get('uploader', 'Unknown'),
                'duration': info_dict.get('duration', 0),
                'upload_date': info_dict.get('upload_date', 'Unknown'),
            }
            
            # Format duration as minutes:seconds
            if metadata['duration']:
                minutes = int(metadata['duration'] // 60)
                seconds = int(metadata['duration'] % 60)
                metadata['duration_formatted'] = f"{minutes}:{seconds:02d}"
            else:
                metadata['duration_formatted'] = "Unknown"
                
            # Format upload date (YYYYMMDD to YYYY)
            if metadata['upload_date'] and metadata['upload_date'] != 'Unknown':
                metadata['year'] = metadata['upload_date'][:4]
            else:
                metadata['year'] = 'Unknown'
                
            return audio_file, metadata
    except youtube_dl.utils.DownloadError as e:
        print(f"YouTube download error: {str(e)}")
        return None, None
    except youtube_dl.utils.ExtractorError as e:
        print(f"YouTube extractor error: {str(e)}")
        return None, None
    except Exception as e:
        print(f"Error downloading YouTube video: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        return None, None

def load_and_analyze_audio(audio_file):
    """
    Load and prepare audio data for analysis
    
    Args:
        audio_file: Path to the audio file
        
    Returns:
        Tuple of (audio data, sampling rate)
    """
    try:
        # Load audio file
        y, sr = librosa.load(audio_file, sr=None)
        return y, sr
    except Exception as e:
        print(f"Error loading audio: {str(e)}")
        return None, None