# YouTube Music Chord Analyzer

A web application that analyzes YouTube videos for their music content and generates chord previews and diagrams.

## Features

- Download and extract audio from YouTube videos
- Analyze music to detect chords and progressions
- Generate visual chord diagrams
- Display chord progression timeline
- Detailed chord list with timing information

## Requirements

- Python 3.11+
- Flask
- youtube-dl
- librosa
- pychord
- matplotlib
- FFmpeg

## Installation

1. Clone the repository:
   ```
   git clone [repository-url]
   cd youtube-music-chord-analyzer
   ```

2. Create a virtual environment and activate it:
   ```
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install the required packages:
   ```
   pip install youtube-dl librosa Flask pychord matplotlib
   ```

4. Make sure FFmpeg is installed on your system:
   ```
   sudo apt-get install ffmpeg
   ```

## Usage

1. Start the application:
   ```
   ./run.sh
   ```
   
   Alternatively:
   ```
   source venv/bin/activate
   python app.py
   ```

2. Open your browser and navigate to `http://localhost:5000`

3. Enter a YouTube URL containing music you want to analyze

4. The application will download the audio, analyze it, and generate chord diagrams

## Project Structure

- `app.py`: Main Flask application file
- `youtube_utils.py`: Functions for downloading and processing YouTube videos
- `chord_analysis.py`: Functions for music analysis and chord detection
- `templates/`: HTML templates for the web interface
- `static/chord_previews/`: Generated chord diagrams
- `downloads/`: Temporary storage for downloaded audio files

## How It Works

1. The application extracts audio from YouTube videos using youtube-dl
2. The audio is analyzed using librosa to detect musical features
3. Chord detection is performed on the audio data
4. Visual chord diagrams are generated using matplotlib and pychord
5. Results are displayed in a user-friendly web interface

## Limitations

- Basic chord detection may not be accurate for complex musical pieces
- Analysis works best on clear recordings with minimal background noise
- Processing time increases with video length

## Future Improvements

- Improved chord detection algorithms
- Support for more advanced music theory analysis
- Ability to save and share analyses
- User accounts for tracking analysis history