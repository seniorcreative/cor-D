from flask import Flask, render_template, request, jsonify, url_for, flash, redirect
import os
import json
from youtube_utils import download_youtube_audio, load_and_analyze_audio
from chord_analysis import extract_chords, generate_chord_progression_preview

app = Flask(__name__)
app.secret_key = 'chord_analyzer_secret_key'

# Create directories
os.makedirs('downloads', exist_ok=True)
os.makedirs('static/chord_previews', exist_ok=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    video_url = request.form.get('video_url')
    
    if not video_url:
        flash('Please enter a YouTube URL', 'error')
        return redirect(url_for('home'))
    
    try:
        # Download audio from YouTube
        audio_file, metadata = download_youtube_audio(video_url)
        
        if not audio_file:
            flash('Failed to download the YouTube video. Please check that the URL is valid and the video is available.', 'error')
            return redirect(url_for('home'))
        
        # Load and analyze audio
        y, sr = load_and_analyze_audio(audio_file)
        
        if y is None or sr is None:
            flash('Failed to analyze the audio', 'error')
            return redirect(url_for('home'))
        
        # Extract chords
        chords = extract_chords(y, sr)
        
        # Generate chord diagrams
        diagram_paths = generate_chord_progression_preview(chords)
        
        # Format results for template
        results = []
        for chord_name, start_time, end_time in chords:
            diagram_path = None
            if chord_name in diagram_paths:
                diagram_path = os.path.basename(diagram_paths[chord_name])
            
            results.append({
                'chord': chord_name,
                'start_time': round(start_time, 2),
                'end_time': round(end_time, 2),
                'duration': round(end_time - start_time, 2),
                'diagram_path': diagram_path
            })
        
        # Save results for later reference
        video_id = video_url.split('v=')[1] if 'v=' in video_url else 'video'
        result_file = os.path.join('static', f"{video_id}_chords.json")
        with open(result_file, 'w') as f:
            json.dump(results, f)
        
        return render_template('results.html', results=results, video_url=video_url, metadata=metadata)
    
    except Exception as e:
        flash(f'Error analyzing the video: {str(e)}', 'error')
        return redirect(url_for('home'))

@app.route('/api/status')
def status():
    return jsonify({"status": "online"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)