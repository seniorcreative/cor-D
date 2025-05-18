from flask import Flask, render_template, request, jsonify, url_for, flash, redirect, send_file
import os
import json
import shutil
from youtube_utils import download_youtube_audio, load_and_analyze_audio
from chord_analysis import extract_chords, generate_chord_progression_preview, generate_waveform_image
import librosa

app = Flask(__name__)
app.secret_key = 'chord_analyzer_secret_key'

# Create directories
os.makedirs('downloads', exist_ok=True)
os.makedirs('static/chord_previews', exist_ok=True)
os.makedirs('static/waveforms', exist_ok=True)
os.makedirs('static/audio', exist_ok=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    video_url = request.form.get('video_url')
    min_duration = request.form.get('min_duration', '0.5')
    left_handed = request.form.get('left_handed') == 'true'
    
    # Validate and convert min_duration to float
    try:
        min_duration = float(min_duration)
        if min_duration < 0.1:
            min_duration = 0.1
        elif min_duration > 5.0:
            min_duration = 5.0
    except (ValueError, TypeError):
        min_duration = 0.5  # Default if invalid
    
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
        
        # Extract chords with the specified minimum duration
        chords = extract_chords(y, sr, min_duration=min_duration)
        
        # Generate chord diagrams (including left-handed if requested)
        diagram_paths = generate_chord_progression_preview(chords, include_left_handed=left_handed)
        
        # Generate waveform visualization
        video_id = video_url.split('v=')[1] if 'v=' in video_url else 'video'
        waveform_file = os.path.join('static', 'waveforms', f"{video_id}_waveform.png")
        waveform_path = generate_waveform_image(y, sr, waveform_file)
        
        # Copy the audio file to static directory for web playback
        audio_web_path = os.path.join('static', 'audio', f"{video_id}.mp3")
        shutil.copy2(audio_file, audio_web_path)
        
        # Format results for template
        results = []
        for chord_name, start_time, end_time in chords:
            # Get standard chord diagram
            diagram_path = None
            if chord_name in diagram_paths:
                diagram_path = os.path.basename(diagram_paths[chord_name])
            
            # Get left-handed chord diagram if available
            left_handed_diagram_path = None
            if left_handed and f"{chord_name}_left_handed" in diagram_paths:
                left_handed_diagram_path = os.path.join('left_handed', os.path.basename(diagram_paths[f"{chord_name}_left_handed"]))
            
            results.append({
                'chord': chord_name,
                'start_time': round(start_time, 2),
                'end_time': round(end_time, 2),
                'duration': round(end_time - start_time, 2),
                'diagram_path': diagram_path,
                'left_handed_diagram_path': left_handed_diagram_path
            })
        
        # Add analysis metadata
        analysis_info = {
            "min_duration": min_duration,
            "total_chords": len(chords),
            "avg_chord_duration": round(sum(end - start for _, start, end in chords) / max(1, len(chords)), 2),
            "left_handed": left_handed,
            "waveform_path": os.path.join('waveforms', f"{video_id}_waveform.png"),
            "audio_path": os.path.join('audio', f"{video_id}.mp3"),
            "duration": librosa.get_duration(y=y, sr=sr)
        }
        
        # Save results for later reference
        video_id = video_url.split('v=')[1] if 'v=' in video_url else 'video'
        result_file = os.path.join('static', f"{video_id}_chords.json")
        with open(result_file, 'w') as f:
            json.dump(results, f)
        
        return render_template('results.html', 
                              results=results, 
                              video_url=video_url, 
                              metadata=metadata, 
                              analysis_info=analysis_info)
    
    except Exception as e:
        flash(f'Error analyzing the video: {str(e)}', 'error')
        return redirect(url_for('home'))

@app.route('/api/status')
def status():
    return jsonify({"status": "online"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)