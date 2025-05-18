import librosa
import numpy as np
from pychord import Chord
import matplotlib
# Set non-interactive backend to avoid tkinter thread issues
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from collections import Counter

def detect_key(y, sr, hop_length=1024):
    """
    Detect the musical key of an audio track
    
    Args:
        y: Audio time series
        sr: Sampling rate
        hop_length: Number of samples between successive chroma frames
        
    Returns:
        Dictionary containing detected key information (root, mode, confidence)
    """
    # Extract chroma features
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    
    # Aggregate chroma features across time to get overall pitch content
    chroma_sum = np.sum(chroma, axis=1)
    chroma_normalized = chroma_sum / np.sum(chroma_sum)
    
    # Define key profiles for major and minor keys (Krumhansl-Schmuckler key profiles)
    # Values represent the hierarchy of importance for each pitch class in a given key
    major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    
    # Normalize profiles
    major_profile = major_profile / np.sum(major_profile)
    minor_profile = minor_profile / np.sum(minor_profile)
    
    # Compute correlations for all possible keys (12 major + 12 minor = 24 keys)
    key_correlations = []
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    # Calculate correlation for all major keys
    for i in range(12):
        # Shift the profile to match each possible key
        major_shifted = np.roll(major_profile, i)
        correlation = np.corrcoef(chroma_normalized, major_shifted)[0, 1]
        key_correlations.append((notes[i], 'major', correlation))
    
    # Calculate correlation for all minor keys
    for i in range(12):
        # Shift the profile to match each possible key
        minor_shifted = np.roll(minor_profile, i)
        correlation = np.corrcoef(chroma_normalized, minor_shifted)[0, 1]
        key_correlations.append((notes[i], 'minor', correlation))
    
    # Find the key with the highest correlation
    key_correlations.sort(key=lambda x: x[2], reverse=True)
    best_key = key_correlations[0]
    
    # Format the key name (e.g., "C major", "A minor")
    if best_key[1] == 'major':
        key_name = f"{best_key[0]}"
    else:
        key_name = f"{best_key[0]}m"
    
    # Get the top 3 key candidates for reference
    top_keys = key_correlations[:3]
    
    return {
        'key': key_name,
        'root': best_key[0],
        'mode': best_key[1],
        'confidence': best_key[2],
        'alternatives': [{'key': f"{k[0]}{'' if k[1]=='major' else 'm'}", 'confidence': k[2]} for k in top_keys[1:3]]
    }

def suggest_chord_progressions(key):
    """
    Generate common chord progressions based on the detected key
    
    Args:
        key: Dictionary containing key information
        
    Returns:
        Dictionary with suggested chord progressions
    """
    # Define root note and whether it's major or minor
    root = key['root']
    is_major = key['mode'] == 'major'
    
    # Note indices for building chord progressions
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    root_idx = notes.index(root)
    
    # Define scale degrees based on major or minor key
    if is_major:
        # Major scale: I, ii, iii, IV, V, vi, vii°
        scale_degrees = [
            (0, ''),    # I (major)
            (2, 'm'),   # ii (minor)
            (4, 'm'),   # iii (minor)
            (5, ''),    # IV (major)
            (7, ''),    # V (major)
            (9, 'm'),   # vi (minor)
            (11, 'm7')  # vii (minor 7th or diminished)
        ]
    else:
        # Natural minor scale: i, ii°, III, iv, v, VI, VII
        scale_degrees = [
            (0, 'm'),   # i (minor)
            (2, 'm7'),  # ii° (diminished, using m7 as approximation)
            (3, ''),    # III (major)
            (5, 'm'),   # iv (minor)
            (7, 'm'),   # v (minor) or (7, '') for V (major) in harmonic minor
            (8, ''),    # VI (major)
            (10, '')    # VII (major)
        ]
    
    # Generate actual chords based on the key
    chords = {}
    for degree, (interval, chord_type) in enumerate(scale_degrees, 1):
        # Calculate the note index, wrapping around if necessary
        note_idx = (root_idx + interval) % 12
        chord_name = f"{notes[note_idx]}{chord_type}"
        # Store as roman numeral -> chord name
        numeral = ['i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii'][degree-1]
        if is_major:
            if degree in [1, 4, 5]:  # Major chords in major key
                numeral = numeral.upper()
        else:
            if degree in [3, 6, 7]:  # Major chords in minor key
                numeral = numeral.upper()
        chords[numeral] = chord_name
    
    # Define common chord progressions using roman numerals
    common_progressions = [
        # Major key progressions
        {'name': 'I-IV-V', 'description': 'The most basic and common progression in popular music'},
        {'name': 'I-V-vi-IV', 'description': 'The "pop-punk" progression, used in countless pop songs'},
        {'name': 'ii-V-I', 'description': 'The jazz standard progression'},
        {'name': 'I-vi-IV-V', 'description': '50s progression, used in doo-wop and early rock and roll'},
        {'name': 'vi-IV-I-V', 'description': 'Minor variation of the pop progression, often used for emotional songs'},
        
        # Minor key progressions
        {'name': 'i-iv-v', 'description': 'Basic minor key progression'},
        {'name': 'i-VI-III-VII', 'description': 'Common minor progression in rock and film music'},
        {'name': 'i-iv-VII-III', 'description': 'Andalusian cadence, common in flamenco and rock music'},
        {'name': 'i-VII-VI-VII', 'description': 'Minor rock progression'},
        {'name': 'i-v-VI-VII', 'description': 'Natural minor key rock progression'}
    ]
    
    # Build actual chord progressions based on the key
    progressions = []
    
    # Choose progressions based on major/minor key
    if is_major:
        relevant_progressions = [p for p in common_progressions if p['name'][0].isupper() or p['name'][0] == 'i']
    else:
        relevant_progressions = [p for p in common_progressions if p['name'][0].islower() or p['name'][0] == 'I']
    
    # Convert roman numerals to actual chord names
    for prog in relevant_progressions:
        numerals = prog['name'].split('-')
        
        # Map each numeral to its corresponding chord
        chord_progression = []
        for numeral in numerals:
            # Handle numeral variations (case sensitivity)
            if numeral.lower() in chords:
                chord_progression.append(chords[numeral.lower()])
            else:
                # Try uppercase version as fallback
                chord_progression.append(chords.get(numeral.upper(), "?"))
        
        progressions.append({
            'name': prog['name'],
            'description': prog['description'],
            'chords': chord_progression
        })
    
    return {
        'key': key['key'],
        'progressions': progressions[:5]  # Return top 5 progressions
    }

def extract_chords(y, sr, hop_length=1024, n_chroma=12, min_duration=0.5):
    """
    Extract chords from audio data with beat synchronization and improved temporal stability
    
    Args:
        y: Audio time series
        sr: Sampling rate
        hop_length: Number of samples between successive chroma frames (increased for longer segments)
        n_chroma: Number of chroma bins
        min_duration: Minimum duration in seconds for a chord to be detected
        
    Returns:
        List of (chord, start_time, end_time) tuples
    """
    # 1. Extract beat information for musical structure alignment
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
    beat_times = librosa.frames_to_time(librosa.util.fix_frames(beats), sr=sr, hop_length=hop_length)
    
    # If no beats were detected, create artificial beats at regular intervals
    if len(beat_times) < 2:
        duration = librosa.get_duration(y=y, sr=sr)
        # Create artificial beats at 0.5 second intervals
        beat_times = np.arange(0, duration, 0.5)
    
    # 2. Extract chroma features using constant-Q transform for better pitch detection
    chroma = librosa.feature.chroma_cqt(
        y=y, 
        sr=sr, 
        hop_length=hop_length, 
        n_chroma=n_chroma,
        bins_per_octave=36  # Higher resolution for better chord detection
    )
    
    # 3. Compute beat-synchronized chroma features
    # This aligns the chroma features with the beat structure of the music
    beat_chroma = librosa.util.sync(chroma, beats, aggregate=np.median)
    
    # 4. Apply sophisticated filtering for noise reduction and temporal stability
    # First, apply median filtering to remove noise
    chroma_filtered = librosa.decompose.nn_filter(
        chroma,
        aggregate=np.median,
        metric='cosine'
    )
    
    # Then apply additional smoothing with a larger kernel for temporal stability
    # This helps detect chords that last longer by reducing local variations
    smoothing_size = 7  # Controls smoothing amount (larger = more smoothing)
    chroma_smooth = np.copy(chroma_filtered)
    for i in range(chroma_smooth.shape[1]):
        start = max(0, i - smoothing_size)
        end = min(chroma_smooth.shape[1], i + smoothing_size + 1)
        chroma_smooth[:, i] = np.mean(chroma_filtered[:, start:end], axis=1)
    
    # 5. Use enhanced chord templates including more chord types for accurate detection
    chord_templates = {
        'C': [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],    # C major
        'Cm': [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],   # C minor
        'C7': [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],   # C dominant 7th
        'Cmaj7': [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1], # C major 7th
        'Cm7': [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],   # C minor 7th
        'Csus4': [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0], # C suspended 4th
        'Csus2': [1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0]  # C suspended 2nd
    }
    
    # Generate all 12 chord roots with different qualities by cycling the templates
    all_templates = {}
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    for i, root in enumerate(notes):
        # Generate basic chords (major and minor)
        for chord_type, template in chord_templates.items():
            chord_name = chord_type.replace('C', root)
            all_templates[chord_name] = np.roll(template, i)
    
    # 6. Apply two-level chord detection:
    # First on beat-synchronized features to get the primary chord structure
    # Then refine with frame-level analysis
    
    # 6.1 Beat-level chord detection
    beat_chords = []
    beat_confidences = []
    
    for beat in range(beat_chroma.shape[1]):
        beat_feature = beat_chroma[:, beat]
        
        # Find best matching chord
        max_corr = -np.inf
        best_chord = None
        
        for chord_name, template in all_templates.items():
            corr = np.corrcoef(beat_feature, template)[0, 1]
            if corr > max_corr:
                max_corr = corr
                best_chord = chord_name
        
        beat_chords.append(best_chord)
        beat_confidences.append(max_corr)
    
    # 6.2 Frame-level chord detection (for more detailed analysis)
    frame_chords = []
    frame_confidences = []
    
    for frame in range(chroma_smooth.shape[1]):
        frame_feature = chroma_smooth[:, frame]
        
        # Find best matching chord
        max_corr = -np.inf
        best_chord = None
        
        for chord_name, template in all_templates.items():
            corr = np.corrcoef(frame_feature, template)[0, 1]
            if corr > max_corr:
                max_corr = corr
                best_chord = chord_name
        
        frame_chords.append(best_chord)
        frame_confidences.append(max_corr)
    
    # 7. Merge beat-level and frame-level analyses
    # Start with the beat boundaries for primary segmentation
    beat_segments = []
    
    for i in range(len(beat_times) - 1):
        chord = beat_chords[i]
        start_time = beat_times[i]
        end_time = beat_times[i + 1]
        confidence = beat_confidences[i]
        beat_segments.append((chord, start_time, end_time, confidence))
    
    # Add the last segment if needed
    if len(beat_times) > 0:
        last_beat = len(beat_times) - 1
        if last_beat < len(beat_chords):
            duration = librosa.get_duration(y=y, sr=sr)
            beat_segments.append((beat_chords[last_beat], beat_times[last_beat], duration, beat_confidences[last_beat]))
    
    # 8. Consolidate similar consecutive chords to create longer segments
    consolidated_segments = []
    
    if len(beat_segments) > 0:
        current_chord, current_start, current_end, current_conf = beat_segments[0]
        
        for i in range(1, len(beat_segments)):
            next_chord, next_start, next_end, next_conf = beat_segments[i]
            
            # If next chord is the same as current, extend the current segment
            if next_chord == current_chord:
                current_end = next_end
                # Update confidence using weighted average
                duration1 = next_start - current_start
                duration2 = next_end - next_start
                current_conf = (current_conf * duration1 + next_conf * duration2) / (duration1 + duration2)
            else:
                # Different chord, add current to results and start a new segment
                consolidated_segments.append((current_chord, current_start, current_end, current_conf))
                current_chord, current_start, current_end, current_conf = next_chord, next_start, next_end, next_conf
        
        # Add the final segment
        consolidated_segments.append((current_chord, current_start, current_end, current_conf))
    
    # 9. Apply intelligent segmentation to eliminate short segments
    # Minimum duration check
    duration = librosa.get_duration(y=y, sr=sr)
    
    # Handle potential empty segments list
    if not consolidated_segments:
        # Fallback: use frame-level analysis if beat analysis failed
        frame_duration = duration / len(frame_chords) if len(frame_chords) > 0 else 0
        
        if frame_duration > 0:
            segments = []
            current_chord = frame_chords[0]
            current_conf = frame_confidences[0]
            start_frame = 0
            
            for i in range(1, len(frame_chords)):
                if frame_chords[i] != current_chord:
                    start_time = start_frame * frame_duration
                    end_time = i * frame_duration
                    segments.append((current_chord, start_time, end_time, current_conf))
                    
                    current_chord = frame_chords[i]
                    current_conf = frame_confidences[i]
                    start_frame = i
            
            # Add the last chord
            if len(frame_chords) > 0:
                start_time = start_frame * frame_duration
                segments.append((current_chord, start_time, duration, current_conf))
            
            consolidated_segments = segments
    
    # 10. Final filtering to remove very short segments and handle low-confidence detections
    result = []
    
    for i, (chord, start, end, conf) in enumerate(consolidated_segments):
        segment_duration = end - start
        
        # Always keep segments longer than min_duration
        if segment_duration >= min_duration:
            result.append((chord, start, end))
        # For shorter segments, keep only if they have high confidence
        elif conf > 0.7:  # High confidence threshold
            result.append((chord, start, end))
        # Otherwise merge with neighbors if not at edges
        elif 0 < i < len(consolidated_segments) - 1:
            prev_conf = consolidated_segments[i-1][3]
            next_conf = consolidated_segments[i+1][3]
            
            # Decide whether to merge with previous or next segment
            if prev_conf > next_conf:
                # Extend previous segment
                prev_chord, prev_start, _, _ = consolidated_segments[i-1]
                result.append((prev_chord, prev_start, end))
            else:
                # Extend next segment
                next_chord, _, next_end, _ = consolidated_segments[i+1]
                result.append((next_chord, start, next_end))
        # Edge segments (first or last) - keep if decent confidence
        elif conf > 0.5:
            result.append((chord, start, end))
    
    # 11. Final post-processing: merge duplicate segments and ensure no time gaps
    if result:
        final_result = []
        current_chord, current_start, current_end = result[0]
        
        for i in range(1, len(result)):
            chord, start, end = result[i]
            
            # If same chord or extremely close timing, merge
            if chord == current_chord or abs(start - current_end) < 0.01:
                current_end = end
            else:
                final_result.append((current_chord, current_start, current_end))
                current_chord, current_start, current_end = chord, start, end
        
        # Add the last segment
        final_result.append((current_chord, current_start, current_end))
        
        return final_result
    else:
        # Fallback if all else fails: return a single chord for the whole duration
        if len(frame_chords) > 0:
            # Find the most common chord
            from collections import Counter
            most_common_chord = Counter(frame_chords).most_common(1)[0][0]
            return [(most_common_chord, 0, duration)]
        else:
            # Complete fallback
            return [('C', 0, duration)]  # Default if nothing else works

def create_chord_diagram(chord_name, output_file, left_handed=False):
    """
    Create a visual diagram for a chord
    
    Args:
        chord_name: Name of the chord (e.g., 'C', 'Am')
        output_file: Path to save the diagram
        left_handed: If True, create a mirrored diagram for left-handed players
        
    Returns:
        Path to the saved diagram
    """
    try:
        # Create chord from name
        chord = Chord(chord_name)
        
        # Get chord components
        root = chord.root
        components = chord.components()
        
        # Define fretboard
        frets = 5
        strings = 6
        
        # Predefined chord shapes with finger positions for all common chords
        # Format: {chord_name: [(string, fret, finger), ...]}
        common_shapes = {
            # Major chords
            'C': [(2, 0, 0), (1, 3, 3), (0, 2, 2), (4, 1, 1), (3, 0, 0), (5, 0, 0)],  # (string, fret, finger)
            'G': [(0, 3, 2), (1, 2, 1), (2, 0, 0), (3, 0, 0), (4, 0, 0), (5, 3, 3)],
            'D': [(0, 2, 1), (1, 3, 3), (2, 2, 2), (3, 0, 0), (4, -1, 0), (5, -1, 0)],
            'A': [(0, 0, 0), (1, 2, 2), (2, 2, 3), (3, 2, 4), (4, 0, 0), (5, -1, 0)],
            'E': [(0, 0, 0), (1, 0, 0), (2, 1, 1), (3, 2, 3), (4, 2, 2), (5, 0, 0)],
            'F': [(0, 1, 1), (1, 1, 1), (2, 2, 2), (3, 3, 4), (4, 3, 3), (5, 1, 1)],
            'B': [(0, 2, 1), (1, 4, 3), (2, 4, 4), (3, 4, 4), (4, 2, 1), (5, -1, 0)],
            'Csharp': [(0, 4, 3), (1, 6, 4), (2, 6, 4), (3, 6, 4), (4, 4, 1), (5, -1, 0)],
            'Dsharp': [(0, 6, 1), (1, 8, 3), (2, 8, 4), (3, 7, 2), (4, -1, 0), (5, -1, 0)],
            'Fsharp': [(0, 2, 1), (1, 2, 1), (2, 3, 3), (3, 4, 4), (4, 4, 4), (5, 2, 1)],
            'Gsharp': [(0, 4, 3), (1, 3, 2), (2, 1, 1), (3, 1, 1), (4, 1, 1), (5, 4, 4)],
            
            # Minor chords
            'Am': [(0, 0, 0), (1, 1, 1), (2, 2, 2), (3, 2, 3), (4, 0, 0), (5, -1, 0)],
            'Em': [(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 2, 2), (4, 2, 1), (5, 0, 0)],
            'Dm': [(0, 1, 1), (1, 3, 3), (2, 2, 2), (3, 0, 0), (4, -1, 0), (5, -1, 0)],
            'Bm': [(0, 2, 1), (1, 3, 3), (2, 4, 4), (3, 4, 4), (4, 2, 1), (5, -1, 0)],
            'Fm': [(0, 1, 1), (1, 1, 1), (2, 1, 1), (3, 3, 3), (4, 3, 4), (5, 1, 1)],
            'Gm': [(0, 3, 2), (1, 3, 3), (2, 3, 4), (3, 0, 0), (4, 0, 0), (5, 3, 1)],
            'Cm': [(0, 3, 1), (1, 4, 2), (2, 5, 4), (3, 5, 3), (4, 3, 1), (5, -1, 0)],
            'Csharpm': [(0, 4, 1), (1, 5, 2), (2, 6, 4), (3, 6, 3), (4, 4, 1), (5, -1, 0)],
            'Dsharpm': [(0, 6, 1), (1, 7, 2), (2, 8, 4), (3, 8, 3), (4, 6, 1), (5, -1, 0)],
            'Fsharpm': [(0, 2, 1), (1, 2, 1), (2, 2, 1), (3, 4, 3), (4, 4, 4), (5, 2, 1)],
            'Gsharpm': [(0, 4, 1), (1, 4, 1), (2, 4, 1), (3, 6, 3), (4, 6, 4), (5, 4, 1)],
            'Asharpm': [(0, 1, 1), (1, 2, 2), (2, 3, 3), (3, 3, 4), (4, 1, 1), (5, -1, 0)],
            
            # 7th chords
            'C7': [(0, 0, 0), (1, 1, 1), (2, 3, 3), (3, 2, 2), (4, 3, 4), (5, -1, 0)],
            'D7': [(0, 2, 1), (1, 1, 1), (2, 2, 2), (3, 0, 0), (4, -1, 0), (5, -1, 0)],
            'G7': [(0, 1, 1), (1, 0, 0), (2, 0, 0), (3, 0, 0), (4, 0, 0), (5, 3, 3)],
            'A7': [(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 2, 2), (4, 0, 0), (5, -1, 0)],
            'E7': [(0, 0, 0), (1, 0, 0), (2, 1, 1), (3, 0, 0), (4, 2, 2), (5, 0, 0)],
            'B7': [(0, 2, 2), (1, 0, 0), (2, 2, 3), (3, 1, 1), (4, 2, 4), (5, -1, 0)],
            'F7': [(0, 1, 1), (1, 1, 1), (2, 2, 2), (3, 1, 1), (4, 3, 3), (5, -1, 0)],
            'Dsharp7': [(0, 6, 2), (1, 4, 1), (2, 6, 3), (3, 5, 1), (4, 6, 4), (5, -1, 0)],
            
            # Major 7th chords
            'Cmaj7': [(0, 0, 0), (1, 3, 3), (2, 0, 0), (3, 2, 2), (4, 1, 1), (5, -1, 0)],
            'Dmaj7': [(0, 2, 1), (1, 2, 1), (2, 2, 1), (3, 0, 0), (4, -1, 0), (5, -1, 0)],
            'Gmaj7': [(0, 3, 2), (1, 2, 1), (2, 0, 0), (3, 0, 0), (4, 0, 0), (5, 2, 1)],
            'Amaj7': [(0, 0, 0), (1, 2, 2), (2, 1, 1), (3, 2, 3), (4, 0, 0), (5, -1, 0)],
            'Emaj7': [(0, 0, 0), (1, 0, 0), (2, 1, 1), (3, 1, 1), (4, 2, 3), (5, 0, 0)],
            'Bmaj7': [(0, 2, 1), (1, 4, 3), (2, 3, 2), (3, 4, 4), (4, 2, 1), (5, -1, 0)],
            'Fmaj7': [(0, 0, 0), (1, 1, 1), (2, 2, 2), (3, 3, 4), (4, 3, 3), (5, -1, 0)],
            'Csharpmaj7': [(0, 4, 1), (1, 6, 3), (2, 5, 2), (3, 6, 4), (4, 4, 1), (5, -1, 0)],
            'Dsharpmaj7': [(0, 6, 1), (1, 8, 3), (2, 7, 2), (3, 8, 4), (4, 6, 1), (5, -1, 0)],
            'Fsharpmaj7': [(0, 2, 1), (1, 3, 2), (2, 3, 3), (3, 3, 3), (4, 4, 4), (5, -1, 0)],
            'Gsharpmaj7': [(0, 4, 1), (1, 5, 2), (2, 5, 3), (3, 5, 3), (4, 6, 4), (5, -1, 0)],
            'Asharpmaj7': [(0, 6, 1), (1, 7, 2), (2, 7, 3), (3, 7, 3), (4, 8, 4), (5, -1, 0)],
            
            # Minor 7th chords
            'Am7': [(0, 0, 0), (1, 1, 1), (2, 0, 0), (3, 2, 3), (4, 0, 0), (5, -1, 0)],
            'Em7': [(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0), (4, 2, 2), (5, 0, 0)],
            'Dm7': [(0, 1, 1), (1, 1, 1), (2, 2, 3), (3, 0, 0), (4, -1, 0), (5, -1, 0)],
            'Gm7': [(0, 3, 1), (1, 3, 1), (2, 3, 1), (3, 3, 1), (4, 5, 4), (5, 3, 1)],
            'Cm7': [(0, 3, 1), (1, 4, 2), (2, 3, 1), (3, 5, 4), (4, 3, 1), (5, -1, 0)],
            'Asharpm7': [(0, 0, 0), (1, 2, 2), (2, 0, 0), (3, 3, 3), (4, 1, 1), (5, -1, 0)],
            
            # Suspended chords
            'Csus2': [(0, 3, 2), (1, 3, 3), (2, 0, 0), (3, 0, 0), (4, 1, 1), (5, -1, 0)],
            'Csus4': [(0, 3, 3), (1, 3, 2), (2, 0, 0), (3, 1, 1), (4, 1, 1), (5, -1, 0)],
            'Dsus4': [(0, 0, 0), (1, 3, 3), (2, 2, 2), (3, 0, 0), (4, -1, 0), (5, -1, 0)],
            'Esus4': [(0, 0, 0), (1, 0, 0), (2, 2, 2), (3, 2, 3), (4, 2, 1), (5, 0, 0)],
            'Fsus4': [(0, 1, 1), (1, 1, 1), (2, 3, 3), (3, 3, 4), (4, 3, 4), (5, 1, 1)],
            'Csharpsus2': [(0, 4, 2), (1, 4, 3), (2, 1, 1), (3, 1, 1), (4, 1, 1), (5, -1, 0)],
            'Dsharpsus2': [(0, 6, 2), (1, 6, 3), (2, 3, 1), (3, 3, 1), (4, 4, 1), (5, -1, 0)],
            'Dsharpsus4': [(0, 1, 1), (1, 4, 3), (2, 3, 2), (3, 1, 1), (4, -1, 0), (5, -1, 0)]
        }
        
        # Guitar strings in standard tuning (E A D G B E)
        open_strings = [40, 45, 50, 55, 59, 64]  # MIDI note numbers
        
        # Create chord shape
        chord_shape = []
        finger_assignments = {}
        
        # Check if this is a common chord with predefined shape
        if chord_name in common_shapes:
            # Use predefined shape with finger positions
            for string, fret, finger in common_shapes[chord_name]:
                chord_shape.append((string, fret))
                if finger > 0:  # Only assign fingers to fretted notes (not open strings or muted)
                    finger_assignments[(string, fret)] = finger
        else:
            # Fallback to algorithmic approach for uncommon chords
            for string in range(strings):
                open_note = open_strings[string] % 12  # Get note value (0-11)
                
                # Look for components on this string
                found = False
                for fret in range(frets + 1):  # +1 to include open strings
                    fret_note = (open_note + fret) % 12
                    if fret_note in components:
                        chord_shape.append((string, fret))
                        found = True
                        break
                
                # If no component found on this string, mark as muted
                if not found:
                    chord_shape.append((string, -1))  # -1 indicates muted string
            
            # Assign finger numbers for algorithmic chords
            sorted_positions = sorted([(s, f) for s, f in chord_shape if f > 0], key=lambda x: x[1])
            finger_count = 1
            for string, fret in sorted_positions:
                finger_assignments[(string, fret)] = finger_count
                finger_count = min(finger_count + 1, 4)
        
        # Create the chord diagram with extra space for legend
        fig, ax = plt.subplots(figsize=(4, 7))
        
        # Draw fretboard
        for i in range(frets + 1):
            ax.axhline(i, color='black', linewidth=1)
        
        for i in range(strings):
            ax.axvline(i, color='black', linewidth=1)
            
        # Define colors for each finger
        finger_colors = {
            1: 'red',     # index finger
            2: 'green',   # middle finger
            3: 'blue',    # ring finger
            4: 'purple'   # pinky
        }
        
        # Draw chord dots with finger numbers
        for string, fret in chord_shape:
            # For left-handed diagrams, mirror the string position
            display_string = strings - 1 - string if left_handed else string
            
            if fret > 0:  # Fretted note
                # Get finger number and choose appropriate color
                finger_num = finger_assignments.get((string, fret), '')
                color = finger_colors.get(finger_num, 'black')
                
                # Draw colored dot for finger position
                ax.plot(display_string, fret, 'o', markersize=15, color=color)
                
                # Add finger number
                ax.text(display_string, fret, str(finger_num), color='white', fontsize=10, 
                        ha='center', va='center', fontweight='bold')
            elif fret == 0:  # Open string
                ax.plot(display_string, fret, 'o', markersize=10, color='lightgray', alpha=0.7)
            else:  # Muted string
                ax.text(display_string, -0.3, 'x', fontsize=15, ha='center')
        
        # Add title
        title = f"{chord_name} Chord"
        if left_handed:
            title += " (Left-Handed)"
        ax.set_title(title)
        
        # Set axis properties
        ax.set_xlim(-0.5, strings - 0.5)
        ax.set_ylim(frets, -1)
        ax.set_xticks([])
        ax.set_yticks(range(frets + 1))
        
        # Remove spines
        for spine in ax.spines.values():
            spine.set_visible(False)
            
        # Add legend for finger colors
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Index (1)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Middle (2)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Ring (3)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='purple', markersize=10, label='Pinky (4)')
        ]
        ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=2)
        
        # Save the diagram with extra padding for the legend
        plt.tight_layout()
        plt.savefig(output_file, bbox_inches='tight', pad_inches=0.5)
        plt.close()
        
        return output_file
    except Exception as e:
        print(f"Error creating chord diagram: {str(e)}")
        return None

def generate_waveform_image(y, sr, output_file, figsize=(12, 3)):
    """
    Generate a waveform visualization of audio data
    
    Args:
        y: Audio time series
        sr: Sampling rate
        output_file: Path to save the waveform image
        figsize: Size of the figure (width, height)
        
    Returns:
        Path to the saved waveform image
    """
    try:
        # Create figure
        plt.figure(figsize=figsize)
        
        # Plot waveform
        plt.plot(np.linspace(0, len(y)/sr, len(y)), y, color='#3498db', linewidth=0.5)
        
        # Add labels and title
        plt.title('Audio Waveform')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Amplitude')
        
        # Add grid and tight layout
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save figure
        plt.savefig(output_file)
        plt.close()
        
        return output_file
    except Exception as e:
        print(f"Error creating waveform image: {str(e)}")
        return None

def generate_chord_progression_preview(chords, output_dir='static/chord_previews', include_left_handed=False):
    """
    Generate visual chord diagrams for a chord progression
    
    Args:
        chords: List of chord names
        output_dir: Directory to save the diagrams
        include_left_handed: If True, also generate left-handed versions
        
    Returns:
        Dictionary of chord names to diagram paths
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create left-handed directory if needed
    left_handed_dir = os.path.join(output_dir, 'left_handed')
    if include_left_handed and not os.path.exists(left_handed_dir):
        os.makedirs(left_handed_dir)
    
    # Generate diagrams for each unique chord
    unique_chords = set([chord for chord, _, _ in chords])
    diagram_paths = {}
    
    for chord_name in unique_chords:
        # Generate right-handed (standard) diagram
        output_file = os.path.join(output_dir, f"{chord_name.replace('#', 'sharp')}.png")
        diagram_path = create_chord_diagram(chord_name, output_file)
        if diagram_path:
            diagram_paths[chord_name] = diagram_path
        
        # Generate left-handed diagram if requested
        if include_left_handed:
            left_handed_output_file = os.path.join(left_handed_dir, f"{chord_name.replace('#', 'sharp')}.png")
            left_handed_diagram_path = create_chord_diagram(chord_name, left_handed_output_file, left_handed=True)
            if left_handed_diagram_path:
                diagram_paths[f"{chord_name}_left_handed"] = left_handed_diagram_path
    
    return diagram_paths