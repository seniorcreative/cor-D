import librosa
import numpy as np
from pychord import Chord
import matplotlib.pyplot as plt
import os

def extract_chords(y, sr, hop_length=512, n_chroma=12):
    """
    Extract chords from audio data
    
    Args:
        y: Audio time series
        sr: Sampling rate
        hop_length: Number of samples between successive chroma frames
        n_chroma: Number of chroma bins
        
    Returns:
        List of (chord, start_time, end_time) tuples
    """
    # Extract chroma features
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length, n_chroma=n_chroma)
    
    # Apply median filtering to remove noise
    chroma_filtered = librosa.decompose.nn_filter(
        chroma,
        aggregate=np.median,
        metric='cosine'
    )
    
    # We'll use a simple approach here - for each frame, find the chord that best matches the chroma
    # In a real application, you'd want to use a more advanced chord detection algorithm
    
    # Map of chord roots and quality
    chord_templates = {
        'C': [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],  # C major
        'Cm': [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]  # C minor
    }
    
    # Generate all 12 major and minor chords by cycling the templates
    all_templates = {}
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    for i, root in enumerate(notes):
        # Generate major chord
        template = np.roll(chord_templates['C'], i)
        all_templates[f"{root}"] = template
        
        # Generate minor chord
        template = np.roll(chord_templates['Cm'], i)
        all_templates[f"{root}m"] = template
    
    # Detect chords for each frame
    frames = chroma_filtered.shape[1]
    chord_frames = []
    
    for frame in range(frames):
        frame_chroma = chroma_filtered[:, frame]
        
        # Find best matching chord
        max_corr = -np.inf
        best_chord = None
        
        for chord_name, template in all_templates.items():
            corr = np.corrcoef(frame_chroma, template)[0, 1]
            if corr > max_corr:
                max_corr = corr
                best_chord = chord_name
        
        chord_frames.append(best_chord)
    
    # Convert frames to time segments
    duration = librosa.get_duration(y=y, sr=sr)
    frame_duration = duration / frames
    
    # Simplify by merging consecutive identical chords
    result = []
    current_chord = chord_frames[0]
    start_frame = 0
    
    for i in range(1, len(chord_frames)):
        if chord_frames[i] != current_chord:
            start_time = start_frame * frame_duration
            end_time = i * frame_duration
            result.append((current_chord, start_time, end_time))
            
            current_chord = chord_frames[i]
            start_frame = i
    
    # Add the last chord
    start_time = start_frame * frame_duration
    end_time = frames * frame_duration
    result.append((current_chord, start_time, end_time))
    
    return result

def create_chord_diagram(chord_name, output_file):
    """
    Create a visual diagram for a chord
    
    Args:
        chord_name: Name of the chord (e.g., 'C', 'Am')
        output_file: Path to save the diagram
        
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
        
        # Instead of algorithmic approach, use predefined chord shapes for common chords
        # Format: {chord_name: [(string, fret, finger), ...]}
        common_shapes = {
            'C': [(2, 0, 0), (1, 3, 3), (0, 2, 2), (4, 1, 1), (3, 0, 0), (5, 0, 0)],  # (string, fret, finger)
            'G': [(0, 3, 2), (1, 2, 1), (2, 0, 0), (3, 0, 0), (4, 0, 0), (5, 3, 3)],
            'D': [(0, 2, 1), (1, 3, 3), (2, 2, 2), (3, 0, 0), (4, -1, 0), (5, -1, 0)],
            'A': [(0, 0, 0), (1, 2, 2), (2, 2, 3), (3, 2, 4), (4, 0, 0), (5, -1, 0)],
            'E': [(0, 0, 0), (1, 0, 0), (2, 1, 1), (3, 2, 3), (4, 2, 2), (5, 0, 0)],
            'Am': [(0, 0, 0), (1, 1, 1), (2, 2, 2), (3, 2, 3), (4, 0, 0), (5, -1, 0)],
            'Em': [(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 2, 2), (4, 2, 1), (5, 0, 0)],
            'F': [(0, 1, 1), (1, 1, 1), (2, 2, 2), (3, 3, 4), (4, 3, 3), (5, 1, 1)]
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
            if fret > 0:  # Fretted note
                # Get finger number and choose appropriate color
                finger_num = finger_assignments.get((string, fret), '')
                color = finger_colors.get(finger_num, 'black')
                
                # Draw colored dot for finger position
                ax.plot(string, fret, 'o', markersize=15, color=color)
                
                # Add finger number
                ax.text(string, fret, str(finger_num), color='white', fontsize=10, 
                        ha='center', va='center', fontweight='bold')
            elif fret == 0:  # Open string
                ax.plot(string, fret, 'o', markersize=10, color='lightgray', alpha=0.7)
            else:  # Muted string
                ax.text(string, -0.3, 'x', fontsize=15, ha='center')
        
        # Add title
        ax.set_title(f"{chord_name} Chord")
        
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

def generate_chord_progression_preview(chords, output_dir='static/chord_previews'):
    """
    Generate visual chord diagrams for a chord progression
    
    Args:
        chords: List of chord names
        output_dir: Directory to save the diagrams
        
    Returns:
        List of paths to the saved diagrams
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Generate diagrams for each unique chord
    unique_chords = set([chord for chord, _, _ in chords])
    diagram_paths = {}
    
    for chord_name in unique_chords:
        output_file = os.path.join(output_dir, f"{chord_name.replace('#', 'sharp')}.png")
        diagram_path = create_chord_diagram(chord_name, output_file)
        if diagram_path:
            diagram_paths[chord_name] = diagram_path
    
    return diagram_paths