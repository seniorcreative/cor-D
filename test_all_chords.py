#!/usr/bin/env python3
from chord_analysis import create_chord_diagram
import os

def test_all_chord_diagrams(include_left_handed=True):
    """
    Test generating diagrams for all chord types with finger patterns
    
    Args:
        include_left_handed: If True, also generate left-handed versions
    """
    # Create output directories
    output_dir = 'static/chord_previews'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    left_handed_dir = os.path.join(output_dir, 'left_handed')
    if include_left_handed and not os.path.exists(left_handed_dir):
        os.makedirs(left_handed_dir)
    
    # List of chord types to test
    chord_types = [
        # Major chords
        'C', 'G', 'D', 'A', 'E', 'F', 'B', 'C#', 'D#', 'F#', 'G#',
        
        # Minor chords
        'Am', 'Em', 'Dm', 'Bm', 'Fm', 'Gm', 'Cm', 'C#m', 'D#m', 'F#m', 
        'G#m', 'A#m',
        
        # 7th chords
        'C7', 'D7', 'G7', 'A7', 'E7', 'B7', 'F7', 'D#7',
        
        # Major 7th chords
        'Cmaj7', 'Dmaj7', 'Gmaj7', 'Amaj7', 'Emaj7', 'Bmaj7', 'Fmaj7', 'C#maj7',
        'D#maj7', 'F#maj7', 'G#maj7', 'A#maj7',
        
        # Minor 7th chords
        'Am7', 'Em7', 'Dm7', 'Gm7', 'Cm7', 'A#m7',
        
        # Suspended chords
        'Csus2', 'Csus4', 'Dsus4', 'Esus4', 'Fsus4', 'C#sus2', 'D#sus2', 'D#sus4'
    ]
    
    # Generate all chord diagrams
    for chord_name in chord_types:
        print(f"Generating diagram for {chord_name} chord...")
        
        # Standard (right-handed) chord diagram
        safe_name = chord_name.replace('#', 'sharp')
        output_file = os.path.join(output_dir, f"{safe_name}.png")
        diagram_path = create_chord_diagram(chord_name, output_file)
        if diagram_path:
            print(f"  - Standard diagram saved to {diagram_path}")
        else:
            print(f"  - Failed to generate standard diagram for {chord_name}")
        
        # Left-handed chord diagram if requested
        if include_left_handed:
            left_handed_output_file = os.path.join(left_handed_dir, f"{safe_name}.png")
            left_handed_diagram_path = create_chord_diagram(chord_name, left_handed_output_file, left_handed=True)
            if left_handed_diagram_path:
                print(f"  - Left-handed diagram saved to {left_handed_diagram_path}")
            else:
                print(f"  - Failed to generate left-handed diagram for {chord_name}")
    
    print(f"\nDone! {len(chord_types)} chord types processed.")
    print(f"Check the {output_dir} directory for the generated chord diagrams.")
    if include_left_handed:
        print(f"Left-handed versions are in {left_handed_dir}.")

if __name__ == "__main__":
    test_all_chord_diagrams(include_left_handed=True)