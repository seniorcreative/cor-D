#!/usr/bin/env python3
from chord_analysis import create_chord_diagram
import os

# Create output directory if it doesn't exist
output_dir = 'static/chord_previews'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Test with a few common chords
test_chords = ['C', 'G', 'D', 'A', 'E', 'Am', 'Em', 'F']

for chord in test_chords:
    output_file = os.path.join(output_dir, f"{chord.replace('#', 'sharp')}.png")
    print(f"Generating diagram for {chord} chord...")
    diagram_path = create_chord_diagram(chord, output_file)
    print(f"Saved to {diagram_path}")

print("Done! Check the static/chord_previews directory for the generated chord diagrams.")