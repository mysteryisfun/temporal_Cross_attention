#!/usr/bin/env python3
"""
Quick diagnosis of label format differences
"""
import json

# Load both files
with open('data/raw/labels/labels.json', 'r') as f:
    categories = json.load(f)

with open('data/raw/labels/train.json', 'r') as f:
    train_data = json.load(f)

# Find specific examples
spreading_keys = [k for k in categories.keys() if "spreading" in k.lower()]
print("Spreading keys in categories:", spreading_keys)

holding_keys = [k for k in categories.keys() if "holding" in k.lower()]
print("Holding keys in categories:", holding_keys[:3])

# Check video 42326 template
video_42326 = next((item for item in train_data if item['id'] == '42326'), None)
if video_42326:
    template = video_42326['template']
    print(f"Video 42326 template: '{template}'")
    print(f"Exact match in categories: {template in categories}")
    
    # Check if we need to normalize
    clean_template = template.replace('[something]', 'something')
    print(f"Clean template: '{clean_template}'")
    print(f"Clean match in categories: {clean_template in categories}")

print("\nAll 'Spreading' related categories:")
for k, v in categories.items():
    if "spreading" in k.lower():
        print(f"  {v}: {k}")
