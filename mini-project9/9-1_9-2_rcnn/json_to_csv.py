import os
import json
import csv
import argparse
from collections import defaultdict

def parse_arguments():
    parser = argparse.ArgumentParser(description="Convert LabelMe JSON annotations to CSV format")
    parser.add_argument("-i", "--input_dir", type=str, required=True, help="Directory containing JSON files")
    parser.add_argument("-o", "--output_file", type=str, required=True, help="Output CSV file path")
    return parser.parse_args()

def extract_bounding_box(shape):
    points = shape['points']
    x_coords, y_coords = zip(*points)
    return [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]

def process_json_file(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    image_path = data['imagePath']
    annotations = []
    
    for shape in data['shapes']:
        label = shape['label']
        bbox = extract_bounding_box(shape)
        annotations.append([image_path, label] + bbox)
    
    return annotations

def convert_json_to_csv(input_dir, output_file):
    all_annotations = []
    class_count = defaultdict(int)

    for filename in os.listdir(input_dir):
        if filename.endswith('.json'):
            json_path = os.path.join(input_dir, filename)
            annotations = process_json_file(json_path)
            all_annotations.extend(annotations)
            for ann in annotations:
                class_count[ann[1]] += 1

    # Sort annotations by image path
    all_annotations.sort(key=lambda x: x[0])

    with open(output_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['image_path', 'class', 'x_min', 'y_min', 'x_max', 'y_max'])
        csv_writer.writerows(all_annotations)

    print(f"Conversion complete. Output saved to {output_file}")
    print("Class distribution:")
    for class_name, count in class_count.items():
        print(f"  {class_name}: {count}")

def main():
    args = parse_arguments()
    convert_json_to_csv(args.input_dir, args.output_file)

if __name__ == "__main__":
    main()