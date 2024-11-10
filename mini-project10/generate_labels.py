import os
import cv2
import numpy as np
from ultralytics import YOLO

class LabelGenerator:
    def __init__(self, base_dir='data/raw'):
        # Set environment variable to handle OpenMP error
        os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
        
        self.base_dir = base_dir
        self.image_dir = os.path.join(base_dir, 'images')
        self.label_dir = os.path.join(base_dir, 'labels')
        self.vehicle_classes = ['car', 'truck', 'bus']
        
        # Initialize YOLO model
        self.model = YOLO('yolov8x-seg.pt')
        
        # Create label directory if it doesn't exist
        os.makedirs(self.label_dir, exist_ok=True)

    def generate_labels(self):
        """Generate YOLO format labels for all images"""
        print("\nGenerating labels for images...")
        
        # Get list of all images
        image_files = sorted([f for f in os.listdir(self.image_dir) 
                            if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        
        print(f"Found {len(image_files)} images to process")
        
        processed_count = 0
        
        for filename in image_files:
            try:
                image_path = os.path.join(self.image_dir, filename)
                label_path = os.path.join(self.label_dir, 
                                        os.path.splitext(filename)[0] + '.txt')
                
                # Run YOLO prediction
                results = self.model.predict(image_path, conf=0.25)
                
                with open(label_path, 'w') as f:
                    for result in results:
                        if result.boxes is not None:
                            # Get image dimensions for normalization
                            img = cv2.imread(image_path)
                            height, width = img.shape[:2]
                            
                            # Process each detection
                            for box, cls in zip(result.boxes.data, result.boxes.cls):
                                class_name = self.model.names[int(cls)]
                                
                                if class_name in self.vehicle_classes:
                                    # Get normalized coordinates
                                    x1, y1, x2, y2, conf = box[:5]
                                    
                                    # Convert to YOLO format (center_x, center_y, width, height)
                                    center_x = ((x1 + x2) / 2) / width
                                    center_y = ((y1 + y2) / 2) / height
                                    box_width = (x2 - x1) / width
                                    box_height = (y2 - y1) / height
                                    
                                    # Map class name to index
                                    class_idx = self.vehicle_classes.index(class_name)
                                    
                                    # Write in YOLO format:
                                    # <class_idx> <center_x> <center_y> <width> <height> <confidence>
                                    f.write(f"{class_idx} {center_x:.6f} {center_y:.6f} "
                                           f"{box_width:.6f} {box_height:.6f} {conf:.6f}\n")
                
                processed_count += 1
                if processed_count % 50 == 0:
                    print(f"Processed {processed_count} images...")
                
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue
        
        print(f"\nLabel generation completed:")
        print(f"Successfully processed {processed_count} images")
        
        return processed_count

    def verify_labels(self):
        """Verify that all images have corresponding labels"""
        image_files = set(os.path.splitext(f)[0] for f in os.listdir(self.image_dir) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png')))
        label_files = set(os.path.splitext(f)[0] for f in os.listdir(self.label_dir) 
                         if f.endswith('.txt'))
        
        missing_labels = image_files - label_files
        extra_labels = label_files - image_files
        
        print("\nLabel Verification:")
        print(f"Total images: {len(image_files)}")
        print(f"Total labels: {len(label_files)}")
        print(f"Missing labels: {len(missing_labels)}")
        print(f"Extra labels: {len(extra_labels)}")
        
        if missing_labels:
            print("\nFirst 5 images missing labels:")
            for img in sorted(missing_labels)[:5]:
                print(f"  {img}")
        
        return len(missing_labels) == 0

    def visualize_labels(self, num_samples=5):
        """Visualize some images with their labels"""
        print("\nGenerating visualization...")
        
        # Get list of images with labels
        image_files = [f for f in os.listdir(self.image_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))
                      and os.path.exists(os.path.join(self.label_dir, 
                                                    os.path.splitext(f)[0] + '.txt'))]
        
        if not image_files:
            print("No images with labels found")
            return
        
        # Create visualization directory
        vis_dir = os.path.join('results', 'figures', 'label_visualization')
        os.makedirs(vis_dir, exist_ok=True)
        
        # Process random samples
        import random
        samples = random.sample(image_files, min(num_samples, len(image_files)))
        
        for image_file in samples:
            # Load image
            image_path = os.path.join(self.image_dir, image_file)
            label_path = os.path.join(self.label_dir, 
                                    os.path.splitext(image_file)[0] + '.txt')
            
            image = cv2.imread(image_path)
            height, width = image.shape[:2]
            
            # Draw labels
            with open(label_path, 'r') as f:
                for line in f:
                    class_idx, center_x, center_y, box_w, box_h, conf = map(float, line.strip().split())
                    
                    # Convert normalized coordinates to pixel coordinates
                    center_x *= width
                    center_y *= height
                    box_w *= width
                    box_h *= height
                    
                    # Calculate box coordinates
                    x1 = int(center_x - box_w/2)
                    y1 = int(center_y - box_h/2)
                    x2 = int(center_x + box_w/2)
                    y2 = int(center_y + box_h/2)
                    
                    # Draw box and label
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{self.vehicle_classes[int(class_idx)]} {conf:.2f}"
                    cv2.putText(image, label, (x1, y1-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Save visualization
            output_path = os.path.join(vis_dir, f"labeled_{image_file}")
            cv2.imwrite(output_path, image)
        
        print(f"Visualizations saved to {vis_dir}")

if __name__ == "__main__":
    # Initialize label generator
    generator = LabelGenerator()
    
    # Generate labels
    num_processed = generator.generate_labels()
    
    # Verify labels
    if generator.verify_labels():
        print("\nAll images have corresponding labels!")
        
        # Generate visualizations
        generator.visualize_labels(num_samples=5)
        
        print("\nLabel generation completed successfully!")
    else:
        print("\nSome images are missing labels. Please check the output above.")