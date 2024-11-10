import os
import cv2
import numpy as np
from bing_image_downloader import downloader
import shutil
import matplotlib.pyplot as plt
from ultralytics import YOLO

class DatasetGenerator:
    def __init__(self, base_dir='data/raw'):
        self.base_dir = base_dir
        self.image_dir = os.path.join(base_dir, 'images')
        self.mask_dir = os.path.join(base_dir, 'masks')
        self.vehicle_classes = ['car', 'truck', 'bus']
        
        # Initialize YOLO model
        self.model = YOLO('yolov8x-seg.pt')
        
        # Create directories
        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(self.mask_dir, exist_ok=True)
        os.makedirs('results/figures', exist_ok=True)

    def download_and_organize_images(self, limit=500):
        """Download and organize vehicle images"""
        print(f"Downloading {limit} images...")
        
        # Use multiple queries for diverse vehicle images
        queries = [
            "car photo",
            "single car image",
            "car photography",
            "automobile photo",
            "vehicle picture"
        ]
        
        images_per_query = limit // len(queries) + 1
        
        for query_idx, query in enumerate(queries):
            query_dir = os.path.join(self.image_dir, query.replace(" ", "_"))
            
            # Download images to query-specific subfolder
            downloader.download(
                query,
                limit=images_per_query,
                output_dir=self.image_dir,
                adult_filter_off=True,
                force_replace=False,
                timeout=60
            )
            
            # Move and rename images from query subfolder
            if os.path.exists(query_dir):
                print(f"Processing images from {query_dir}")
                for idx, filename in enumerate(os.listdir(query_dir)):
                    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                        src_path = os.path.join(query_dir, filename)
                        
                        # Generate new filename
                        new_filename = f"downloaded_car_{query_idx}_{idx + 1}{os.path.splitext(filename)[1]}"
                        dst_path = os.path.join(self.image_dir, new_filename)
                        
                        # Handle filename conflicts
                        counter = 1
                        while os.path.exists(dst_path):
                            new_filename = f"downloaded_car_{query_idx}_{idx + 1}_{counter}{os.path.splitext(filename)[1]}"
                            dst_path = os.path.join(self.image_dir, new_filename)
                            counter += 1
                        
                        shutil.move(src_path, dst_path)
                
                # Remove empty query directory
                shutil.rmtree(query_dir)

    def rename_existing_images(self):
        """Rename all images in the image directory"""
        print("\nRenaming existing images...")
        
        # Get all image files
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png']:
            image_files.extend([f for f in os.listdir(self.image_dir) 
                              if f.lower().endswith(ext)])
        
        # Sort files to ensure consistent ordering
        image_files.sort()
        
        # Rename files
        for idx, old_filename in enumerate(image_files, 1):
            old_path = os.path.join(self.image_dir, old_filename)
            new_filename = f"vehicle_{idx:04d}{os.path.splitext(old_filename)[1]}"
            new_path = os.path.join(self.image_dir, new_filename)
            
            # Handle filename conflicts
            counter = 1
            while os.path.exists(new_path) and new_path != old_path:
                new_filename = f"vehicle_{idx:04d}_{counter}{os.path.splitext(old_filename)[1]}"
                new_path = os.path.join(self.image_dir, new_filename)
                counter += 1
            
            # Rename file
            if new_path != old_path:
                os.rename(old_path, new_path)
                print(f"Renamed: {old_filename} -> {new_filename}")

    def generate_vehicle_masks(self):
        """Generate masks for vehicle images"""
        print("\nGenerating vehicle masks...")
        valid_images = []
        
        for filename in sorted(os.listdir(self.image_dir)):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(self.image_dir, filename)
                
                try:
                    # Run YOLO prediction
                    results = self.model.predict(image_path, conf=0.25)
                    
                    for result in results:
                        # Check if image contains vehicles
                        if result.boxes is not None:
                            classes = [self.model.names[int(cls)] for cls in result.boxes.cls]
                            num_vehicles = sum(1 for c in classes if c in self.vehicle_classes)
                            
                            # Keep only images with 1-3 vehicles
                            if 1 <= num_vehicles <= 3:
                                # Create mask for vehicles
                                image = cv2.imread(image_path)
                                if image is None:
                                    print(f"Failed to load image: {image_path}")
                                    continue
                                    
                                height, width = image.shape[:2]
                                mask = np.zeros((height, width), dtype=np.uint8)
                                
                                # Add vehicle segments to mask
                                if result.masks is not None:
                                    for segment, cls in zip(result.masks.data, result.boxes.cls):
                                        if self.model.names[int(cls)] in self.vehicle_classes:
                                            segment = segment.cpu().numpy()
                                            segment = (segment * 255).astype(np.uint8)
                                            segment = cv2.resize(segment, (width, height))
                                            mask = cv2.bitwise_or(mask, segment)
                                
                                # Save mask if it's valid
                                if np.sum(mask) > 1000:  # Ensure mask isn't empty
                                    mask_filename = os.path.splitext(filename)[0] + '_mask.png'
                                    mask_path = os.path.join(self.mask_dir, mask_filename)
                                    cv2.imwrite(mask_path, mask)
                                    valid_images.append(filename)
                                    print(f"Generated mask for {filename}")
                
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
                    continue
        
        # Remove images without valid masks
        for filename in os.listdir(self.image_dir):
            if filename not in valid_images:
                try:
                    os.remove(os.path.join(self.image_dir, filename))
                    print(f"Removed {filename} (no valid mask)")
                except Exception as e:
                    print(f"Error removing {filename}: {e}")
        
        print(f"\nRetained {len(valid_images)} valid vehicle images with masks")
        return valid_images

    def display_results(self, num_samples=5):
        """Display sample results"""
        print("\nGenerating visualization...")
        
        # Get matching image-mask pairs
        image_files = sorted([f for f in os.listdir(self.image_dir) 
                            if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        
        if not image_files:
            print("No valid image-mask pairs found")
            return
        
        # Create visualization
        num_samples = min(num_samples, len(image_files))
        fig, axes = plt.subplots(num_samples, 2, figsize=(12, 4*num_samples))
        
        for i, image_file in enumerate(image_files[:num_samples]):
            # Load image and mask
            image_path = os.path.join(self.image_dir, image_file)
            mask_path = os.path.join(self.mask_dir, 
                                   os.path.splitext(image_file)[0] + '_mask.png')
            
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            # Display
            axes[i, 0].imshow(image)
            axes[i, 0].set_title('Original Image')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(mask, cmap='gray')
            axes[i, 1].set_title('Vehicle Mask')
            axes[i, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig('results/figures/sample_results.png')
        plt.close()
        print("Results saved to results/figures/sample_results.png")

if __name__ == "__main__":
    # Initialize generator
    dataset_gen = DatasetGenerator()
    
    # Download new images
    dataset_gen.download_and_organize_images(limit=500)
    
    # Rename all images (both existing and newly downloaded)
    dataset_gen.rename_existing_images()
    
    # Generate vehicle masks
    valid_images = dataset_gen.generate_vehicle_masks()
    
    # Display results
    if valid_images:
        dataset_gen.display_results(num_samples=5)
        print("\nDataset creation completed successfully!")
    else:
        print("\nNo valid vehicle images were generated.")