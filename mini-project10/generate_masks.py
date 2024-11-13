import os
import cv2
import numpy as np
from bing_image_downloader import downloader
import shutil
import matplotlib.pyplot as plt
from ultralytics import YOLO

class DatasetOrganizer:
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

    def organize_subfolder_images(self):
        """Move images from subfolders to main directory"""
        print("\nMoving images from subfolders...")
        
        # Get all subfolders
        subfolders = [f for f in os.listdir(self.image_dir) 
                     if os.path.isdir(os.path.join(self.image_dir, f))]
        
        # Track moved files
        moved_files = []
        
        for subfolder in subfolders:
            subfolder_path = os.path.join(self.image_dir, subfolder)
            print(f"Processing subfolder: {subfolder}")
            
            try:
                # Get all files in subfolder
                files = os.listdir(subfolder_path)
                
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        src_path = os.path.join(subfolder_path, file)
                        
                        # Generate new filename
                        base, ext = os.path.splitext(file)
                        new_filename = f"temp_{len(moved_files):04d}{ext.lower()}"
                        dst_path = os.path.join(self.image_dir, new_filename)
                        
                        # Handle filename conflicts
                        counter = 1
                        while os.path.exists(dst_path):
                            new_filename = f"temp_{len(moved_files):04d}_{counter}{ext.lower()}"
                            dst_path = os.path.join(self.image_dir, new_filename)
                            counter += 1
                        
                        # Move file
                        shutil.move(src_path, dst_path)
                        moved_files.append(new_filename)
                        print(f"Moved: {file} -> {new_filename}")
                
                # Remove empty subfolder
                shutil.rmtree(subfolder_path)
                print(f"Removed empty subfolder: {subfolder}")
                
            except Exception as e:
                print(f"Error processing subfolder {subfolder}: {e}")
                continue
        
        print(f"\nMoved {len(moved_files)} files from subfolders")
        return moved_files

    def rename_all_images_sequentially(self, start_number=1):
        """Rename all images in the main directory sequentially"""
        print("\nRenaming all images sequentially...")
        
        # Get all image files
        image_files = sorted([f for f in os.listdir(self.image_dir) 
                            if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        
        renamed_files = []
        current_number = start_number
        
        for old_filename in image_files:
            old_path = os.path.join(self.image_dir, old_filename)
            
            # Generate new filename
            ext = os.path.splitext(old_filename)[1].lower()
            new_filename = f"vehicle_{current_number:04d}{ext}"
            new_path = os.path.join(self.image_dir, new_filename)
            
            # Handle filename conflicts
            while os.path.exists(new_path) and new_path != old_path:
                current_number += 1
                new_filename = f"vehicle_{current_number:04d}{ext}"
                new_path = os.path.join(self.image_dir, new_filename)
            
            # Rename file
            os.rename(old_path, new_path)
            renamed_files.append(new_filename)
            print(f"Renamed: {old_filename} -> {new_filename}")
            
            current_number += 1
        
        print(f"\nRenamed {len(renamed_files)} files")
        return renamed_files

    def generate_vehicle_masks(self):
        """Generate masks for vehicle images"""
        print("\nGenerating vehicle masks...")
        valid_images = []
        
        # Get all image files
        image_files = sorted([f for f in os.listdir(self.image_dir) 
                            if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        
        for filename in image_files:
            try:
                image_path = os.path.join(self.image_dir, filename)
                
                # Run YOLO prediction
                results = self.model.predict(image_path, conf=0.25)
                
                for result in results:
                    if result.boxes is not None:
                        classes = [self.model.names[int(cls)] for cls in result.boxes.cls]
                        num_vehicles = sum(1 for c in classes if c in self.vehicle_classes)
                        
                        # Keep only images with 1-3 vehicles
                        if 1 <= num_vehicles <= 3:
                            image = cv2.imread(image_path)
                            if image is None:
                                print(f"Failed to load image: {image_path}")
                                continue
                            
                            height, width = image.shape[:2]
                            mask = np.zeros((height, width), dtype=np.uint8)
                            
                            if result.masks is not None:
                                for segment, cls in zip(result.masks.data, result.boxes.cls):
                                    if self.model.names[int(cls)] in self.vehicle_classes:
                                        segment = segment.cpu().numpy()
                                        segment = (segment * 255).astype(np.uint8)
                                        segment = cv2.resize(segment, (width, height))
                                        mask = cv2.bitwise_or(mask, segment)
                                
                                if np.sum(mask) > 1000:
                                    mask_filename = os.path.splitext(filename)[0] + '_mask.png'
                                    mask_path = os.path.join(self.mask_dir, mask_filename)
                                    cv2.imwrite(mask_path, mask)
                                    valid_images.append(filename)
                                    print(f"Generated mask for {filename}")
            
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue
        
        print(f"\nGenerated masks for {len(valid_images)} valid vehicle images")
        return valid_images

    def display_results(self, valid_images, num_samples=5):
        """Display sample results"""
        if not valid_images:
            print("No valid images to display")
            return
        
        num_samples = min(num_samples, len(valid_images))
        fig, axes = plt.subplots(num_samples, 2, figsize=(12, 4*num_samples))
        
        for i, image_file in enumerate(sorted(valid_images)[:num_samples]):
            image_path = os.path.join(self.image_dir, image_file)
            mask_path = os.path.join(self.mask_dir, 
                                   os.path.splitext(image_file)[0] + '_mask.png')
            
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
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
    # Initialize organizer
    organizer = DatasetOrganizer()
    
    # Step 1: Move images from subfolders
    moved_files = organizer.organize_subfolder_images()
    
    # Step 2: Rename all images sequentially
    renamed_files = organizer.rename_all_images_sequentially()
    
    # Step 3: Generate masks
    valid_images = organizer.generate_vehicle_masks()
    
    # Step 4: Display results
    if valid_images:
        organizer.display_results(valid_images)
        print("\nProcess completed successfully!")
    else:
        print("\nNo valid vehicle images were processed.")