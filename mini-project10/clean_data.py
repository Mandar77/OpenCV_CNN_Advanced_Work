import os
import shutil
import json
from datetime import datetime

class DataCleaner:
    def __init__(self, base_dir='data/raw'):
        self.base_dir = base_dir
        self.image_dir = os.path.join(base_dir, 'images')
        self.mask_dir = os.path.join(base_dir, 'masks')
        self.removed_dir = os.path.join(base_dir, 'removed_images')
        
        # Create removed images directory
        os.makedirs(self.removed_dir, exist_ok=True)
        
        # Create log directory
        self.log_dir = os.path.join(base_dir, 'logs')
        os.makedirs(self.log_dir, exist_ok=True)

    def get_valid_image_mask_pairs(self):
        """Get list of images that have corresponding masks"""
        image_files = set(os.path.splitext(f)[0] for f in os.listdir(self.image_dir)
                         if f.lower().endswith(('.jpg', '.jpeg', '.png')))
        mask_files = set(os.path.splitext(f)[0].replace('_mask', '') 
                        for f in os.listdir(self.mask_dir)
                        if f.endswith('_mask.png'))
        
        valid_pairs = image_files.intersection(mask_files)
        invalid_images = image_files - mask_files
        
        return list(valid_pairs), list(invalid_images)

    def move_invalid_images(self):
        """Move images without masks to removed_images directory"""
        print("\nCleaning dataset...")
        
        # Get valid and invalid images
        valid_pairs, invalid_images = self.get_valid_image_mask_pairs()
        
        print(f"Found {len(valid_pairs)} valid image-mask pairs")
        print(f"Found {len(invalid_images)} images without masks")
        
        # Track moved files
        moved_files = []
        errors = []
        
        # Move invalid images
        for base_name in invalid_images:
            # Find the image file with any extension
            image_file = None
            for ext in ['.jpg', '.jpeg', '.png']:
                temp_name = base_name + ext
                if os.path.exists(os.path.join(self.image_dir, temp_name)):
                    image_file = temp_name
                    break
            
            if image_file:
                src_path = os.path.join(self.image_dir, image_file)
                dst_path = os.path.join(self.removed_dir, image_file)
                
                # Handle filename conflicts
                if os.path.exists(dst_path):
                    base, ext = os.path.splitext(image_file)
                    counter = 1
                    while os.path.exists(dst_path):
                        new_name = f"{base}_{counter}{ext}"
                        dst_path = os.path.join(self.removed_dir, new_name)
                        counter += 1
                
                try:
                    shutil.move(src_path, dst_path)
                    moved_files.append({
                        'original_name': image_file,
                        'new_path': dst_path
                    })
                    print(f"Moved: {image_file}")
                except Exception as e:
                    errors.append({
                        'file': image_file,
                        'error': str(e)
                    })
                    print(f"Error moving {image_file}: {e}")
        
        # Save cleaning log
        log_data = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'valid_pairs': len(valid_pairs),
            'invalid_images': len(invalid_images),
            'moved_files': moved_files,
            'errors': errors
        }
        
        log_file = os.path.join(self.log_dir, 
                               f'cleaning_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=4)
        
        print(f"\nCleaning completed:")
        print(f"- Valid pairs remaining: {len(valid_pairs)}")
        print(f"- Invalid images moved: {len(moved_files)}")
        print(f"- Errors encountered: {len(errors)}")
        print(f"\nCleaning log saved to: {log_file}")
        
        return len(valid_pairs), len(moved_files), len(errors)

    def verify_dataset(self):
        """Verify that all remaining images have corresponding masks"""
        print("\nVerifying dataset...")
        
        image_files = sorted([f for f in os.listdir(self.image_dir)
                            if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        
        all_valid = True
        for image_file in image_files:
            base_name = os.path.splitext(image_file)[0]
            mask_file = f"{base_name}_mask.png"
            
            if not os.path.exists(os.path.join(self.mask_dir, mask_file)):
                print(f"Warning: No mask found for {image_file}")
                all_valid = False
        
        if all_valid:
            print("Verification successful: All images have corresponding masks")
        else:
            print("Verification failed: Some images are missing masks")
        
        return all_valid

if __name__ == "__main__":
    # Initialize cleaner
    cleaner = DataCleaner()
    
    # Clean dataset
    valid_count, moved_count, error_count = cleaner.move_invalid_images()
    
    # Verify final dataset
    if cleaner.verify_dataset():
        print("\nDataset cleaning completed successfully!")
        print(f"Final dataset contains {valid_count} valid image-mask pairs")
        print(f"Invalid images can be found in: {cleaner.removed_dir}")
    else:
        print("\nWarning: Dataset may still contain inconsistencies")