import os
import cv2
from bing_image_downloader import downloader
import shutil

class DatasetGenerator:
    def __init__(self, base_dir='data/raw'):
        self.base_dir = base_dir
        self.image_dir = os.path.join(base_dir, 'images')
        
        # Create directory
        os.makedirs(self.image_dir, exist_ok=True)

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
        downloaded_count = 0
        
        for query_idx, query in enumerate(queries):
            query_dir = os.path.join(self.image_dir, query.replace(" ", "_"))
            
            # Download images to query-specific subfolder
            try:
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
                    print(f"\nProcessing images from {query_dir}")
                    
                    # Get list of valid image files
                    valid_extensions = ('.jpg', '.jpeg', '.png')
                    image_files = [f for f in os.listdir(query_dir) 
                                 if f.lower().endswith(valid_extensions)]
                    
                    for idx, filename in enumerate(image_files):
                        src_path = os.path.join(query_dir, filename)
                        
                        # Generate new filename
                        ext = os.path.splitext(filename)[1].lower()
                        new_filename = f"downloaded_car_{query_idx}_{idx + 1}{ext}"
                        dst_path = os.path.join(self.image_dir, new_filename)
                        
                        # Handle filename conflicts
                        counter = 1
                        while os.path.exists(dst_path):
                            new_filename = f"downloaded_car_{query_idx}_{idx + 1}_{counter}{ext}"
                            dst_path = os.path.join(self.image_dir, new_filename)
                            counter += 1
                        
                        try:
                            shutil.move(src_path, dst_path)
                            downloaded_count += 1
                            print(f"Moved: {filename} -> {new_filename}")
                        except Exception as e:
                            print(f"Error moving {filename}: {e}")
                            continue
                    
                    try:
                        # Remove empty query directory
                        shutil.rmtree(query_dir)
                        print(f"Removed directory: {query_dir}")
                    except Exception as e:
                        print(f"Error removing directory {query_dir}: {e}")
                
            except Exception as e:
                print(f"Error downloading images for query '{query}': {e}")
                continue
        
        print(f"\nTotal images downloaded and organized: {downloaded_count}")
        return downloaded_count

    def rename_existing_images(self, start_number=1):
        """Rename all images in the image directory sequentially"""
        print("\nRenaming existing images...")
        
        # Get all image files
        valid_extensions = ('.jpg', '.jpeg', '.png')
        image_files = [f for f in os.listdir(self.image_dir) 
                      if f.lower().endswith(valid_extensions)]
        
        # Sort files to ensure consistent ordering
        image_files.sort()
        
        # Track current number for sequential naming
        current_number = start_number
        renamed_count = 0
        
        # Rename files
        for old_filename in image_files:
            old_path = os.path.join(self.image_dir, old_filename)
            ext = os.path.splitext(old_filename)[1].lower()
            
            # Generate new filename
            new_filename = f"vehicle_{current_number:04d}{ext}"
            new_path = os.path.join(self.image_dir, new_filename)
            
            # Handle filename conflicts
            while os.path.exists(new_path) and new_path != old_path:
                current_number += 1
                new_filename = f"vehicle_{current_number:04d}{ext}"
                new_path = os.path.join(self.image_dir, new_filename)
            
            try:
                # Rename file
                os.rename(old_path, new_path)
                print(f"Renamed: {old_filename} -> {new_filename}")
                renamed_count += 1
                current_number += 1
            except Exception as e:
                print(f"Error renaming {old_filename}: {e}")
                continue
        
        print(f"\nRenamed {renamed_count} images")
        return renamed_count

if __name__ == "__main__":
    # Initialize generator
    dataset_gen = DatasetGenerator()
    
    # Download new images
    downloaded_count = dataset_gen.download_and_organize_images(limit=500)
    
    if downloaded_count > 0:
        # Rename all images sequentially
        renamed_count = dataset_gen.rename_existing_images()
        print(f"\nProcess completed successfully!")
        print(f"Total images in dataset: {renamed_count}")
    else:
        print("\nNo images were downloaded. Please check your internet connection and try again.")