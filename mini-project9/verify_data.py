import os

# Paths to the image and label directories
image_dirs = ['datasets/dataset/Images/train', 'datasets/dataset/Images/val']
label_dirs = ['datasets/dataset/labels/train', 'datasets/dataset/labels/val']

# Loop through each image directory
for img_dir, lbl_dir in zip(image_dirs, label_dirs):
    # Get all image filenames (without extension)
    image_files = [os.path.splitext(f)[0] for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))]
    
    # Get all label filenames (without extension)
    label_files = [os.path.splitext(f)[0] for f in os.listdir(lbl_dir) if f.endswith('.txt')]
    
    # Find images without corresponding labels
    missing_labels = set(image_files) - set(label_files)
    
    # Remove images without labels
    for img in missing_labels:
        os.remove(os.path.join(img_dir, img + '.jpg'))
        print(f"Removed {img}.jpg because it has no corresponding label.")

print("Unlabeled images have been removed.")