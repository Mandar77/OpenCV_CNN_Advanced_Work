from bing_image_downloader import downloader

def download_images(query, limit, output_dir):
    downloader.download(query, limit=limit, output_dir=output_dir,
                        adult_filter_off=True, force_replace=False, timeout=60)

# Set the output directory
output_dir = r"D:\KhouryGithub\CS5330_FA24_Group1\mini-project9\dataset"

# List of classes
classes = ["Stop Sign", "Traffic Signal"]

# Download images for each class
for class_name in classes:
    print(f"Downloading images for {class_name}")
    download_images(class_name, 150, output_dir)  # Downloading 150 images to ensure at least 100 usable images
    print(f"Finished downloading images for {class_name}")