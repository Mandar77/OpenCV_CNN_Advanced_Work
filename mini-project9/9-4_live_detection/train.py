# train.py
import os
import shutil
import yaml
from pathlib import Path

def remove_cache(dataset_path):
    for split in ['train', 'valid', 'test']:
        cache_file = Path(dataset_path) / split / 'labels.cache'
        if cache_file.exists():
            cache_file.unlink()
            print(f"Removed cache file: {cache_file}")

def update_hyperparameters(hyp_path):
    with open(hyp_path, 'r') as f:
        hyp = yaml.safe_load(f)
    
    # Update hyperparameters
    hyp['lr0'] = 0.01
    hyp['lrf'] = 0.01
    
    with open(hyp_path, 'w') as f:
        yaml.dump(hyp, f)

def train_yolov5():
    # Define paths
    project_root = Path('D:/KhouryGithub/CS5330_FA24_Group1/mini-project9')
    yolov5_path = project_root / 'yolov5'
    dataset_path = yolov5_path / 'datasets'
    data_yaml = yolov5_path / 'data.yaml'
    hyp_path = yolov5_path / 'data/hyps/hyp.scratch-low.yaml'

    # Remove cache files
    remove_cache(dataset_path)

    # Update hyperparameters
    update_hyperparameters(hyp_path)

    # Training configuration
    epochs = 100
    batch_size = 16
    img_size = 640
    workers = 8  # Adjust based on your CPU cores

    # Load and update the data configuration
    with open(data_yaml, 'r') as f:
        data_dict = yaml.safe_load(f)
    
    data_dict['train'] = str(dataset_path / 'train' / 'images')
    data_dict['val'] = str(dataset_path / 'valid' / 'images')
    data_dict['test'] = str(dataset_path / 'test' / 'images')

    with open(data_yaml, 'w') as f:
        yaml.dump(data_dict, f)

    # Set up the training command with optimizations
    train_cmd = (
        f"python {yolov5_path}/train.py "
        f"--img {img_size} "
        f"--batch {batch_size} "
        f"--epochs {epochs} "
        f"--data {data_yaml} "
        f"--weights yolov5s.pt "
        f"--workers {workers} "
        f"--cache "
        f"--project {project_root}/runs/train "
        f"--name yolov5s_custom "
        f"--optimizer Adam "
        f"--patience 50 "
        f"--save-period 10 "
        f"--hyp {hyp_path}"
    )

    # Execute the training command
    os.chdir(yolov5_path)
    os.system(train_cmd)

if __name__ == '__main__':
    train_yolov5()