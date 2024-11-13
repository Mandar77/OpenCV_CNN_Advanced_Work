import yaml
import os
import tensorflow as tf
from src.models.train_model import train_model
from src.visualization.visualize import save_results
from src.data.make_dataset import load_and_prepare_data, verify_data

def load_config():
    """Load configuration from YAML file"""
    with open('config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    return config

if __name__ == "__main__":
    # Set environment variable for OpenMP
    os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
    
    # Define global directories
    BASE_DIR = "D:/KhouryGithub/CS5330_FA24_Group1/mini-project10"
    IMAGE_DIR = os.path.join(BASE_DIR, "data/raw/images")
    MASK_DIR = os.path.join(BASE_DIR, "data/raw/masks")
    RESULTS_DIR = os.path.join(BASE_DIR, "results")
    
    # Create necessary directories
    os.makedirs(os.path.join(RESULTS_DIR, "models"), exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "figures"), exist_ok=True)

    # Load and update config
    config = load_config()
    config['IMAGE_DATASET_PATH'] = IMAGE_DIR
    config['MASK_DATASET_PATH'] = MASK_DIR
    config['MODEL_PATH'] = os.path.join(RESULTS_DIR, "models/unet_model.keras")
    config['PLOT_PATH'] = os.path.join(RESULTS_DIR, "figures/predictions.png")
    config['HISTORY_PLOT_PATH'] = os.path.join(RESULTS_DIR, "figures/training_history.png")

    print(f"Using image path: {config['IMAGE_DATASET_PATH']}")
    print(f"Using mask path: {config['MASK_DATASET_PATH']}")

    # Load and prepare data
    X_train, X_test, y_train, y_test = load_and_prepare_data(config)
    
    if X_train is None or y_train is None:
        print("Failed to load data. Exiting.")
        exit(1)

    # Verify data
    verify_data(X_train, X_test, y_train, y_test)

    # Train model and save results
    try:
        model, history = train_model(config, X_train, y_train, X_test, y_test)
        
        # Save all results
        metrics = save_results(model, history, X_test, y_test, config)
        
        print("\nTraining and saving completed successfully!")
        print(f"\nFinal Metrics:")
        print(f"Accuracy: {metrics['accuracy']['final']:.4f}")
        print(f"Dice Coefficient: {metrics['dice_coefficient']['final']:.4f}")
        print(f"Validation Accuracy: {metrics['val_accuracy']['final']:.4f}")
        print(f"Validation Dice Coefficient: {metrics['val_dice_coefficient']['final']:.4f}")
        
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        exit(1)