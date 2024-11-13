import matplotlib.pyplot as plt
import numpy as np
import os
import json
import tensorflow as tf

class Visualizer:
    def __init__(self, config):
        self.config = config
        # Create necessary directories
        os.makedirs(os.path.dirname(config['PLOT_PATH']), exist_ok=True)
        os.makedirs(os.path.dirname(config['MODEL_PATH']), exist_ok=True)

    def save_metrics(self, history):
        """Save training metrics to JSON"""
        metrics = {
            'accuracy': {
                'final': float(history.history['accuracy'][-1]),
                'best': float(max(history.history['accuracy'])),
                'history': [float(x) for x in history.history['accuracy']]
            },
            'val_accuracy': {
                'final': float(history.history['val_accuracy'][-1]),
                'best': float(max(history.history['val_accuracy'])),
                'history': [float(x) for x in history.history['val_accuracy']]
            },
            'loss': {
                'final': float(history.history['loss'][-1]),
                'best': float(min(history.history['loss'])),
                'history': [float(x) for x in history.history['loss']]
            },
            'val_loss': {
                'final': float(history.history['val_loss'][-1]),
                'best': float(min(history.history['val_loss'])),
                'history': [float(x) for x in history.history['val_loss']]
            },
            'dice_coefficient': {
                'final': float(history.history['dice_coefficient'][-1]),
                'best': float(max(history.history['dice_coefficient'])),
                'history': [float(x) for x in history.history['dice_coefficient']]
            },
            'val_dice_coefficient': {
                'final': float(history.history['val_dice_coefficient'][-1]),
                'best': float(max(history.history['val_dice_coefficient'])),
                'history': [float(x) for x in history.history['val_dice_coefficient']]
            }
        }

        # Save metrics to JSON file
        metrics_file = os.path.join(os.path.dirname(self.config['PLOT_PATH']), 
                                  'model_metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        print(f"\nMetrics saved to: {metrics_file}")
        return metrics

    def plot_training_history(self, history):
        """Plot and save training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        ax1.plot(history.history['accuracy'], label='Training')
        ax1.plot(history.history['val_accuracy'], label='Validation')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot loss
        ax2.plot(history.history['loss'], label='Training')
        ax2.plot(history.history['val_loss'], label='Validation')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.config['HISTORY_PLOT_PATH'])
        plt.close()
        
        print(f"Training history plot saved to: {self.config['HISTORY_PLOT_PATH']}")

    def display_predictions(self, model, X_test, y_test):
        """Display and save sample predictions"""
        predictions = model.predict(X_test[:5])
        
        fig, axes = plt.subplots(5, 3, figsize=(15, 25))
        
        for idx in range(5):
            # Original image
            axes[idx, 0].imshow(X_test[idx])
            axes[idx, 0].set_title('Original Image')
            axes[idx, 0].axis('off')
            
            # True mask
            axes[idx, 1].imshow(y_test[idx, ..., 0], cmap='gray')
            axes[idx, 1].set_title('True Mask')
            axes[idx, 1].axis('off')
            
            # Predicted mask
            axes[idx, 2].imshow(predictions[idx, ..., 0], cmap='gray')
            axes[idx, 2].set_title(f'Predicted Mask\nDice: {self.calculate_dice(y_test[idx], predictions[idx]):.3f}')
            axes[idx, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.config['PLOT_PATH'])
        plt.close()
        
        print(f"Predictions visualization saved to: {self.config['PLOT_PATH']}")

    def calculate_dice(self, y_true, y_pred, smooth=1e-7):
        """Calculate Dice coefficient for a single prediction"""
        y_true_f = y_true.flatten()
        y_pred_f = y_pred.flatten()
        intersection = np.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

# Wrapper functions for backward compatibility
def display_predictions(model, X_test, y_test, config):
    visualizer = Visualizer(config)
    visualizer.display_predictions(model, X_test, y_test)

def plot_training_history(history, config):
    visualizer = Visualizer(config)
    visualizer.plot_training_history(history)

def save_results(model, history, X_test, y_test, config):
    """Save all results: model, metrics, and visualizations"""
    visualizer = Visualizer(config)
    
    # Save model
    model.save(config['MODEL_PATH'])
    print(f"\nModel saved to: {config['MODEL_PATH']}")
    
    # Save metrics
    metrics = visualizer.save_metrics(history)
    
    # Generate and save visualizations
    visualizer.plot_training_history(history)
    visualizer.display_predictions(model, X_test, y_test)
    
    return metrics