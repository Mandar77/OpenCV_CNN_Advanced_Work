import matplotlib.pyplot as plt
import numpy as np
import os

def plot_training_history(history, config):
    """Plot training history metrics"""
    plt.figure(figsize=(15, 5))
    
    # Plot accuracy
    plt.subplot(121)
    plt.plot(history.history['accuracy'], label='Training')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Plot loss
    plt.subplot(122)
    plt.plot(history.history['loss'], label='Training')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(config['HISTORY_PLOT_PATH'])
    plt.close()

def display_predictions(model, X_test, y_test, config):
    """Display sample predictions"""
    # Make predictions
    predictions = model.predict(X_test[:5])
    
    # Create figure
    plt.figure(figsize=(15, 10))
    
    for idx in range(5):
        # Original image
        plt.subplot(5, 3, idx*3 + 1)
        plt.imshow(X_test[idx])
        plt.title('Original Image')
        plt.axis('off')
        
        # True mask
        plt.subplot(5, 3, idx*3 + 2)
        plt.imshow(y_test[idx, ..., 0], cmap='gray')
        plt.title('True Mask')
        plt.axis('off')
        
        # Predicted mask
        plt.subplot(5, 3, idx*3 + 3)
        plt.imshow(predictions[idx, ..., 0], cmap='gray')
        plt.title('Predicted Mask')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(config['PLOT_PATH'])
    plt.close()

def plot_attention_maps(model, image, layer_names=None):
    """Visualize attention maps from the model"""
    # Get intermediate layer outputs
    layer_outputs = [layer.output for layer in model.layers 
                    if 'attention' in layer.name.lower()]
    attention_model = tf.keras.Model(inputs=model.input, 
                                   outputs=layer_outputs)
    
    # Get attention maps
    attention_maps = attention_model.predict(np.expand_dims(image, 0))
    
    # Plot attention maps
    num_maps = len(attention_maps)
    if num_maps == 0:
        print("No attention layers found")
        return
    
    plt.figure(figsize=(15, 5))
    
    # Plot original image
    plt.subplot(1, num_maps+1, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')
    
    # Plot attention maps
    for idx, attention_map in enumerate(attention_maps):
        plt.subplot(1, num_maps+1, idx+2)
        plt.imshow(np.mean(attention_map[0], axis=-1), cmap='viridis')
        plt.title(f'Attention Map {idx+1}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('attention_maps.png')
    plt.close()