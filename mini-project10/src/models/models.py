import tensorflow as tf
from tensorflow.keras import layers, Model

def create_unet_model(input_shape, num_classes=1):
    """Create U-Net model with improved architecture"""
    
    # Input layer
    inputs = layers.Input(input_shape)
    
    # Contracting Path (Encoder)
    def encoder_block(x, filters, dropout_rate=0.3):
        x = layers.Conv2D(filters, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(dropout_rate)(x)
        
        x = layers.Conv2D(filters, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        return x
    
    # Encoder blocks with increasing filters
    enc1 = encoder_block(inputs, 64)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(enc1)
    
    enc2 = encoder_block(pool1, 128)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(enc2)
    
    enc3 = encoder_block(pool2, 256)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(enc3)
    
    enc4 = encoder_block(pool3, 512)
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(enc4)
    
    # Bridge
    bridge = encoder_block(pool4, 1024)
    
    # Expanding Path (Decoder)
    def decoder_block(x, skip_features, filters, dropout_rate=0.3):
        # Upsampling
        x = layers.Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(x)
        
        # Attention mechanism
        gate = layers.Conv2D(filters, (1, 1), activation='sigmoid')(x)
        skip_features = layers.multiply([skip_features, gate])
        
        # Concatenate
        x = layers.Concatenate()([x, skip_features])
        
        # Convolution blocks
        x = layers.Conv2D(filters, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(dropout_rate)(x)
        
        x = layers.Conv2D(filters, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        return x
    
    # Decoder blocks with skip connections
    dec4 = decoder_block(bridge, enc4, 512)
    dec3 = decoder_block(dec4, enc3, 256)
    dec2 = decoder_block(dec3, enc2, 128)
    dec1 = decoder_block(dec2, enc1, 64)
    
    # Output layer
    if num_classes == 1:
        outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(dec1)
    else:
        outputs = layers.Conv2D(num_classes, (1, 1), activation='softmax')(dec1)
    
    # Create model
    model = Model(inputs=[inputs], outputs=[outputs])
    
    return model

def dice_coefficient(y_true, y_pred, smooth=1e-7):
    """Calculate Dice coefficient"""
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    """Calculate Dice loss"""
    return 1 - dice_coefficient(y_true, y_pred)

def bce_dice_loss(y_true, y_pred):
    """Combined binary crossentropy and Dice loss"""
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    return 0.5 * bce + 0.5 * dice

# Custom metrics
def iou_score(y_true, y_pred, smooth=1e-7):
    """Calculate IoU (Intersection over Union) score"""
    intersection = tf.keras.backend.sum(y_true * y_pred)
    union = tf.keras.backend.sum(y_true) + tf.keras.backend.sum(y_pred) - intersection
    return (intersection + smooth) / (union + smooth)