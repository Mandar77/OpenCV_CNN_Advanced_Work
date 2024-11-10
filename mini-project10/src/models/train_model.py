import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

def dice_coefficient(y_true, y_pred):
    smooth = 1e-15
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)

def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    return 0.5 * bce + 0.5 * dice

class ModelTrainer:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.history = None

    def build_model(self):
        """Build and compile the U-Net model"""
        input_shape = (
            self.config['INPUT_IMAGE_HEIGHT'],
            self.config['INPUT_IMAGE_WIDTH'],
            self.config['NUM_CHANNELS']
        )
        
        # Create model
        inputs = tf.keras.layers.Input(input_shape)
        
        # Encoder
        def conv_block(x, filters, dropout_rate=0.3):
            x = tf.keras.layers.Conv2D(filters, (3, 3), padding='same')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation('relu')(x)
            x = tf.keras.layers.Dropout(dropout_rate)(x)
            
            x = tf.keras.layers.Conv2D(filters, (3, 3), padding='same')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation('relu')(x)
            return x

        # Encoder path
        enc1 = conv_block(inputs, 64)
        pool1 = tf.keras.layers.MaxPooling2D((2, 2))(enc1)
        
        enc2 = conv_block(pool1, 128)
        pool2 = tf.keras.layers.MaxPooling2D((2, 2))(enc2)
        
        enc3 = conv_block(pool2, 256)
        pool3 = tf.keras.layers.MaxPooling2D((2, 2))(enc3)
        
        enc4 = conv_block(pool3, 512)
        pool4 = tf.keras.layers.MaxPooling2D((2, 2))(enc4)

        # Bridge
        bridge = conv_block(pool4, 1024)

        # Decoder
        def upconv_block(x, skip_features, filters):
            x = tf.keras.layers.Conv2DTranspose(filters, (2, 2), strides=(2, 2))(x)
            x = tf.keras.layers.Concatenate()([x, skip_features])
            x = conv_block(x, filters)
            return x

        # Decoder path
        dec4 = upconv_block(bridge, enc4, 512)
        dec3 = upconv_block(dec4, enc3, 256)
        dec2 = upconv_block(dec3, enc2, 128)
        dec1 = upconv_block(dec2, enc1, 64)

        outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(dec1)
        
        self.model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
        
        # Compile model
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=float(self.config['INIT_LR'])),
            loss=bce_dice_loss,
            metrics=['accuracy', dice_coefficient]
        )
        
        return self.model

    def train(self, X_train, y_train, X_test, y_test):
        """Train the model with callbacks"""
        if self.model is None:
            self.build_model()

        # Create callbacks
        callbacks = [
            ModelCheckpoint(
                filepath=self.config['MODEL_PATH'],
                monitor='val_dice_coefficient',
                mode='max',
                save_best_only=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=float(self.config['REDUCE_LR_FACTOR']),
                patience=int(self.config['REDUCE_LR_PATIENCE']),
                min_lr=float(self.config['MIN_LR']),
                verbose=1
            ),
            EarlyStopping(
                monitor='val_dice_coefficient',
                mode='max',
                patience=int(self.config['EARLY_STOPPING_PATIENCE']),
                restore_best_weights=True,
                verbose=1
            )
        ]

        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            batch_size=int(self.config['BATCH_SIZE']),
            epochs=int(self.config['NUM_EPOCHS']),
            callbacks=callbacks,
            verbose=1
        )
        
        return self.model, self.history

def train_model(config, X_train, y_train, X_test, y_test):
    """Main training function"""
    trainer = ModelTrainer(config)
    return trainer.train(X_train, y_train, X_test, y_test)