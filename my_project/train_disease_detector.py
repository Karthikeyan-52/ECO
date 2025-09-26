import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import numpy as np
import time  # ← ADD THIS

class PlantDiseaseDetector:
    def __init__(self, num_classes=4):
        self.num_classes = num_classes
        self.image_size = (224, 224)
        self.model = None
        
    def create_model(self):
        """Create CNN model for plant disease detection"""
        base_model = keras.applications.MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)
        )
        base_model.trainable = False
        
        model = keras.Sequential([
            base_model,
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def train_model(self, images, labels, epochs=50):
        """Train the disease detection model"""
        # FIX: Check if images/labels are provided
        if images is None or labels is None:
            print("❌ ERROR: No training data provided!")
            print("💡 Tip: Run data_preprocessor.py first to generate augmented data")
            return None
        
        X_train, X_val, y_train, y_val = train_test_split(
            images, labels, test_size=0.2, random_state=42
        )
        
        X_train = X_train / 255.0
        X_val = X_val / 255.0
        
        y_train = keras.utils.to_categorical(y_train, self.num_classes)
        y_val = keras.utils.to_categorical(y_val, self.num_classes)
        
        self.model = self.create_model()
        
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        callbacks = [
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=5),
            keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True)
        ]
        
        print("🚀 Starting model training...")
        start_time = time.time()
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=32,
            callbacks=callbacks
        )
        
        training_time = time.time() - start_time
        print(f"✅ Training completed in {training_time:.2f} seconds")
        
        return history
    
    def convert_to_tflite(self, output_path='plant_disease_model.tflite'):
        """Convert to TensorFlow Lite for Raspberry Pi"""
        if self.model is None:
            print("❌ ERROR: No model trained yet! Call train_model() first.")
            return
        
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"✅ Model converted to {output_path}")

# FIXED TRAINING SECTION - Add proper data loading
if __name__ == "__main__":
    # First, load your data properly
    from data_processor import PlantDataPreprocessor
    
    # Initialize preprocessor
    preprocessor = PlantDataPreprocessor()
    
    # Load dataset (replace with your actual dataset path)
    try:
        images, labels = preprocessor.load_dataset("plant_dataset/")
        augmented_images, augmented_labels = preprocessor.augment_data(images, labels)
        
        print(f"📊 Loaded {len(augmented_images)} training images")
        
        # Now train the model
        detector = PlantDiseaseDetector(num_classes=4)
        history = detector.train_model(augmented_images, augmented_labels, epochs=50)
        
        if history is not None:
            detector.convert_to_tflite('plant_disease_detector.tflite')
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        print("💡 Make sure you have plant_dataset/ folder with images")