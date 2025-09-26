import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

class TomatoLeafProcessor:
    def __init__(self):
        self.image_size = (224, 224)
        # Focus only on tomato classes
        self.tomato_classes = {
            'Tomato___healthy': 'healthy',
            'Tomato___Early_blight': 'diseased',
            'Tomato___Late_blight': 'diseased', 
            'Tomato___Leaf_Mold': 'diseased',
            'Tomato___Septoria_leaf_spot': 'diseased',
            'Tomato___Spider_mites': 'diseased',
            'Tomato___Target_Spot': 'diseased',
            'Tomato___Tomato_mosaic_virus': 'diseased',
            'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 'diseased'
        }
    
    def find_tomato_folders(self, dataset_path):
        """Find all tomato-related folders in the dataset"""
        print("🔍 Searching for tomato leaf folders...")
        
        tomato_folders = {}
        for folder_name in os.listdir(dataset_path):
            if folder_name.startswith('Tomato___'):
                simplified_name = self.tomato_classes.get(folder_name, 'diseased')
                tomato_folders[folder_name] = simplified_name
                print(f"✅ Found: {folder_name} → {simplified_name}")
        
        if not tomato_folders:
            print("❌ No tomato folders found! Checking all folders...")
            all_folders = os.listdir(dataset_path)
            print("Available folders:", all_folders)
        
        return tomato_folders
    
    def load_tomato_dataset(self, dataset_path):
        """Load only tomato leaf images"""
        images = []
        labels = []  # 0=healthy, 1=diseased
        class_counts = {'healthy': 0, 'diseased': 0}
        
        tomato_folders = self.find_tomato_folders(dataset_path)
        
        if not tomato_folders:
            print("❌ No tomato data found. Creating sample data for testing.")
            return self.create_sample_data()
        
        for folder_name, class_type in tomato_folders.items():
            folder_path = os.path.join(dataset_path, folder_name)
            
            if not os.path.exists(folder_path):
                print(f"⚠️  Folder not found: {folder_path}")
                continue
            
            image_files = [f for f in os.listdir(folder_path) 
                          if f.endswith(('.jpg', '.png', '.jpeg'))]
            
            print(f"📁 Processing {folder_name}: {len(image_files)} images")
            
            for i, image_file in enumerate(image_files):
                if i % 100 == 0:  # Progress indicator
                    print(f"   Loading image {i}/{len(image_files)}")
                
                image_path = os.path.join(folder_path, image_file)
                
                try:
                    # Load and preprocess image
                    image = cv2.imread(image_path)
                    if image is None:
                        continue
                    
                    image = cv2.resize(image, self.image_size)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    images.append(image)
                    
                    # Label: 0 for healthy, 1 for diseased
                    if class_type == 'healthy':
                        labels.append(0)
                        class_counts['healthy'] += 1
                    else:
                        labels.append(1) 
                        class_counts['diseased'] += 1
                        
                except Exception as e:
                    print(f"❌ Error loading {image_path}: {e}")
        
        print(f"\n📊 Dataset Summary:")
        print(f"   Healthy leaves: {class_counts['healthy']}")
        print(f"   Diseased leaves: {class_counts['diseased']}")
        print(f"   Total images: {len(images)}")
        
        return np.array(images), np.array(labels)
    
    def create_sample_data(self):
        """Create sample data if no dataset found"""
        print("📝 Creating sample tomato leaf data...")
        
        images = []
        labels = []
        
        # Create sample images (green=healthy, spots=diseased)
        for i in range(200):
            if i % 2 == 0:  # Healthy leaves (green)
                img = np.random.randint(100, 150, (224, 224, 3)).astype(np.uint8)
                # Add vein-like patterns
                img[100:120, :] = [80, 120, 80]  # Darker vein
                labels.append(0)
            else:  # Diseased leaves (spots)
                img = np.random.randint(100, 150, (224, 224, 3)).astype(np.uint8)
                # Add disease spots
                spots = np.random.randint(50, 200, (30, 30, 3))
                x, y = np.random.randint(0, 194), np.random.randint(0, 194)
                img[x:x+30, y:y+30] = spots  # Brown/yellow spots
                labels.append(1)
            
            images.append(img)
        
        return np.array(images), np.array(labels)
    
    def augment_data(self, images, labels):
        """Data augmentation for tomato leaves"""
        if images is None or len(images) == 0:
            print("❌ No images to augment!")
            return images, labels
            
        data_augmentation = keras.Sequential([
            keras.layers.RandomFlip("horizontal_and_vertical"),
            keras.layers.RandomRotation(0.2),
            keras.layers.RandomZoom(0.2),
            keras.layers.RandomContrast(0.3),
            keras.layers.RandomBrightness(0.2)
        ])
        
        augmented_images = []
        augmented_labels = []
        
        print("🔄 Augmenting tomato leaf images...")
        
        for image, label in zip(images, labels):
            augmented_images.append(image)
            augmented_labels.append(label)
            
            # Create 3 augmented versions of each image
            for _ in range(3):
                augmented_img = data_augmentation(tf.expand_dims(image, 0))
                augmented_images.append(augmented_img[0].numpy())
                augmented_labels.append(label)
        
        print(f"✅ Augmented {len(images)} → {len(augmented_images)} images")
        return np.array(augmented_images), np.array(augmented_labels)
    
    def visualize_samples(self, images, labels, num_samples=8):
        """Visualize sample tomato leaves"""
        plt.figure(figsize=(12, 6))
        
        for i in range(num_samples):
            plt.subplot(2, 4, i+1)
            idx = np.random.randint(len(images))
            
            plt.imshow(images[idx].astype(np.uint8))
            plt.title(f"{'Healthy' if labels[idx] == 0 else 'Diseased'}")
            plt.axis('off')
        
        
        plt.tight_layout()
        plt.savefig('tomato_samples.png')
        plt.show()

# Test the tomato-specific processor
if __name__ == "__main__":
    processor = TomatoLeafProcessor()
    
    # Load dataset - adjust path to your dataset location
    dataset_path = r"C:\Users\HP\Documents\sih\dataset"
  # Change this to your actual path
    
    images, labels = processor.load_tomato_dataset(dataset_path)
    
    if len(images) > 0:
        processor.visualize_samples(images, labels)
        
        # Augment data
        augmented_images, augmented_labels = processor.augment_data(images, labels)
        
        print(f"🎯 Final dataset ready for training!")
        print(f"   Original: {len(images)} images")
        print(f"   Augmented: {len(augmented_images)} images")
        print(f"   Healthy: {np.sum(labels == 0)}")
        print(f"   Diseased: {np.sum(labels == 1)}")
    else:
        print("❌ No images loaded. Check your dataset path.")