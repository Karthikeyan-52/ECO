# fixed_tomato_processor.py
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf

class FixedTomatoProcessor:
    def __init__(self):
        self.image_size = (224, 224)
    
    def explore_dataset(self, dataset_path):
        """Explore what's actually in your dataset"""
        print("🔍 Exploring dataset structure...")
        
        for root, dirs, files in os.walk(dataset_path):
            level = root.replace(dataset_path, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            
            # Show first 5 files in each directory
            for file in files[:5]:
                if file.endswith(('.jpg', '.png', '.jpeg')):
                    print(f"{subindent}{file}")
        
        # Check what's inside test and train folders
        test_path = os.path.join(dataset_path, 'test')
        train_path = os.path.join(dataset_path, 'train')
        
        if os.path.exists(test_path):
            print(f"\n📁 Test folder contents: {os.listdir(test_path)}")
        if os.path.exists(train_path):
            print(f"📁 Train folder contents: {os.listdir(train_path)}")
    
    def load_from_train_test_folders(self, dataset_path):
        """Load images from train/test folder structure"""
        images = []
        labels = []  # 0=healthy, 1=diseased
        
        # Map folder names to categories
        health_indicators = ['healthy', 'health', 'normal', 'good']
        disease_indicators = ['blight', 'mold', 'spot', 'virus', 'disease', 'infected']
        
        def classify_folder(folder_name):
            folder_lower = folder_name.lower()
            if any(indicator in folder_lower for indicator in health_indicators):
                return 'healthy', 0
            elif any(indicator in folder_lower for indicator in disease_indicators):
                return 'diseased', 1
            else:
                return 'unknown', -1
        
        # Process train and test folders
        for main_folder in ['train', 'test']:
            main_path = os.path.join(dataset_path, main_folder)
            
            if not os.path.exists(main_path):
                print(f"⚠️  {main_folder} folder not found")
                continue
                
            print(f"\n📂 Processing {main_folder} folder...")
            
            for class_folder in os.listdir(main_path):
                class_path = os.path.join(main_path, class_folder)
                
                if not os.path.isdir(class_path):
                    continue
                
                category, label = classify_folder(class_folder)
                
                if label == -1:  # Unknown category
                    print(f"❓ Unknown category: {class_folder}")
                    continue
                
                image_files = [f for f in os.listdir(class_path) 
                             if f.endswith(('.jpg', '.png', '.jpeg'))]
                
                print(f"   📁 {class_folder} → {category} ({len(image_files)} images)")
                
                # Load images (limit for testing)
                for image_file in image_files[:100]:  # Load first 100 images per class
                    image_path = os.path.join(class_path, image_file)
                    
                    try:
                        image = cv2.imread(image_path)
                        if image is None:
                            continue
                        
                        image = cv2.resize(image, self.image_size)
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        
                        images.append(image)
                        labels.append(label)
                        
                    except Exception as e:
                        continue
        
        if len(images) == 0:
            print("❌ No images loaded! Creating sample data...")
            return self.create_sample_data()
        
        print(f"\n📊 Dataset Summary:")
        print(f"   Total images: {len(images)}")
        print(f"   Healthy: {np.sum(np.array(labels) == 0)}")
        print(f"   Diseased: {np.sum(np.array(labels) == 1)}")
        
        return np.array(images), np.array(labels)
    
    def create_sample_data(self):
        """Create better sample data with visible differences"""
        print("🎨 Creating realistic sample tomato leaf images...")
        
        images = []
        labels = []
        
        for i in range(200):
            # Base green color for leaves
            base_green = np.random.randint(80, 120)
            
            if i % 2 == 0:  # Healthy leaves
                img = np.full((224, 224, 3), [base_green, base_green+30, base_green-20], dtype=np.uint8)
                
                # Add realistic leaf texture and veins
                for _ in range(50):  # Add vein patterns
                    x, y = np.random.randint(0, 224, 2)
                    cv2.line(img, (x, y), (x+np.random.randint(10,50), y), 
                            [base_green-20, base_green+10, base_green-30], 1)
                
                labels.append(0)
                
            else:  # Diseased leaves
                img = np.full((224, 224, 3), [base_green, base_green+30, base_green-20], dtype=np.uint8)
                
                # Add disease spots (brown/yellow)
                for _ in range(15):  # Add multiple spots
                    x, y = np.random.randint(20, 204, 2)
                    radius = np.random.randint(5, 15)
                    
                    # Brown/yellow spots for disease
                    spot_color = [180, 150, 80] if np.random.random() > 0.5 else [150, 120, 60]
                    cv2.circle(img, (x, y), radius, spot_color, -1)
                
                labels.append(1)
            
            # Add some noise for realism
            noise = np.random.randint(-10, 10, (224, 224, 3))
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            
            images.append(img)
        
        return np.array(images), np.array(labels)
    
    def visualize_samples(self, images, labels, num_samples=8):
        """Create better visualization"""
        plt.figure(figsize=(15, 8))
        
        for i in range(num_samples):
            plt.subplot(2, 4, i+1)
            idx = np.random.randint(len(images))
            
            plt.imshow(images[idx].astype(np.uint8))
            status = "Healthy" if labels[idx] == 0 else "Diseased"
            plt.title(f"{status}\n(Image {idx})", fontsize=12, fontweight='bold')
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('tomato_leaf_samples.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("📊 Sample breakdown:")
        print(f"   Healthy samples shown: {np.sum([labels[i] == 0 for i in range(num_samples)])}")
        print(f"   Diseased samples shown: {np.sum([labels[i] == 1 for i in range(num_samples)])}")

# Main execution
if __name__ == "__main__":
    processor = FixedTomatoProcessor()
    
    # Update this path to your actual dataset location
    dataset_path = "plant_dataset"  # Change this to your actual path
    
    print("🚀 Starting Tomato Leaf Processor...")
    
    # First, explore what's in your dataset
    processor.explore_dataset(dataset_path)
    
    # Then load images
    images, labels = processor.load_from_train_test_folders(dataset_path)
    
    # Visualize samples
    processor.visualize_samples(images, labels)
    
    print(f"\n🎯 Data preparation complete!")
    print(f"   Ready for training with {len(images)} images")
    print(f"   Healthy: {np.sum(labels == 0)}")
    print(f"   Diseased: {np.sum(labels == 1)}")