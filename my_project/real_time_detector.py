import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import time  # ← ADD THIS IMPORT

class RealTimePlantAnalyzer:
    def __init__(self, model_path='plant_disease_detector.tflite'):
        try:
            # Load TensorFlow Lite model
            self.interpreter = tf.lite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            self.classes = ['healthy', 'fungal_infection', 'bacterial_spot', 'viral_disease']
            self.confidence_threshold = 0.7
            print("✅ AI Model loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
    
    def preprocess_image(self, image):
        """Preprocess image for model inference"""
        image = cv2.resize(image, (224, 224))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image / 255.0
        image = np.expand_dims(image, axis=0).astype(np.float32)
        return image
    
    def analyze_plant(self, image_path_or_array):
        """Analyze plant image for diseases"""
        try:
            if isinstance(image_path_or_array, str):
                image = cv2.imread(image_path_or_array)
                if image is None:
                    raise ValueError(f"Cannot load image: {image_path_or_array}")
            else:
                image = image_path_or_array
            
            processed_image = self.preprocess_image(image)
            
            self.interpreter.set_tensor(self.input_details[0]['index'], processed_image)
            self.interpreter.invoke()
            
            predictions = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
            
            max_confidence = np.max(predictions)
            predicted_class = self.classes[np.argmax(predictions)]
            
            disease_info = self.get_disease_info(predicted_class, max_confidence)
            
            return {
                'predicted_class': predicted_class,
                'confidence': float(max_confidence),
                'all_predictions': dict(zip(self.classes, predictions)),
                'disease_info': disease_info,
                'needs_treatment': predicted_class != 'healthy' and max_confidence > self.confidence_threshold
            }
            
        except Exception as e:
            print(f"❌ Analysis error: {e}")
            return {
                'predicted_class': 'error',
                'confidence': 0.0,
                'all_predictions': {},
                'disease_info': {'description': 'Analysis failed', 'action': 'Retry', 'severity': 'Unknown'},
                'needs_treatment': False
            }
    
    def get_disease_info(self, disease, confidence):
        """Get information about detected disease"""
        info = {
            'healthy': {
                'description': 'Plant is healthy and thriving',
                'action': 'Continue regular monitoring',
                'severity': 'None'
            },
            'fungal_infection': {
                'description': 'Fungal infection detected',
                'action': 'Apply fungicide spray immediately',
                'severity': 'High' if confidence > 0.8 else 'Medium'
            },
            'bacterial_spot': {
                'description': 'Bacterial infection causing spots on leaves',
                'action': 'Use copper-based bactericide',
                'severity': 'Medium'
            },
            'viral_disease': {
                'description': 'Viral infection detected',
                'action': 'Isolate plant and consult expert',
                'severity': 'High'
            }
        }
        return info.get(disease, {'description': 'Unknown', 'action': 'Monitor', 'severity': 'Low'})

# FIXED TEST FUNCTION
def test_detector():
    """Test the detector with a sample image"""
    analyzer = RealTimePlantAnalyzer()
    
    # Use a sample image for testing
    sample_image_path = "sample_plant.jpg"  # Replace with your image path
    
    try:
        result = analyzer.analyze_plant(sample_image_path)
        
        print(f"🔍 Analysis Result:")
        print(f"Status: {result['predicted_class']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Action: {result['disease_info']['action']}")
        print(f"Severity: {result['disease_info']['severity']}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("💡 Make sure you have a sample image file")

if __name__ == "__main__":
    test_detector()