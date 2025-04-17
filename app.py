from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import base64
import os
import sys

# Import the model architecture
from model_architecture import PlantDiseaseModel

app = Flask(__name__)
CORS(app)  # This allows requests from any origin

# Set the path to your model file
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plant-disease-model.pth")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define your class labels (adjust to match your 44 classes)
# Define your class labels with the exact names from your client
class_names = [
    'Apple___Apple_scab', 
    'Apple___Black_rot', 
    'Apple___Cedar_apple_rust', 
    'Apple___healthy', 
    'Blueberry___healthy', 
    'Cherry_(including_sour)___healthy', 
    'Cherry_(including_sour)___Powdery_mildew', 
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
    'Corn_(maize)___Common_rust_', 
    'Corn_(maize)___healthy', 
    'Corn_(maize)___Northern_Leaf_Blight', 
    'Cotton___Aphids', 
    'Cotton___Army_worm', 
    'Cotton___Bacterial_blight', 
    'Cotton___Healthy', 
    'Cotton___Powdery_mildew', 
    'Cotton___Target_spot', 
    'Grape___Black_rot', 
    'Grape___Esca_(Black_Measles)', 
    'Grape___healthy', 
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
    'Orange___Haunglongbing_(Citrus_greening)', 
    'Peach___Bacterial_spot', 
    'Peach___healthy', 
    'Pepper,_bell___Bacterial_spot', 
    'Pepper,_bell___healthy', 
    'Potato___Early_blight', 
    'Potato___healthy', 
    'Potato___Late_blight', 
    'Raspberry___healthy', 
    'Soybean___healthy', 
    'Squash___Powdery_mildew', 
    'Strawberry___healthy', 
    'Strawberry___Leaf_scorch', 
    'Tomato___Bacterial_spot', 
    'Tomato___Early_blight', 
    'Tomato___healthy', 
    'Tomato___Late_blight', 
    'Tomato___Leaf_Mold', 
    'Tomato___Septoria_leaf_spot', 
    'Tomato___Spider_mites Two-spotted_spider_mite', 
    'Tomato___Target_Spot', 
    'Tomato___Tomato_mosaic_virus', 
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus'
]

# Load the model
try:
    print("Loading model from:", model_path)
    model = PlantDiseaseModel(num_classes=44)  # Match your model's output size
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    # Don't exit, allow service to start anyway for debugging

# Define image transformation - adjust according to how your model was trained
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.json:
        return jsonify({'error': 'No image provided'}), 400
    
    try:
        # Get image from request
        image_data = request.json['image']
        # Handle both formats: with and without data:image prefix
        if 'base64,' in image_data:
            image_data = image_data.split('base64,')[1]
        
        # Decode and process image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Transform and predict
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            _, predicted = torch.max(outputs, 1)
            predicted_class = predicted.item()
            
            # Get class name
            disease = class_names[predicted_class]
            
            # Get confidence scores
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence = probabilities[predicted_class].item()
            
        return jsonify({
            'disease': disease,
            'confidence': round(confidence * 100, 2)
        })
    
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)