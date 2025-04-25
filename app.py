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

# Define your class labels with the exact names
class_names = ['Cotton__Aphids', 'Cotton_Army_worm', 'Cotton_Bacterial_blight', 'Cotton_Healthy', 'Cotton_Powdery_mildew', 'Cotton__Target_spot']

# Load the model
try:
    print("Loading model from:", model_path)
    model = PlantDiseaseModel(num_classes=6)  # Match your model's output size
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