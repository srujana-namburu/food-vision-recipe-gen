from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import io
from PIL import Image
import os
import tempfile
import cv2
import numpy as np

# Import our lightweight food vision functions
from food_vision_lite import get_caption, get_detected_ingredients, generate_recipe

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok", "message": "Food Vision API (Lite Version) is running"})

@app.route('/api/process-image', methods=['POST'])
def process_image():
    try:
        data = request.json
        if not data or 'image' not in data:
            return jsonify({"error": "No image data provided"}), 400
        
        # Decode base64 image
        image_data = data['image']
        if image_data.startswith('data:image'):
            # Remove the prefix if present (e.g., 'data:image/jpeg;base64,')
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Process the image
        caption = get_caption(image)
        detected_objects = get_detected_ingredients(image)
        
        # Filter out non-food words from caption
        caption_words = [word.lower() for word in caption.split() 
                        if len(word) > 3 and word.lower() not in 
                        ['with', 'and', 'the', 'that', 'this', 'there', 'their', 'they', 'them']]
        
        # Combine detected objects with caption words for ingredients
        ingredients = list(set(detected_objects + caption_words))
        
        # Generate recipe
        recipe = generate_recipe(ingredients)
        
        return jsonify({
            "caption": caption,
            "ingredients": ingredients,
            "recipe": recipe
        })
    
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/webcam', methods=['GET'])
def start_webcam():
    """Endpoint to start webcam capture and processing"""
    try:
        # Create a temporary file to store the captured image
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            temp_path = temp_file.name
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return jsonify({"error": "Could not open webcam"}), 500
        
        # Capture a single frame
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return jsonify({"error": "Failed to capture frame"}), 500
        
        # Save the frame to the temporary file
        cv2.imwrite(temp_path, frame)
        
        # Process the image
        image = Image.open(temp_path)
        caption = get_caption(image)
        detected_objects = get_detected_ingredients(image)
        
        # Combine detected objects with caption words for ingredients
        ingredients = list(set(detected_objects + caption.split()))
        
        # Generate recipe
        recipe = generate_recipe(ingredients)
        
        # Convert the image to base64 for sending back to the client
        with open(temp_path, "rb") as img_file:
            img_base64 = base64.b64encode(img_file.read()).decode('utf-8')
        
        # Clean up the temporary file
        os.unlink(temp_path)
        
        return jsonify({
            "image": f"data:image/jpeg;base64,{img_base64}",
            "caption": caption,
            "ingredients": ingredients,
            "recipe": recipe
        })
    
    except Exception as e:
        print(f"Error with webcam: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("Starting Food Vision API (Lite Version) server...")
    app.run(host='0.0.0.0', port=5000, debug=True)
