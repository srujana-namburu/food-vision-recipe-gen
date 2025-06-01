from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import io
from PIL import Image
import torch
import os
import tempfile
import cv2
import numpy as np

# Import our food vision functions
from food_vision import get_caption, get_detected_ingredients, generate_recipe

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok", "message": "Food Vision API is running"})

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
        print("Generating caption...")
        caption = get_caption(image)
        print(f"Caption: {caption}")
        
        print("Detecting ingredients...")
        detected_objects = get_detected_ingredients(image)
        print(f"Detected objects: {detected_objects}")
        
        # Extract potential food items from caption
        food_keywords = ["potato", "tomato", "onion", "garlic", "chicken", "beef", "carrot", 
                        "broccoli", "spinach", "rice", "pasta", "cheese", "egg", "mushroom",
                        "bell pepper", "olive oil", "salt", "pepper", "basil", "oregano", "food",
                        "dish", "meal", "vegetable", "fruit", "meat"]
        
        caption_ingredients = []
        for word in caption.lower().split():
            if word in food_keywords or any(keyword in word for keyword in food_keywords):
                caption_ingredients.append(word)
        
        # Combine detected objects with caption words for ingredients
        all_ingredients = detected_objects + caption_ingredients
        
        # Clean up ingredients - remove duplicates and non-food items
        common_non_food = ['a', 'the', 'and', 'with', 'of', 'in', 'on', 'plate', 'bowl', 'dish', 'image']
        ingredients = []
        for item in all_ingredients:
            if len(item) > 2 and item.lower() not in common_non_food:
                ingredients.append(item)
        
        # Remove duplicates
        ingredients = list(set(ingredients))
        
        # If no ingredients detected, add some default ones based on the image type
        if not ingredients:
            print("No ingredients detected, adding defaults based on caption")
            if "potato" in caption.lower():
                ingredients = ["potato", "butter", "salt", "pepper", "garlic"]
            else:
                ingredients = ["vegetable", "salt", "pepper", "olive oil"]
        
        print(f"Final ingredients list: {ingredients}")
        
        # Generate recipe
        recipe = generate_recipe(ingredients)
        
        # Create detection results for frontend
        detection_results = []
        for ingredient in ingredients:
            detection_results.append({
                "label": ingredient,
                "score": 0.9  # Default high confidence
            })
        
        # Create caption result in expected format
        caption_result = [{"generated_text": caption}]
        
        return jsonify({
            "caption": caption,
            "ingredients": ingredients,
            "recipe": recipe,
            "detectionResults": detection_results,
            "foodKeywords": food_keywords,
            "captionResult": caption_result
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
    print("Starting Food Vision API server...")
    app.run(host='0.0.0.0', port=5002, debug=True)
