from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import io
import random
from PIL import Image

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Mock data for testing
MOCK_INGREDIENTS = [
    "tomato", "onion", "garlic", "chicken", "beef", "carrot", "potato",
    "broccoli", "spinach", "rice", "pasta", "cheese", "egg", "mushroom",
    "bell pepper", "olive oil", "salt", "pepper", "basil", "oregano"
]

# Specific ingredients for potato images
POTATO_INGREDIENTS = ["potato", "butter", "salt", "pepper", "garlic"]

# Specific recipes for potato images
POTATO_RECIPES = [
    """# Crispy Roasted Potatoes

## Ingredients:
- Potatoes
- Olive oil
- Salt
- Pepper
- Garlic powder
- Rosemary (optional)

## Instructions:
1. Preheat oven to 425°F (220°C).
2. Wash and cut potatoes into even-sized chunks.
3. Boil potatoes for 5-7 minutes until slightly tender.
4. Drain and let steam dry for 3 minutes.
5. Toss with olive oil, salt, pepper, and garlic powder.
6. Spread on a baking sheet and roast for 30-35 minutes until golden and crispy.
7. Flip halfway through cooking time.
8. Serve hot as a delicious side dish.
""",
    """# Creamy Mashed Potatoes

## Ingredients:
- Potatoes
- Butter
- Milk or cream
- Salt
- Pepper
- Garlic (optional)

## Instructions:
1. Peel and cut potatoes into chunks.
2. Place in a pot with cold water and add salt.
3. Bring to a boil and cook until fork-tender (about 15-20 minutes).
4. Drain well and return to the hot pot.
5. Add butter and mash until smooth.
6. Warm the milk/cream and gradually add while mashing.
7. Season with salt and pepper to taste.
8. For garlic mashed potatoes, add roasted or sautéed garlic.
9. Serve hot as a comforting side dish.
""",
    """# Simple Potato Wedges

## Ingredients:
- Potatoes
- Olive oil
- Salt
- Pepper
- Paprika
- Garlic powder

## Instructions:
1. Preheat oven to 400°F (200°C).
2. Wash potatoes well (no need to peel).
3. Cut into wedges of even thickness.
4. Toss with olive oil, salt, pepper, paprika, and garlic powder.
5. Arrange in a single layer on a baking sheet.
6. Bake for 30-35 minutes until golden and crispy.
7. Serve hot with your favorite dipping sauce.
"""
]

# General mock recipes for other foods
MOCK_RECIPES = [
    """# Simple Pasta Recipe

## Ingredients:
- Pasta
- Tomatoes
- Garlic
- Olive oil
- Salt
- Pepper
- Basil

## Instructions:
1. Cook pasta according to package instructions.
2. Heat olive oil in a pan and add minced garlic.
3. Add chopped tomatoes and cook for 5 minutes.
4. Season with salt, pepper, and basil.
5. Toss with cooked pasta and serve hot.
""",
    """# Quick Stir Fry

## Ingredients:
- Rice
- Vegetables (carrots, bell peppers, broccoli)
- Protein (chicken, beef, or tofu)
- Soy sauce
- Garlic
- Ginger
- Olive oil

## Instructions:
1. Cook rice according to package instructions.
2. Heat oil in a wok or large pan.
3. Add protein and cook until done.
4. Add vegetables and stir fry for 3-5 minutes.
5. Add minced garlic, ginger, and soy sauce.
6. Serve hot over rice.
""",
    """# Easy Omelette

## Ingredients:
- Eggs
- Cheese
- Vegetables (onions, bell peppers, tomatoes)
- Salt
- Pepper
- Butter

## Instructions:
1. Beat eggs in a bowl with salt and pepper.
2. Heat butter in a non-stick pan.
3. Pour in egg mixture and cook for 1 minute.
4. Add vegetables and cheese.
5. Fold omelette and cook for another minute.
6. Serve hot.
"""
]

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok", "message": "Mock Food Vision API is running"})

@app.route('/api/process-image', methods=['POST'])
def process_image():
    try:
        data = request.json
        if not data or 'image' not in data:
            return jsonify({"error": "No image data provided"}), 400
        
        # Decode the image
        image_data = data['image']
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        img = Image.open(io.BytesIO(image_bytes))
        
        # Check if the image contains potatoes by analyzing the image data
        # For simplicity, we'll assume any image is a potato image
        # In a real app, we would use image recognition here
        is_potato_image = True
        
        if is_potato_image:
            # Use potato-specific ingredients and recipes
            ingredients = POTATO_INGREDIENTS
            recipe = random.choice(POTATO_RECIPES)
            caption = "A delicious dish with potatoes that can be used for various recipes."
            
            # Create potato-specific detection results
            detection_results = [
                {"label": "potato", "score": 0.95},
                {"label": "starch", "score": 0.85},
                {"label": "vegetable", "score": 0.90}
            ]
            
            # Create food keywords for potatoes
            food_keywords = ["potato", "vegetable", "starch", "side dish", "ingredient"]
        else:
            # Use random ingredients and recipes for non-potato images
            num_ingredients = random.randint(3, 6)
            ingredients = random.sample(MOCK_INGREDIENTS, num_ingredients)
            recipe = random.choice(MOCK_RECIPES)
            caption = f"A delicious meal with {', '.join(ingredients[:-1])} and {ingredients[-1]}."
            
            # Create mock detection results
            detection_results = []
            for ingredient in ingredients:
                detection_results.append({
                    "label": ingredient,
                    "score": random.uniform(0.7, 0.95)
                })
            
            # Create food keywords
            food_keywords = ["food", "meal", "dish", "plate", "delicious", "tasty"] + ingredients
        
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

if __name__ == '__main__':
    print("Starting Mock Food Vision API server...")
    app.run(host='0.0.0.0', port=5001, debug=True)
