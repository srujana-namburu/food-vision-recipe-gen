import cv2
from PIL import Image
import torch
from transformers import (
    VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer,
    DetrForObjectDetection, DetrImageProcessor,
    AutoModelForCausalLM, AutoTokenizer as CausalTokenizer,
)
import random

# --- Load Models ---
print("Loading models...")

# Image Captioning
print("Loading image captioning model...")
caption_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
caption_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
caption_tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# Object Detection
print("Loading object detection model...")
det_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
det_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

# Recipe Generation - Using a smaller model instead of Mistral-7B
print("Loading recipe generation model...")
try:
    # Try to use a smaller language model for recipe generation
    recipe_model = AutoModelForCausalLM.from_pretrained("gpt2")
    recipe_tokenizer = CausalTokenizer.from_pretrained("gpt2")
    use_real_recipe_model = True
except Exception as e:
    print(f"Could not load recipe model: {e}")
    print("Using fallback recipe generation")
    use_real_recipe_model = False
    
    # Define some food-related keywords for recipe generation
    FOOD_KEYWORDS = ["potato", "tomato", "onion", "garlic", "chicken", "beef", "carrot", 
                    "broccoli", "spinach", "rice", "pasta", "cheese", "egg", "mushroom",
                    "bell pepper", "olive oil", "salt", "pepper", "basil", "oregano"]

print("Models loaded successfully.")

# --- Helper Functions ---
def get_caption(image: Image.Image) -> str:
    inputs = caption_processor(images=image, return_tensors="pt").pixel_values
    outputs = caption_model.generate(inputs, max_length=64)
    return caption_tokenizer.decode(outputs[0], skip_special_tokens=True)

def get_detected_ingredients(image: Image.Image) -> list:
    # Lower the detection threshold to catch more potential food items
    threshold = 0.3  # Lower threshold to detect more objects
    
    inputs = det_processor(images=image, return_tensors="pt")
    outputs = det_model(**inputs)
    target_sizes = torch.tensor([image.size[::-1]])
    results = det_processor.post_process_object_detection(outputs, threshold=threshold, target_sizes=target_sizes)[0]
    
    # Get labels and scores
    labels = []
    scores = []
    for i, (label_id, score) in enumerate(zip(results["labels"], results["scores"])):
        label = det_model.config.id2label[label_id.item()]
        score_val = score.item()
        print(f"Detected: {label} with confidence {score_val:.2f}")
        labels.append(label)
        scores.append(score_val)
    
    # Filter for food items or common objects that could be food
    food_related = [
        "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", 
        "donut", "cake", "bowl", "cup", "fork", "knife", "spoon", "food", "fruit", "vegetable",
        "potato", "tomato", "onion", "garlic", "pepper", "rice", "pasta", "bread", "cheese",
        "meat", "chicken", "beef", "pork", "fish", "egg", "milk", "butter", "oil", "salt",
        "bottle", "plate", "dining table"
    ]
    
    # Add any detected items that might be food-related
    detected_items = []
    for label in labels:
        # Check if the label is food-related or contains food-related keywords
        label_lower = label.lower()
        if any(food in label_lower for food in food_related) or label_lower in food_related:
            detected_items.append(label)
    
    # If no food items detected, try to extract potential food items from the caption
    if not detected_items:
        print("No food items detected directly, will rely on caption analysis")
    
    return list(set(detected_items))

def generate_recipe(ingredients: list) -> str:
    print(f"Generating recipe for ingredients: {ingredients}")
    # Filter out non-food items from ingredients
    food_items = []
    for item in ingredients:
        item_lower = item.lower()
        if item_lower not in ['person', 'people', 'man', 'woman', 'child', 'boy', 'girl', 'a', 'the', 'and', 'with', 'of', 'in', 'on']:
            food_items.append(item)
    
    if not food_items:
        return "No food items detected to generate a recipe."
    
    print(f"Filtered food items: {food_items}")
    
    # Try to use GPT-2 for recipe generation if available
    try:
        # Create a prompt for recipe generation
        prompt = f"Recipe with ingredients: {', '.join(food_items)}\n\n"
        print(f"Using prompt for recipe generation: {prompt}")
        
        # Generate recipe using GPT-2
        inputs = recipe_tokenizer(prompt, return_tensors="pt")
        outputs = recipe_model.generate(
            inputs["input_ids"],
            max_length=300,  # Increased max length for more detailed recipes
            num_return_sequences=1,
            temperature=0.8,  # Slightly increased temperature for more creativity
            no_repeat_ngram_size=2
        )
        recipe = recipe_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Clean up the generated recipe
        recipe = recipe.replace(prompt, "")
        print(f"Generated recipe from model: {recipe[:100]}...")
        
        # If the recipe is too short or doesn't look like a proper recipe, fall back to template
        if len(recipe.split('\n')) < 3 or 'ingredient' not in recipe.lower():
            print("Generated recipe doesn't look complete, falling back to template")
            raise Exception("Recipe doesn't look complete")
        
        return recipe
    
    except Exception as e:
        print(f"Error generating recipe with GPT-2: {e}")
        print("Falling back to template-based recipe generation")
        
        # Fallback to template-based recipe generation
        recipe_templates = [
            "RECIPE_TITLE\n\nIngredients:\n{ingredients}\n\nInstructions:\n1. Preheat the oven to 375°F (190°C).\n2. Prepare all ingredients: wash, peel, and chop as needed.\n3. {step_with_main_ingredient}\n4. {step_with_secondary_ingredient}\n5. Cook for about 20-25 minutes until done.\n6. Season with salt, pepper, and herbs to taste.\n7. Serve hot and enjoy!",
            
            "RECIPE_TITLE\n\nIngredients:\n{ingredients}\n\nInstructions:\n1. Heat oil in a large pan over medium heat.\n2. {step_with_main_ingredient}\n3. {step_with_secondary_ingredient}\n4. Add the remaining ingredients and cook for 10-15 minutes.\n5. Season with salt and pepper to taste.\n6. Garnish and serve immediately.",
            
            "RECIPE_TITLE\n\nIngredients:\n{ingredients}\n\nInstructions:\n1. {step_with_main_ingredient}\n2. In a separate bowl, combine all spices and seasonings.\n3. {step_with_secondary_ingredient}\n4. Mix everything together and cook for 15-20 minutes.\n5. Check for doneness and adjust seasonings if needed.\n6. Let rest for 5 minutes before serving."
        ]
        
        # Select a random template
        template = random.choice(recipe_templates)
        
        # Add some common cooking ingredients that might not be in the image
        common_additions = ["Salt and pepper to taste", "2 tablespoons olive oil", "1 clove garlic, minced", 
                           "Fresh herbs for garnish", "1/2 teaspoon paprika", "1 tablespoon butter"]
        
        # Add 2-3 common ingredients to the food items
        food_items.extend(random.sample(common_additions, min(3, len(common_additions))))
        
        # Format the ingredients list
        ingredients_text = "\n".join([f"- {item}" for item in food_items])
        
        # Get the main ingredient (first one in the list)
        main_ingredient = food_items[0] if food_items else "ingredients"
        secondary_ingredient = food_items[1] if len(food_items) > 1 else "other ingredients"
        
        # Create steps with the main ingredient
        steps_with_main = [
            f"Add the {main_ingredient} to the pan and sauté until golden.",
            f"Place the {main_ingredient} in a baking dish and season well.",
            f"Combine the {main_ingredient} with spices and mix thoroughly.",
            f"Cut the {main_ingredient} into bite-sized pieces and set aside.",
            f"In a large bowl, marinate the {main_ingredient} with olive oil and spices."
        ]
        
        # Create steps with the secondary ingredient
        steps_with_secondary = [
            f"Add the {secondary_ingredient} and cook for another 5 minutes.",
            f"Sprinkle the {secondary_ingredient} over the top and continue cooking.",
            f"Mix in the {secondary_ingredient} until well combined.",
            f"Layer the {secondary_ingredient} on top and bake until golden.",
            f"Stir in the {secondary_ingredient} and simmer for 10 minutes."
        ]
        
        # Replace placeholders in the template
        recipe = template.replace("{ingredients}", ingredients_text)
        recipe = recipe.replace("{step_with_main_ingredient}", random.choice(steps_with_main))
        recipe = recipe.replace("{step_with_secondary_ingredient}", random.choice(steps_with_secondary))
        
        # Generate a title based on ingredients
        if "potato" in str(food_items).lower():
            titles = ["Delicious Potato Dish", "Roasted Potato Medley", "Potato Comfort Food", "Creamy Potato Casserole", "Herb-Infused Potato Recipe"]
        elif "chicken" in str(food_items).lower():
            titles = ["Savory Chicken Recipe", "Herb-Roasted Chicken", "Classic Chicken Dish", "Tender Chicken Delight", "Spiced Chicken Creation"]
        elif "vegetable" in str(food_items).lower() or "vegetables" in str(food_items).lower():
            titles = ["Garden Vegetable Medley", "Roasted Vegetable Platter", "Seasonal Vegetable Dish", "Colorful Vegetable Stir-Fry", "Vegetable Harmony Bowl"]
        elif "beef" in str(food_items).lower():
            titles = ["Hearty Beef Stew", "Tender Beef Recipe", "Savory Beef Dish", "Slow-Cooked Beef Delight", "Spiced Beef Creation"]
        elif "fish" in str(food_items).lower() or "salmon" in str(food_items).lower():
            titles = ["Delicate Fish Recipe", "Perfectly Seasoned Fish", "Baked Fish Delight", "Zesty Fish Creation", "Herb-Crusted Fish Dish"]
        else:
            titles = [f"{main_ingredient.title()} Special", f"Homemade {main_ingredient.title()} Recipe", f"Easy {main_ingredient.title()} Dish", 
                     f"Gourmet {main_ingredient.title()} Creation", f"Delicious {main_ingredient.title()} Medley"]
        
        recipe = recipe.replace("RECIPE_TITLE", random.choice(titles))
        print(f"Generated template recipe: {recipe[:100]}...")
        
        return recipe

# --- OpenCV Webcam Loop ---
def run_webcam():
    cap = cv2.VideoCapture(0)
    print("🎥 Webcam started. Press 'c' to capture, 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to capture frame.")
            break

        cv2.imshow("RecipeSnap - Press 'c' to Capture", frame)
        key = cv2.waitKey(1)

        if key == ord('c'):
            print("📸 Capturing image...")
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_frame)

            print("🔍 Detecting ingredients...")
            caption = get_caption(pil_img)
            detected = get_detected_ingredients(pil_img)

            ingredients = list(set(detected + caption.split()))
            print("🥦 Ingredients:", ingredients)

            print("👩‍🍳 Generating recipe...")
            recipe = generate_recipe(ingredients)

            print("\n🍽️ Recipe from AI:\n")
            print(recipe)
            print("\n🔄 Ready for another capture...\n")

        elif key == ord('q'):
            print("👋 Exiting.")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_webcam()
