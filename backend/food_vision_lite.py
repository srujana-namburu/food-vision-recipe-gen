import cv2
from PIL import Image
import torch
from transformers import (
    VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer,
    DetrForObjectDetection, DetrImageProcessor,
)

# --- Load Models ---
print("Loading lightweight models for testing...")

# Image Captioning - Using a smaller model
caption_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
caption_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
caption_tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# Object Detection - Using DETR but with lower threshold
det_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
det_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

print("Models loaded successfully.")

# --- Helper Functions ---
def get_caption(image: Image.Image) -> str:
    inputs = caption_processor(images=image, return_tensors="pt").pixel_values
    outputs = caption_model.generate(inputs, max_length=64)
    return caption_tokenizer.decode(outputs[0], skip_special_tokens=True)

def get_detected_ingredients(image: Image.Image) -> list:
    inputs = det_processor(images=image, return_tensors="pt")
    outputs = det_model(**inputs)
    target_sizes = torch.tensor([image.size[::-1]])
    results = det_processor.post_process_object_detection(outputs, threshold=0.5, target_sizes=target_sizes)[0]
    
    # Filter for food-related objects
    food_categories = ["banana", "apple", "sandwich", "orange", "broccoli", "carrot", 
                       "hot dog", "pizza", "donut", "cake", "bowl", "cup", "fork", "knife", 
                       "spoon", "food", "fruit", "vegetable"]
    
    labels = []
    for i, label in enumerate(results["labels"]):
        label_name = det_model.config.id2label[label.item()]
        if any(food in label_name.lower() for food in food_categories):
            labels.append(label_name)
    
    return list(set(labels))

def generate_recipe(ingredients: list) -> str:
    """Simple mock recipe generator for testing"""
    if not ingredients:
        return "I couldn't identify any food ingredients in the image. Please try again with a clearer image of food items."
    
    # Basic template-based recipe generation
    recipe = f"""Here's a simple recipe using {', '.join(ingredients)}:

Recipe: {' '.join(ingredients).title()} Special

Ingredients:
{chr(10).join(['- ' + ing for ing in ingredients])}

Instructions:
1. Prepare all ingredients by washing and cutting them as needed.
2. Combine the ingredients in a bowl.
3. Season with salt and pepper to taste.
4. Cook for about 15-20 minutes until done.
5. Serve hot and enjoy!

This is a simple recipe generated for testing purposes. For more detailed recipes, the full AI model would provide better results.
"""
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
