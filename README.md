# 🧠 RecipeSnap - AI Cooking Assistant from Your Fridge

**RecipeSnap** is a real-time AI cooking assistant that captures an image from your webcam, detects ingredients using object detection and image captioning, and then generates a recipe using GPT-2.

---

## 📸 What It Does

1. Takes a real-time webcam image.
2. Detects ingredients using:
   - **Image Captioning**
   - **Object Detection**
3. Uses **GPT-2** to suggest a simple recipe based on detected ingredients.

---

## 🧠 Models Used

| Task                 | Model Used                                | Source                         |
|----------------------|--------------------------------------------|--------------------------------|
| Image Captioning     | `nlpconnect/vit-gpt2-image-captioning`     | Hugging Face Transformers      |
| Object Detection     | `facebook/detr-resnet-50`                  | Hugging Face Transformers      |
| Recipe Generation    | `gpt2`                                     | Hugging Face Transformers      |

---

## 🚀 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/recipesnap.git
cd recipesnap
pip install torch torchvision transformers pillow opencv-python
python app.py
npm install
npm run dev
```

### 💡 Example Flow
You open the app → Webcam feed starts.

The image is captured or Uploaded.

The image is processed:

Caption is generated.

Objects are detected.

Ingredients are inferred.

GPT-2 suggests a recipe using only those ingredients.

You cook and enjoy! 🍳
