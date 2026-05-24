import os
from dotenv import load_dotenv
import google.generativeai as genai
from google.api_core import exceptions
from transformers import pipeline
from io import BytesIO
from PIL import Image
import torch

# 🔐 Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# ✅ Force CPU-only if no GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔄 Loading fallback model (BLIP) on {device.upper()}...")

# ✅ Correct way: use device=0 for GPU, -1 for CPU automatically
device_arg = 0 if device == "cuda" else -1
hf_model = pipeline(
    "image-to-text",
    model="Salesforce/blip-image-captioning-base",
    device=device_arg
)

print("✅ Fallback model ready.\n")

# 🍅 Common food word list for BLIP caption parsing
FOOD_KEYWORDS = [
    "tomato", "cheese", "onion", "rice", "bread", "butter", "milk", "egg",
    "chicken", "paneer", "carrot", "lettuce", "potato", "beans", "apple",
    "banana", "chocolate", "pizza", "burger", "salad", "sauce", "pasta",
    "noodles", "curry", "spinach", "mushroom", "fish", "meat", "dal",
    "chapati", "roti", "idli", "dosa", "sambar", "poha", "paratha", "chutney"
]


def detect_food_words(caption: str):
    """Extract known food-related words from BLIP caption."""
    caption_lower = caption.lower()
    return [word for word in FOOD_KEYWORDS if word in caption_lower]


def detect_ingredients(image_bytes: bytes):
    """
    Detect visible food ingredients using Gemini 2.5 Flash.
    Falls back to BLIP (local model) if Gemini fails or quota ends.
    """
    try:
        # 🧠 Try Gemini model first
        model = genai.GenerativeModel("models/gemini-2.5-flash-image")
        response = model.generate_content([
            "List all visible food ingredients or food items in this image, separated by commas.",
            {"mime_type": "image/jpeg", "data": image_bytes}
        ])

        text = getattr(response, "text", "").strip()
        print("🧠 Gemini raw response:", text)

        if not text:
            raise ValueError("Empty response from Gemini")

        ingredients = [i.strip().lower() for i in text.split(",") if i.strip()]
        print("✅ Using Gemini model.")
        return {"ingredients": ingredients, "source": "Gemini"}

    except (exceptions.ResourceExhausted, exceptions.GoogleAPIError, exceptions.ServiceUnavailable) as g_error:
        print(f"⚠️ Gemini API/Quota error: {g_error}")
        print("↩️ Switching to fallback (BLIP)...")

    except Exception as e:
        print(f"⚠️ Gemini unexpected error: {e}")
        print("↩️ Switching to fallback (BLIP)...")

    # 🪄 Fallback to BLIP
    try:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        result = hf_model(image)[0]
        caption = result.get("generated_text", "")
        print("🧠 BLIP raw caption:", caption)

        # Parse likely ingredients
        ingredients = detect_food_words(caption)
        if not ingredients:
            ingredients = [caption.strip().lower()]

        print("✅ Using BLIP fallback model.")
        return {"ingredients": ingredients, "source": "BLIP fallback"}

    except Exception as hf_error:
        print(f"❌ Fallback model failed: {hf_error}")
        return {"ingredients": ["Error processing image"], "source": "None"}
