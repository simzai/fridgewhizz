




import os
from dotenv import load_dotenv
from google import genai
from PIL import Image
from io import BytesIO
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

# ---------------------------
# 🔐 LOAD ENV
# ---------------------------
load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# ---------------------------
# 🧠 DEVICE SETUP
# ---------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔄 Loading BLIP model on {device.upper()}...")

# ---------------------------
# 🤖 BLIP MODEL (SAFE WAY - NO PIPELINE)
# ---------------------------
processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

model.to(device)

print("✅ BLIP fallback model ready.\n")


# ---------------------------
# 🧠 GEMINI FUNCTION
# ---------------------------
def gemini_detect(image_bytes: bytes):
    response = client.models.generate_content(
        model="gemini-1.5-flash",
        contents=[
            "List all visible food ingredients or food items in this image, separated by commas.",
            {
                "mime_type": "image/jpeg",
                "data": image_bytes
            }
        ]
    )

    text = response.text.strip() if response.text else ""
    return text


# ---------------------------
# 🪄 BLIP FUNCTION
# ---------------------------
def blip_caption(image_bytes: bytes):
    image = Image.open(BytesIO(image_bytes)).convert("RGB")

    inputs = processor(images=image, return_tensors="pt").to(device)

    output = model.generate(**inputs)

    caption = processor.decode(output[0], skip_special_tokens=True)

    return caption


# ---------------------------
# 🍅 FOOD KEYWORDS (fallback parsing)
# ---------------------------
FOOD_KEYWORDS = [
    "tomato", "cheese", "onion", "rice", "bread", "butter", "milk", "egg",
    "chicken", "paneer", "carrot", "lettuce", "potato", "beans", "apple",
    "banana", "chocolate", "pizza", "burger", "salad", "sauce", "pasta",
    "noodles", "curry", "spinach", "mushroom", "fish", "meat", "dal",
    "chapati", "roti", "idli", "dosa", "sambar", "poha", "paratha", "chutney"
]


def extract_food_words(text: str):
    text = text.lower()
    return [word for word in FOOD_KEYWORDS if word in text]


# ---------------------------
# 🚀 MAIN FUNCTION
# ---------------------------
def detect_ingredients(image_bytes: bytes):
    """
    Gemini → primary
    BLIP → fallback
    """

    # ---------------- GEMINI ----------------
    try:
        print("🧠 Trying Gemini...")
        text = gemini_detect(image_bytes)

        if text:
            ingredients = [i.strip().lower() for i in text.split(",") if i.strip()]
            print("✅ Using Gemini")
            return {
                "ingredients": ingredients,
                "source": "Gemini"
            }

        raise ValueError("Empty Gemini response")

    except Exception as e:
        print(f"⚠️ Gemini failed: {e}")
        print("↩️ Switching to BLIP fallback...")

    # ---------------- BLIP ----------------
    try:
        caption = blip_caption(image_bytes)
        print("🧠 BLIP caption:", caption)

        ingredients = extract_food_words(caption)

        if not ingredients:
            ingredients = [caption.strip().lower()]

        print("✅ Using BLIP fallback")
        return {
            "ingredients": ingredients,
            "source": "BLIP"
        }

    except Exception as e:
        print(f"❌ BLIP failed: {e}")

        return {
            "ingredients": ["error processing image"],
            "source": "none"
        }