import json
import os
import numpy as np

# Directory and file for user taste data
DATA_DIR = "data"
TASTE_FILE = os.path.join(DATA_DIR, "user_taste.json")

# Ensure the directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# Default taste model
default_profile = {
    "spicy": 5,
    "sweet": 3,
    "garlic": 1,        # 1 = love, 0 = neutral, -1 = hate
    "coriander": -1,
    "veg_preference": 1,  # 1 = veg, 0 = non-veg
    "street_food": 4,
    "healthy": 2
}

def load_taste_profile(user_id="default"):
    """Load taste preferences from file (or return defaults)"""
    if not os.path.exists(TASTE_FILE):
        save_taste_profile(default_profile, user_id)
        return default_profile

    with open(TASTE_FILE, "r") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            data = {}

    return data.get(user_id, default_profile)

def save_taste_profile(profile, user_id="default"):
    """Save or update taste preferences"""
    if os.path.exists(TASTE_FILE):
        with open(TASTE_FILE, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = {}
    else:
        data = {}

    data[user_id] = profile

    with open(TASTE_FILE, "w") as f:
        json.dump(data, f, indent=2)

    print(f"✅ Taste profile saved for user: {user_id}")

def update_preference(user_id, feedback):
    """
    Update preferences from user feedback.
    feedback = {"spicy": +1, "sweet": -1}
    """
    profile = load_taste_profile(user_id)
    for key, change in feedback.items():
        if key in profile:
            profile[key] = min(max(profile[key] + change, 0), 10)  # keep within 0–10
    save_taste_profile(profile, user_id)
    print(f"🔥 Updated taste profile for {user_id}: {feedback}")

def score_recipe(recipe_tags, user_id="default"):
    """
    Compare recipe features with user's profile
    to generate a match score (0–1)
    """
    profile = load_taste_profile(user_id)
    recipe_vec = np.array([recipe_tags.get(k, 0) for k in profile.keys()])
    taste_vec = np.array(list(profile.values()))

    denom = np.linalg.norm(recipe_vec) * np.linalg.norm(taste_vec)
    if denom == 0:
        return 0.0

    similarity = np.dot(recipe_vec, taste_vec) / denom
    return round(float(similarity), 2)
