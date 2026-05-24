import re
import random
import google.generativeai as genai
from utils import taste_model


def predict_recipes(ingredients, mood, diet, skill, mode, user_id="default"):
    """
    Predicts 3 personalized recipes using Gemini 2.5 Flash model.
    Returns beautifully formatted recipe text (title, description, steps, score).
    """
    try:
        model = genai.GenerativeModel("models/gemini-2.5-flash")

        # Load user's taste preferences
        user_taste = taste_model.load_taste_profile(user_id)
        taste_summary = ", ".join([f"{k}: {v}" for k, v in user_taste.items()])

        # Prompt
        prompt = f"""
You are a Gen Z-friendly Indian cooking assistant.

User wants to cook using: {', '.join(ingredients)}.

Profile:
- Mood: {mood}
- Diet: {diet}
- Skill level: {skill}
- Cooking mode: {mode}
- Taste profile: {taste_summary}

Suggest 3 Indian or fusion-style recipes.

For each recipe, use **this exact clean format** (NO JSON, NO dicts):

🍛 Recipe Title 🍛

📝 Short one-line description.

👩‍🍳 Steps to make it:
1. Step one
2. Step two
3. Step three (keep it short & fun)

⭐ Score: <random value between 0.6 and 1.0> ⭐

Formatting rules:
- Separate each recipe with one blank line.
- Use emojis naturally.
- Keep tone chill, creative, and clear.
"""

        # Generate Gemini response
        response = model.generate_content(prompt)
        text = getattr(response, "text", "").strip()

        # 🧹 Clean markdown artifacts
        text = re.sub(r"```[a-zA-Z]*", "", text).replace("```", "").strip()

        # Add scores if missing
        if "⭐ Score:" not in text:
            recipes = re.split(r"\n\s*\n", text)
            clean_text = ""
            for recipe in recipes:
                if recipe.strip():
                    score = round(random.uniform(0.6, 1.0), 2)
                    clean_text += f"{recipe.strip()}\n⭐ Score: {score} ⭐\n\n"
            text = clean_text.strip()

        # ✅ Ensure nice line spacing
        text = re.sub(r"\n{3,}", "\n\n", text)

        print("\n========= 🍳 GEMINI RECIPE RESPONSE 🍳 =========\n")
        print(text)
        print("\n==============================================\n")

        return text

    except Exception as e:
        print(f"❌ Error in predict_recipes: {e}")
        return f"⚠️ Error loading recipes: {str(e)}"

