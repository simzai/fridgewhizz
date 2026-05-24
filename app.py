from fastapi import FastAPI, File, Form, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from utils.ingredient_detector import detect_ingredients
from utils.recipe_predictor import predict_recipes
from utils.taste_model import update_preference, load_taste_profile
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI(debug=True)

# Mount static and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    print("✅ Home page loaded")
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/analyze", response_class=HTMLResponse)
async def analyze(
    request: Request,
    file: UploadFile = File(...),
    mood: str = Form(...),
    diet: str = Form(...),
    skill: str = Form(...),
    mode: str = Form(...),
    user_id: str = Form("default")
):
    print(f"\n🚀 Analyzing for user: {user_id}")
    print(f"🧠 Inputs → mood={mood}, diet={diet}, skill={skill}, mode={mode}")

    image_bytes = await file.read()

    # Step 1: Detect ingredients
    ingredients = detect_ingredients(image_bytes)
    print(f"🍅 Ingredients detected: {ingredients}")

    # Step 2: Load taste profile
    user_taste = load_taste_profile(user_id)
    print(f"👤 Current taste profile: {user_taste}")

    # Step 3: Update taste model slightly based on context
    feedback = {}
    if mood.lower() in ["happy", "excited"]:
        feedback["spicy"] = +1
    elif mood.lower() in ["sad", "calm"]:
        feedback["sweet"] = +1
    if diet.lower() == "healthy":
        feedback["healthy"] = +1
    if skill.lower() == "beginner":
        feedback["street_food"] = +1

    if feedback:
        print(f"🎯 Context-based feedback: {feedback}")
        update_preference(user_id, feedback)
        print("💾 Taste profile updated and saved!")

    # Step 4: Predict recipes
    recipes = predict_recipes(ingredients, mood, diet, skill, mode)
    print(f"🍽️ Recommended recipes: {len(recipes)} found")

    return templates.TemplateResponse(
        "results.html",
        {
            "request": request,
            "ingredients": ingredients,
            "recipes": recipes,
            "mood": mood,
            "diet": diet,
            "skill": skill,
            "mode": mode,
            "user_id": user_id,
        },
    )


@app.post("/feedback", response_class=HTMLResponse)
async def feedback(
    request: Request,
    user_id: str = Form("default"),
    feedback: str = Form(...),
):
    """
    Handles user feedback to update taste preferences.
    👍 = increase spicy/street_food
    👎 = increase healthy preference
    """
    print(f"\n🗳️ User feedback received: {feedback} (user={user_id})")

    if feedback == "like":
        change = {"spicy": +1, "street_food": +1}
        message = "🔥 Awesome! Boosted your spicy & street-food love!"
    else:
        change = {"healthy": +1, "spicy": -1}
        message = "✨ Got it! We'll tone down the spice next time."

    update_preference(user_id, change)
    print(f"💾 Taste model updated → {change}")

    return templates.TemplateResponse(
        "thanks.html",
        {"request": request, "message": message},
    )
