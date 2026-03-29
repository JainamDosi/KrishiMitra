import os
import sys
import logging
from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import plotly.io as pio

# Setup path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import APP_NAME, GRADIO_SERVER_PORT, LANGUAGES

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")
logger = logging.getLogger("krishimitra.api")

app = FastAPI(title=APP_NAME)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy-loaded module references
_disease_predictor = None
_price_predictor = None
_scheme_advisor = None
_pesticide_advisor = None
_chat_engine = None
_translator = None

def get_disease_predictor():
    global _disease_predictor
    if _disease_predictor is None:
        from src.disease_predictor import get_disease_predictor as _get
        _disease_predictor = _get()
    return _disease_predictor

def get_price_predictor():
    global _price_predictor
    if _price_predictor is None:
        from src.price_predictor import get_price_predictor as _get
        _price_predictor = _get()
    return _price_predictor

def get_scheme_advisor():
    global _scheme_advisor
    if _scheme_advisor is None:
        from src.scheme_advisor import get_scheme_advisor as _get
        _scheme_advisor = _get()
    return _scheme_advisor

def get_pesticide_advisor():
    global _pesticide_advisor
    if _pesticide_advisor is None:
        from src.pesticide_advisor import get_pesticide_advisor as _get
        _pesticide_advisor = _get()
    return _pesticide_advisor

def get_chat_engine():
    global _chat_engine
    if _chat_engine is None:
        from src.chat_engine import get_chat_engine as _get
        _chat_engine = _get()
    return _chat_engine

def get_translator():
    global _translator
    if _translator is None:
        from src.translator import get_translator as _get
        _translator = _get()
    return _translator

# Root and Static File Handlers
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
public_dir = os.path.join(BASE_DIR, "public")
os.makedirs(public_dir, exist_ok=True)

from fastapi.responses import FileResponse

@app.get("/", response_class=HTMLResponse)
async def read_index():
    return FileResponse(os.path.join(public_dir, "index.html"))

# --- API Endpoints ---

@app.get("/api/config")
async def get_config():
    return {
        "languages": list(LANGUAGES.keys()),
        "app_name": APP_NAME
    }

@app.post("/api/disease/predict")
async def predict_disease(image: UploadFile = File(...), language: str = Form("English")):
    try:
        # Read image
        content = await image.read()
        
        # Save temp file
        temp_path = f"/tmp/{image.filename}"
        os.makedirs("/tmp", exist_ok=True)
        with open(temp_path, "wb") as f:
            f.write(content)

        predictor = get_disease_predictor()
        result = predictor.predict(temp_path)

        disease = result.get("disease", "Unknown")
        crop = result.get("crop", "Unknown")
        confidence = result.get("confidence", 0.0)
        is_healthy = result.get("is_healthy", False)

        status = "Healthy" if is_healthy else "Disease Detected"
        treatment_md = predictor.format_result(result)

        # Translate if needed
        lang_code = LANGUAGES.get(language, "en")
        if lang_code != "en":
            try:
                translator = get_translator()
                treatment_md = translator.translate(treatment_md, "en", lang_code)
            except Exception as e:
                logger.error(f"Translation error: {e}")

        # clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)

        return {
            "disease": disease,
            "crop": crop,
            "confidence": f"{confidence*100:.1f}%",
            "status": status,
            "treatment_md": treatment_md
        }
    except Exception as e:
        logger.error(f"Disease prediction error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/api/price/commodities")
async def get_commodities():
    try:
        commodities = get_price_predictor().get_commodities()[:100]
        return {"commodities": commodities}
    except Exception as e:
        return {"commodities": ["Wheat", "Rice", "Tomato", "Onion", "Potato"]}

@app.get("/api/price/states")
async def get_states(commodity: str):
    try:
        states = get_price_predictor().get_states(commodity)
        return {"states": states}
    except Exception as e:
        return {"states": []}

@app.get("/api/price/markets")
async def get_markets(commodity: str, state: str):
    try:
        markets = get_price_predictor().get_markets(commodity, state)
        return {"markets": markets}
    except Exception as e:
        return {"markets": []}

@app.post("/api/price/analyze")
async def analyze_price(request: Request):
    data = await request.json()
    commodity = data.get("commodity")
    state = data.get("state")
    market = data.get("market")
    days = data.get("days", 90)

    if not commodity:
        return JSONResponse(status_code=400, content={"error": "Commodity is required"})

    try:
        predictor = get_price_predictor()
        # Dataframe to dict
        current_df = predictor.get_current_prices(commodity, state if state else None, limit=15)
        current_data = current_df.to_dict(orient="records") if not current_df.empty else []

        trend_chart = predictor.get_price_trends(
            commodity,
            state=state if state else None,
            market=market if market else None,
            days=int(days)
        )
        
        prediction = predictor.predict_price(
            commodity,
            state=state if state else None,
            market=market if market else None,
            days_ahead=7,
        )
        summary = prediction.get("summary", "No prediction available")

        comparison_chart = predictor.get_price_comparison_chart(commodity)

        # Convert Plotly figures to JSON for frontend rendering
        trend_json = pio.to_json(trend_chart) if trend_chart else None
        comparison_json = pio.to_json(comparison_chart) if comparison_chart else None

        return {
            "current_prices": current_data,
            "trend_chart": trend_json,
            "prediction_summary": summary,
            "comparison_chart": comparison_json
        }
    except Exception as e:
        logger.error(f"Price analysis error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/api/scheme/ask")
async def ask_scheme_question(request: Request):
    data = await request.json()
    question = data.get("question")
    language = data.get("language", "English")

    if not question:
        return {"answer": "Please enter a question about government schemes."}

    lang_code = LANGUAGES.get(language, "en")
    try:
        advisor = get_scheme_advisor()
        answer = advisor.answer_query(question, language=lang_code)
        return {"answer": answer}
    except Exception as e:
        logger.error(f"Scheme query error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/api/scheme/list")
async def list_schemes():
    try:
        advisor = get_scheme_advisor()
        schemes = advisor.list_schemes()
        if not schemes:
            return {"markdown": "Scheme data not loaded."}

        md = "## 🏛️ Available Government Schemes\n\n"
        for i, s in enumerate(schemes, 1):
            md += (
                f"**{i}. {s['name']}**\n"
                f"   - Category: {s['category']}\n"
                f"   - {s['benefits_summary']}\n\n"
            )
        return {"markdown": md}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/api/pesticide/crops")
async def get_crops():
    try:
        crops = get_pesticide_advisor().get_crops()
        return {"crops": crops}
    except Exception:
        return {"crops": ["Rice", "Wheat", "Tomato", "Potato", "Cotton", "Soybean"]}

@app.get("/api/pesticide/stages")
async def get_stages(crop: str):
    try:
        stages = get_pesticide_advisor().get_stages(crop)
        return {"stages": stages}
    except Exception:
        return {"stages": ["Vegetative", "Flowering", "Fruiting"]}

@app.post("/api/pesticide/recommend")
async def get_pesticide_recommendation(request: Request):
    data = await request.json()
    crop = data.get("crop")
    stage = data.get("stage", "")
    problem = data.get("problem", "")
    prefer_organic = data.get("prefer_organic", False)
    language = data.get("language", "English")

    if not crop:
        return {"recommendation": "Please select a crop."}

    lang_code = LANGUAGES.get(language, "en")
    try:
        advisor = get_pesticide_advisor()
        rec = advisor.get_recommendation(
            crop=crop,
            stage=stage,
            problem=problem,
            prefer_organic=prefer_organic,
            language=lang_code,
        )
        return {"recommendation": rec}
    except Exception as e:
        logger.error(f"Pesticide recommendation error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/api/chat")
async def chat_handler(request: Request):
    data = await request.json()
    message = data.get("message")
    language = data.get("language", "English")

    if not message:
        return {"response": ""}

    lang_code = LANGUAGES.get(language, "en")
    try:
        engine = get_chat_engine()
        result = engine.process_query(message, language=lang_code)
        return {"response": result.get("response", "I'm sorry, I couldn't process your request.")}
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

# Serve static files from public/ — MUST be after all API routes
# so that /api/* routes are not shadowed by the catch-all mount
app.mount("/", StaticFiles(directory=public_dir), name="public")

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
