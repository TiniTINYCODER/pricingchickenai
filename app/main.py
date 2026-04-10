import csv
import json
import os
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import shutil
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.demand_model import predict_demand
from app.pricing_engine import calculate_price
from app.llm_service import ask_llm

app = FastAPI(title="Whole Chicken Without Skin Pricing AI", version="2.0")

# Allow the dashboard HTML to call from any origin (file://, localhost, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "sales_data.csv"))
DASHBOARD_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dashboard"))

app.mount("/dashboard-static", StaticFiles(directory=DASHBOARD_PATH), name="dashboard-static")

@app.get("/")
def root():
    return FileResponse(os.path.join(DASHBOARD_PATH, "index.html"))


# ──────────────────────────────────────────────
# 0. CSV FILE UPLOAD
# ──────────────────────────────────────────────
@app.post("/upload")
async def upload_sales_data(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")
    try:
        data_dir = os.path.dirname(DATA_PATH)
        os.makedirs(data_dir, exist_ok=True)

        # Write to a temporary file first (atomic-ish write)
        tmp_path = DATA_PATH + ".tmp"
        contents = await file.read()
        with open(tmp_path, "wb") as tmp:
            tmp.write(contents)

        # If the destination already exists, ensure it is writable on Windows
        if os.path.exists(DATA_PATH):
            import stat
            os.chmod(DATA_PATH, stat.S_IWRITE | stat.S_IREAD)
            os.remove(DATA_PATH)

        os.rename(tmp_path, DATA_PATH)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")
    finally:
        await file.seek(0)       # best-effort rewind; ignore errors
    return {"message": "File uploaded successfully", "filename": file.filename}


# ──────────────────────────────────────────────
# 1. PRICE + DEMAND PREDICTION
# ──────────────────────────────────────────────
@app.get("/price")
def get_price(hour: int, day: int, stock: int, price: float, season: str, festival: str, weather: str):
    predicted = predict_demand(hour, day, stock, price, weather, season, festival)
    suggested_price = calculate_price(price, predicted, stock)
    return {
        "predicted_demand": round(float(predicted), 2),
        "suggested_price": suggested_price
    }


# ──────────────────────────────────────────────
# 2. HISTORICAL SALES DATA (for charts)
# ──────────────────────────────────────────────
@app.get("/history")
def get_history():
    rows = []
    try:
        with open(DATA_PATH, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append({
                    "date": row.get("date", ""),
                    "hour": int(str(row.get("hour", "0")).split(":")[0]),
                    "day": row.get("day", ""),
                    "weather": row.get("weather", ""),
                    "season": row.get("season", ""),
                    "festival": row.get("festival", "none"),
                    "stock_start": int(row.get("stock_start", 0)),
                    "sold": int(row.get("sold", 0)),
                    "remaining": int(row.get("remaining", 0)),
                    "price": float(row.get("price", 0)),
                })
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Sales data file not found")
    return rows


# ──────────────────────────────────────────────
# 3. AI CHATBOT (LLM-powered)
# ──────────────────────────────────────────────
class ChatRequest(BaseModel):
    question: str


@app.post("/chat")
def chat(req: ChatRequest):
    # Build a short data summary for context
    rows = []
    try:
        with open(DATA_PATH, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except Exception:
        rows = []

    total_sold = sum(int(r.get("sold", 0)) for r in rows)
    avg_price = (sum(float(r.get("price", 0)) for r in rows) / len(rows)) if rows else 0
    festivals = list({r.get("festival", "none") for r in rows if r.get("festival", "none") != "none"})
    seasons = list({r.get("season", "") for r in rows})

    sales_summary = (
        f"Total records: {len(rows)}\n"
        f"Total units sold across all records: {total_sold}\n"
        f"Average price: ₹{avg_price:.2f}/kg\n"
        f"Festivals in data: {', '.join(festivals)}\n"
        f"Seasons covered: {', '.join(seasons)}\n"
        f"Today's date context: 15-03-2026 (Holi festival season, winter ending)\n"
        f"Recent high-demand observation: Holi days saw 40-50 units sold at ₹240-₹260/kg"
    )

    answer = ask_llm(req.question, sales_summary)
    return {"answer": answer}