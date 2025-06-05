
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from heston import run_enhanced_analysis
import numpy as np

app = FastAPI()

# Allow frontend to access this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ForecastResponse(BaseModel):
    symbol: str
    price: float
    volatility: float
    expectedReturn: float
    sharpeRatio: float
    confidence: float
    forecast: dict

@app.get("/api/heston", response_model=ForecastResponse)
def analyze(symbol: str = Query("AAPL")):
    model, S_paths, v_paths = run_enhanced_analysis(symbol, forecast_months=6)

    final_prices = S_paths[:, -1]
    price = model.S0
    expected_return = float(np.mean((final_prices - price) / price))
    volatility = float(np.std((final_prices - price) / price))
    sharpe = expected_return / volatility if volatility > 0 else 0
    prob_profit = float(np.mean(final_prices > price))

    timeline = [round(i * (6/252), 2) for i in range(S_paths.shape[1])]
    median = list(np.percentile(S_paths, 50, axis=0))
    lower = list(np.percentile(S_paths, 5, axis=0))
    upper = list(np.percentile(S_paths, 95, axis=0))

    return {
        "symbol": symbol.upper(),
        "price": round(price, 2),
        "volatility": round(volatility, 4),
        "expectedReturn": round(expected_return, 4),
        "sharpeRatio": round(sharpe, 2),
        "confidence": round(prob_profit, 4),
        "forecast": {
            "timeline": timeline,
            "median": median,
            "lower": lower,
            "upper": upper,
            "probProfit": prob_profit
        }
    }
