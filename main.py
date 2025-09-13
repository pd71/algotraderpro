# main.py
from fastapi import FastAPI, Depends, HTTPException, Body, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from sqlalchemy import create_engine, Column, Integer, String, Float, JSON, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from sqlalchemy.ext.mutable import MutableList
from pydantic import BaseModel
import yfinance as yf
from alpha_vantage.timeseries import TimeSeries
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import os
import hashlib
from datetime import datetime

# Config
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "MF7A1PMD5RAIF47Q")

# Database setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "..", "trading_platform.db")
DATABASE_URL = f"sqlite:///{DB_PATH}"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# ----- Models -----
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    is_premium = Column(Boolean, default=False)
    strategies = relationship("Strategy", back_populates="owner")
    portfolios = relationship("Portfolio", back_populates="owner")

class Strategy(Base):
    __tablename__ = "strategies"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    description = Column(String)
    blocks = Column(JSON)
    owner_id = Column(Integer, ForeignKey("users.id"))
    owner = relationship("User", back_populates="strategies")
    backtests = relationship("Backtest", back_populates="strategy")

class Backtest(Base):
    __tablename__ = "backtests"
    id = Column(Integer, primary_key=True, index=True)
    strategy_id = Column(Integer, ForeignKey("strategies.id"))
    start_date = Column(String)
    end_date = Column(String)
    initial_capital = Column(Float)
    results = Column(JSON)
    strategy = relationship("Strategy", back_populates="backtests")

class Portfolio(Base):
    __tablename__ = "portfolios"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    is_paper = Column(Boolean, default=True)
    capital = Column(Float, default=100000.0)
    positions = Column(MutableList.as_mutable(JSON), default=[])
    owner_id = Column(Integer, ForeignKey("users.id"))
    owner = relationship("User", back_populates="portfolios")

# Create app
app = FastAPI()

# Create tables
Base.metadata.create_all(bind=engine)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For dev, change in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Password hashing
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

# Startup DB population
@app.on_event("startup")
def startup_event():
    db: Session = SessionLocal()
    try:
        user = db.query(User).filter(User.username == "demo").first()
        if not user:
            hashed_pw = hash_password("demo")
            user = User(username="demo", email="demo@example.com", hashed_password=hashed_pw)
            db.add(user)
            db.commit()
            db.refresh(user)

        strategies_data = [
            {"name": "RSI Mean Reversion", "blocks": [{"type": "indicator", "name": "RSI", "params": {"period": 14}}]},
            {"name": "MACD Crossover", "blocks": [{"type": "indicator", "name": "MACD", "params": {"fast": 12, "slow": 26, "signal": 9}}]},
            {"name": "Momentum", "blocks": [{"type": "indicator", "name": "Momentum", "params": {"period": 10}}]},
            {"name": "Breakout", "blocks": [{"type": "indicator", "name": "Breakout", "params": {"level": 100}}]},
            {"name": "Moving Average Cross", "blocks": [{"type": "indicator", "name": "SMA", "params": {"period": 50}}]},
            {"name": "Bollinger Bands", "blocks": [{"type": "indicator", "name": "Bollinger", "params": {"period": 20, "std": 2}}]},
            {"name": "Supertrend", "blocks": [{"type": "indicator", "name": "Supertrend", "params": {"period": 10, "multiplier": 3}}]},
            {"name": "VWAP", "blocks": [{"type": "indicator", "name": "VWAP"}]},
            {"name": "ADX Trend", "blocks": [{"type": "indicator", "name": "ADX", "params": {"period": 14}}]},
            {"name": "Stochastic Oscillator", "blocks": [{"type": "indicator", "name": "Stochastic", "params": {"k": 14, "d": 3}}]},
            {"name": "ATR Volatility", "blocks": [{"type": "indicator", "name": "ATR", "params": {"period": 14}}]},
            {"name": "Donchian Channel", "blocks": [{"type": "indicator", "name": "Donchian", "params": {"period": 20}}]},
            {"name": "Ichimoku Cloud", "blocks": [{"type": "indicator", "name": "Ichimoku"}]},
            {"name": "Parabolic SAR", "blocks": [{"type": "indicator", "name": "ParabolicSAR", "params": {"acceleration": 0.02}}]},
            {"name": "CCI Strategy", "blocks": [{"type": "indicator", "name": "CCI", "params": {"period": 20}}]},
        ]
        for strat_data in strategies_data:
            strat = db.query(Strategy).filter(Strategy.name == strat_data["name"], Strategy.owner_id == user.id).first()
            if not strat:
                strat = Strategy(name=strat_data["name"], description="", blocks=strat_data["blocks"], owner_id=user.id)
                db.add(strat)
        db.commit()

        portfolio = db.query(Portfolio).filter(Portfolio.name == "Demo Portfolio", Portfolio.owner_id == user.id).first()
        if not portfolio:
            portfolio = Portfolio(name="Demo Portfolio", is_paper=True, capital=100000.0, owner_id=user.id, positions=[])
            db.add(portfolio)
            db.commit()
    finally:
        db.close()

# ----- Pydantic Models -----
class UserCreate(BaseModel):
    username: str
    email: str
    password: str

class UserOut(BaseModel):
    id: int
    username: str
    email: str
    is_premium: bool
    class Config:
        orm_mode = True

class StrategyCreate(BaseModel):
    name: str
    description: Optional[str] = None
    blocks: List[Dict]

class StrategyOut(BaseModel):
    id: int
    name: str
    description: Optional[str]
    blocks: List[Dict]
    class Config:
        orm_mode = True

class BacktestCreate(BaseModel):
    strategy_name: str
    start_date: str
    end_date: str
    initial_capital: float = 100000.0
    symbols: List[str]

class BacktestOut(BaseModel):
    id: int
    results: Dict
    class Config:
        orm_mode = True

class PortfolioCreate(BaseModel):
    name: str
    is_paper: bool = True
    capital: float = 100000.0

class PortfolioOut(BaseModel):
    id: int
    name: str
    is_paper: bool
    capital: float
    positions: List[Dict]
    class Config:
        orm_mode = True

# ----- Utilities -----
import math
def calculate_metrics(trades, initial_capital, final_value, data):
    # Calculate returns
    returns = []
    prev = initial_capital
    for t in trades:
        if t['action'] == 'buy':
            prev = t['price']
        elif t['action'] == 'sell':
            ret = (t['price'] - prev) / prev
            returns.append(ret)
            prev = t['price']
    # Sharpe ratio
    if returns:
        mean_ret = np.mean(returns)
        std_ret = np.std(returns)
        sharpe = (mean_ret / std_ret) * math.sqrt(252) if std_ret != 0 else 0
    else:
        sharpe = 0
    # Max drawdown
    equity_curve = [initial_capital]
    capital = initial_capital
    positions = 0.0
    for t in trades:
        if t['action'] == 'buy':
            positions = capital / t['price']
            capital = 0.0
        elif t['action'] == 'sell':
            capital = positions * t['price']
            positions = 0.0
        equity_curve.append(capital + positions * t['price'])
    max_drawdown = 0
    peak = equity_curve[0]
    for x in equity_curve:
        if x > peak:
            peak = x
        dd = (peak - x) / peak
        if dd > max_drawdown:
            max_drawdown = dd
    # Win rate
    wins = sum(1 for r in returns if r > 0)
    win_rate = wins / len(returns) if returns else 0
    return {
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate
    }

def execute_backtest(strategy_blocks: List[Dict], data: pd.DataFrame, initial_capital: float):
    capital = initial_capital
    positions = 0.0
    trades = []

    data = data.copy()
    # Support multiple strategies
    strat = strategy_blocks[0] if strategy_blocks else {"type": "indicator", "name": "SMA", "params": {"period": 50}}
    name = strat.get("name", "SMA")
    params = strat.get("params", {})
    # Indicator calculations
    if name == "SMA":
        period = int(params.get("period", 50))
        data['SMA'] = data['Close'].rolling(window=period).mean()
        signal = data['SMA']
    elif name == "RSI":
        period = int(params.get("period", 14))
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        signal = data['RSI']
    elif name == "MACD":
        fast = int(params.get("fast", 12))
        slow = int(params.get("slow", 26))
        signal_period = int(params.get("signal", 9))
        ema_fast = data['Close'].ewm(span=fast, adjust=False).mean()
        ema_slow = data['Close'].ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal_period, adjust=False).mean()
        data['MACD'] = macd
        data['MACD_SIGNAL'] = macd_signal
        signal = data['MACD'] - data['MACD_SIGNAL']
    elif name == "Bollinger":
        period = int(params.get("period", 20))
        std = float(params.get("std", 2))
        sma = data['Close'].rolling(window=period).mean()
        upper = sma + std * data['Close'].rolling(window=period).std()
        lower = sma - std * data['Close'].rolling(window=period).std()
        data['Bollinger_Upper'] = upper
        data['Bollinger_Lower'] = lower
        signal = sma
    else:
        # Default to SMA
        period = int(params.get("period", 50))
        data['SMA'] = data['Close'].rolling(window=period).mean()
        signal = data['SMA']

    data = data.dropna(subset=[signal.name if hasattr(signal, 'name') else 'SMA'])
    last_action = None

    for i in range(len(data)):
        price = float(data['Close'].iloc[i])
        if np.isnan(price):
            continue
        # Strategy rules
        if name == "RSI":
            rsi = data['RSI'].iloc[i]
            if rsi < 30 and positions == 0:
                positions = capital / price
                capital = 0.0
                trades.append({"date": str(data.index[i]), "action": "buy", "price": price})
                last_action = "buy"
            elif rsi > 70 and positions > 0:
                capital = positions * price
                trades.append({"date": str(data.index[i]), "action": "sell", "price": price})
                positions = 0.0
                last_action = "sell"
        elif name == "MACD":
            macd_val = data['MACD'].iloc[i]
            macd_signal_val = data['MACD_SIGNAL'].iloc[i]
            if macd_val > macd_signal_val and positions == 0:
                positions = capital / price
                capital = 0.0
                trades.append({"date": str(data.index[i]), "action": "buy", "price": price})
                last_action = "buy"
            elif macd_val < macd_signal_val and positions > 0:
                capital = positions * price
                trades.append({"date": str(data.index[i]), "action": "sell", "price": price})
                positions = 0.0
                last_action = "sell"
        elif name == "Bollinger":
            lower = data['Bollinger_Lower'].iloc[i]
            upper = data['Bollinger_Upper'].iloc[i]
            if price < lower and positions == 0:
                positions = capital / price
                capital = 0.0
                trades.append({"date": str(data.index[i]), "action": "buy", "price": price})
                last_action = "buy"
            elif price > upper and positions > 0:
                capital = positions * price
                trades.append({"date": str(data.index[i]), "action": "sell", "price": price})
                positions = 0.0
                last_action = "sell"
        else:
            # Default SMA strategy
            sig = signal.iloc[i]
            if price > sig and positions == 0:
                positions = capital / price
                capital = 0.0
                trades.append({"date": str(data.index[i]), "action": "buy", "price": price})
                last_action = "buy"
            elif price < sig and positions > 0:
                capital = positions * price
                trades.append({"date": str(data.index[i]), "action": "sell", "price": price})
                positions = 0.0
                last_action = "sell"
    # Final portfolio value
    final_value = capital + positions * data['Close'].iloc[-1] if positions > 0 else capital
    metrics = calculate_metrics(trades, initial_capital, final_value, data)
    return {
        "trades": trades,
        "final_value": final_value,
        "initial_capital": initial_capital,
        "profit": final_value - initial_capital,
        "metrics": metrics
    }

# ----- API endpoint for backtesting with yfinance -----

# ----- API endpoint for live market prices -----
@app.get("/api/market_price")
def get_market_price(symbol: str):
    try:
        ticker = yf.Ticker(symbol)
        price = ticker.info.get("regularMarketPrice")
        if price is None:
            price = ticker.history(period="1d")['Close'].iloc[-1]
        return {"symbol": symbol, "price": price}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
from fastapi import Request

@app.post("/api/backtest")
async def api_backtest(backtest: BacktestCreate):
    if not backtest.symbols:
        raise HTTPException(status_code=400, detail="No symbols provided")
    all_results = {}
    for symbol in backtest.symbols:
        try:
            data = yf.download(symbol, start=backtest.start_date, end=backtest.end_date)
            if data.empty:
                all_results[symbol] = {"error": f"No data found for symbol {symbol}"}
                continue
        except Exception as e:
            all_results[symbol] = {"error": str(e)}
            continue
        results = execute_backtest(backtest.blocks if hasattr(backtest, 'blocks') else [], data, backtest.initial_capital)
        all_results[symbol] = results
    return {"results": all_results}

# ----- Serve HTML files -----
@app.get("/community")
def serve_community():
    file_path = os.path.join(os.path.dirname(__file__), "community.html")
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="text/html")
    else:
        return HTMLResponse(content="<h1>community.html not found</h1>", status_code=404)

@app.get("/profile")
def serve_profile():
    file_path = os.path.join(os.path.dirname(__file__), "profile.html")
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="text/html")
    else:
        return HTMLResponse(content="<h1>profile.html not found</h1>", status_code=404)
from fastapi.responses import FileResponse, HTMLResponse
import os

@app.get("/dashboard")
def get_dashboard():
    file_path = os.path.join(os.path.dirname(__file__), "dashboard.html")
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="text/html")
    else:
        return HTMLResponse(content="<h1>dashboard.html not found</h1>", status_code=404)

@app.get("/backtest")
def get_backtest():
    file_path = os.path.join(os.path.dirname(__file__), "backtest.html")
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="text/html")
    else:
        return HTMLResponse(content="<h1>backtest.html not found</h1>", status_code=404)

@app.get("/market")
def get_market():
    file_path = os.path.join(os.path.dirname(__file__), "market.html")
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="text/html")
    else:
        return HTMLResponse(content="<h1>market.html not found</h1>", status_code=404)

@app.get("/orders")
def get_orders():
    file_path = os.path.join(os.path.dirname(__file__), "orders.html")
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="text/html")
    else:
        return HTMLResponse(content="<h1>orders.html not found</h1>", status_code=404)

@app.get("/papertrading")
def get_papertrading():
    file_path = os.path.join(os.path.dirname(__file__), "papertrading.html")
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="text/html")
    else:
        return HTMLResponse(content="<h1>papertrading.html not found</h1>", status_code=404)

@app.get("/portfolio")
def get_portfolio():
    file_path = os.path.join(os.path.dirname(__file__), "portfolio.html")
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="text/html")
    else:
        return HTMLResponse(content="<h1>portfolio.html not found</h1>", status_code=404)

@app.get("/strategies")
def get_strategies():
    file_path = os.path.join(os.path.dirname(__file__), "strategies.html")
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="text/html")
    else:
        return HTMLResponse(content="<h1>strategies.html not found</h1>", status_code=404)

@app.get("/")
def root():
    file_path = os.path.join(os.path.dirname(__file__), "dashboard.html")
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="text/html")
    else:
        return HTMLResponse(content="<h1>dashboard.html not found</h1>", status_code=404)
