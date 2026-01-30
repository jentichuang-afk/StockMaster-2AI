import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import google.generativeai as genai
from groq import Groq

# --- 1. 頁面設定 ---
st.set_page_config(page_title="股票大師：雙 AI 戰情室", layout="wide", page_icon="⚡")
st.title("⚡ 股票大師：Google Gemini vs Meta Llama 3.3")

# --- 安全性設定 ---
# 1. Gemini (智慧切換版)
gemini_ok = False
try:
    gemini_key = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=gemini_key)
    # 使用目前 Google 推薦的最新穩定版指針
    gemini_model = genai.GenerativeModel('gemini-flash-latest') 
    gemini_ok = True
except Exception as e:
    print(f"Gemini Init Error: {e}")
    gemini_ok = False

# 2. Groq (Llama 3.3)
groq_ok = False
try:
    groq_key = st.secrets["GROQ_API_KEY"]
    groq_client = Groq(api_key=groq_key)
    groq_ok = True
except:
    groq_ok = False

# --- 2. 側邊欄參數 ---
st.sidebar.header("⚙️ 參數設定")
ticker_input = st.sidebar.text_input("輸入股票代碼", value="2330", help="台股請輸入如 2330, 美股如 NVDA")
days_input = st.sidebar.slider("K線觀察天數", 60, 730, 180)

# 刷新按鈕 (只清除股價快取，不清除基本面，保護連線)
if st.sidebar.button("🔄 強制刷新最新股價"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.subheader("📊 技術指標開關")
show_ma = st.sidebar.checkbox("顯示均線", value=True)
show_macd = st.sidebar.checkbox("顯示 MACD", value=True)
show_obv = st.sidebar.checkbox("顯示 OBV", value=True)

run_btn = st.sidebar.button("🚀 啟動雙強對決", type="primary")

# --- 3. 核心函數：計算指標 ---
def calculate_indicators(df):
    df['MA5'] = df['Close'].rolling(5).mean()
    df['MA20'] = df['Close'].rolling(20).mean()
    df['MA60'] = df['Close'].rolling(60).mean()
    
    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['Signal']

    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()

    low_min = df['Low'].rolling(9).min()
    high_max = df['High'].rolling(9).max()
    df['RSV'] = (df['Close'] - low_min) / (high_max - low_min) * 100
    k_list = [50]; d_list = [50]
    for r in df['RSV']:
        if pd.isna(r): k_list.append(50); d_list.append(50)
        else:
            k = (2/3) * k_list[-1] + (1/3) * r
            d = (2/3) * d_list[-1] + (1/3) * k
            k_list.append(k); d_list.append(d)   
    df['K'] = k_list[1:]; df['D'] = d_list[1:]
    
    std = df['Close'].rolling(20).std()
    df['BB_Upper'] = df['MA20'] + (std * 2)
    df['BB_Lower'] = df['MA20'] - (std * 2)
    return df

# --- 4. 數據抓取函數 (分離式快取策略) ---

# 策略 A: 股價資料 (快取 5 分鐘，確保盤中更新)
@st.cache_data(ttl=300)
def get_stock_price_history(symbol, days):
    end = datetime.now() + timedelta(days=1) 
    start = end - timedelta(days=days + 100)
    try:
        df = yf.download(symbol, start=start, end=end, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if
