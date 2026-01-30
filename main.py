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
        if df.empty: return None, "Yahoo 回傳空資料"
        return df, None
    except Exception as e:
        return None, str(e)

# 策略 B: 基本面資料 (快取 1 小時，避免頻繁請求被鎖)
@st.cache_data(ttl=3600)
def get_stock_fundamentals(symbol):
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        financials = stock.financials
        return info, financials
    except Exception as e:
        return {}, pd.DataFrame()

# --- 5. 智慧基本面修復函數 (新增 PEG 處理) ---
def get_smart_fundamentals(info, current_price):
    # 1. PE 本益比
    pe = info.get('trailingPE') or info.get('forwardPE')
    eps = info.get('trailingEps') or info.get('forwardEps')
    
    if pe is not None:
        pe_str = f"{pe:.2f}"
    elif eps is not None:
        if eps > 0:
            manual_pe = current_price / eps
            pe_str = f"{manual_pe:.2f} (估)"
        else:
            pe_str = "虧損 (EPS<0)"
    else:
        pe_str = "N/A"

    # 2. ROE
    roe = info.get('returnOnEquity')
    if roe is not None:
        roe_str = f"{roe*100:.2f}%"
    else:
        roe_str = "N/A"
        
    # 3. 🆕 PEG (本益成長比)
    peg = info.get('pegRatio')
    if peg is not None:
        peg_str = f"{peg:.2f}"
    else:
        peg_str = "N/A"
        
    return pe_str, roe_str, eps, peg_str

# --- 6. AI 分析函數 (升級版：加入 PEG 分析邏輯) ---
def get_prompt(symbol, pe, roe, peg, recent_data):
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    return f"""
    角色設定：你是一位擁有 20 年經驗的華爾街「首席投資長 (CIO)」。你的專長是結合「基本面價值」與「技術面動能」進行深度分析。
    現在時間是 {now_str}。分析標的：{symbol}。

    【📊 財務體質數據】
    - 本益比 (PE): {pe}
    - 股東權益報酬率 (ROE): {roe} (公司賺錢效率)
    - 🆕 PEG 指標 (本益成長比): {peg} 
      (★重要判斷標準：PEG < 1 代表股價被低估，值得買進；PEG > 2 代表股價成長溢價過高，需小心)

    【📈 近五日技術與籌碼數據 (包含 K值, D值, MACD, OBV, 布林通道)】
    {recent_data}

    請撰寫一份【機構級深度投資報告】，必須包含以下五個章節，並使用繁體中文專業財經術語：

    ### 1. 🕵️‍♂️ 盤勢與籌碼解讀 (The Context)
    - 解讀 **OBV (能量潮)**：是「量價齊揚」(吸籌) 還是「量價背離」(出貨)？
    - 解讀 **MACD** 與 **KD** 的多空位階。

    ### 2. 🏢 估值與 PEG 診斷 (Valuation)
    - **重點分析 PEG**：目前的 PEG 數據顯示這家公司是「便宜的成長股」還是「昂貴的泡沫」？
    - 結合 PE 與 ROE，判斷其長期持有價值。

    ### 3. ⚔️ 多空劇本推演 (Scenarios)
    - **劇本 A (多頭續攻)**：關鍵突破價位在哪？
    - **劇本 B (回檔修正)**：關鍵支撐防線在哪？

    ### 4. 🎯 精準操作策略 (Action Plan)
    - **建議動作**：(積極買進 / 拉回佈局 / 區間操作 / 獲利了結 / 放空)
    - **進場舒適區**：具體的價格區間。
    - **停損防守價**：嚴格的風控價位。

    ### 5. ⚖️ 綜合風險評分 (0-100)
    - 給出分數，並簡述理由（PEG 的高低應顯著影響評分）。

    **回答要求**：
    1. 語氣必須**冷靜、客觀、犀利**。
    2. 如果數據中有 **N/A** 或異常值，請在分析中指出。
    3. 特別注意 PEG 與股價的關係。
    """

def call_gemini(prompt):
    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        error_str = str(e)
        if "404" in error_str:
            try:
                fallback_model = genai.GenerativeModel('gemini-pro')
                response = fallback_model.generate_content(prompt)
                return f"⚠️ (自動切換至標準版 Gemini) \n\n{response.text}"
            except Exception as e2:
                return f"Gemini 所有模型皆失敗: {e2}"
        if "429" in error_str:
            return "⚠️ Gemini 正在休息 (免費額度暫時用完)，請過 1 分鐘後再試。"
        return f"Gemini 思考失敗: {e}"

def call_groq(prompt):
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Groq (Llama 3.3) 思考失敗: {e}"

# --- 7. 主程式 ---
if run_btn and ticker_input:
    symbol = ticker_input.strip().upper()
    if symbol.isdigit(): symbol += ".TW"
    
    # 步驟 1: 抓股價
    with st.spinner(f"正在連線 Yahoo Finance 抓取股價 {symbol} ..."):
        df_raw, error_msg = get_stock_price_history(symbol, days_input)

    # 步驟 2: 抓基本面
    info, financials = get_stock_fundamentals(symbol)

    if df_raw is None or df_raw.empty:
        st.error(f"❌ 找不到資料。錯誤訊息: {error_msg}")
    else:
        df = calculate_indicators(df_raw).iloc[-days_input:]
        last = df.iloc[-1]
        chg = last['Close'] - df['Close'].iloc[-2]
        pct = (chg / df['
