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
# 1. Gemini
gemini_ok = False
try:
    gemini_key = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=gemini_key)
    gemini_model = genai.GenerativeModel('gemini-flash-latest') 
    gemini_ok = True
except Exception as e:
    print(f"Gemini Init Error: {e}")

# 2. Groq
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

# --- 4. 數據抓取函數 ---

# 策略 A: 股價資料 (快取 5 分鐘)
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

# 策略 B: 基本面資料 (快取 12 小時，大幅降低 N/A 機率)
@st.cache_data(ttl=43200)
def get_stock_fundamentals(symbol):
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        financials = stock.financials
        return info, financials
    except Exception as e:
        return {}, pd.DataFrame()

# --- 5. 智慧基本面修復函數 ---
def get_smart_fundamentals(info, current_price):
    # 嘗試多種欄位
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

    roe = info.get('returnOnEquity')
    if roe is not None:
        roe_str = f"{roe*100:.2f}%"
    else:
        roe_str = "N/A"
        
    peg = info.get('pegRatio')
    if peg is not None:
        peg_str = f"{peg:.2f}"
    else:
        peg_str = "N/A"
        
    return pe_str, roe_str, eps, peg_str

# --- 6. AI 分析函數 (容錯版 Prompt) ---
def get_prompt(symbol, pe, roe, peg, recent_data):
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    return f"""
    角色設定：你是一位擁有 20 年經驗的華爾街「首席投資長 (CIO)」。
    現在時間是 {now_str}。分析標的：{symbol}。

    【📊 財務體質數據】
    - PE (本益比): {pe}
    - ROE (股東權益報酬率): {roe}
    - PEG (成長估值): {peg}

    【📈 近五日技術數據】
    {recent_data}

    請撰寫一份【深度投資報告】，包含以下五章：

    ### 1. 🕵️‍♂️ 盤勢與籌碼 (Context)
    - 解讀 **OBV** 與 **量價關係**。
    - 判斷 **MACD** 與 **KD** 位階。

    ### 2. 🏢 估值診斷 (Valuation)
    - **⚠️ 特別指令：如果上方 PE/ROE/PEG 為 "N/A"，請明確指出「目前缺乏基本面數據，僅進行技術面分析」，並跳過估值判斷，直接進入下一章。**
    - 如果數據存在，請分析 PEG 是否 < 1 (低估) 或 > 2 (高估)。

    ### 3. ⚔️ 劇本推演 (Scenarios)
    - **多頭劇本**：突破哪裡確認續漲？
    - **回檔劇本**：跌破哪裡轉弱？

    ### 4. 🎯 操作策略 (Action)
    - **建議動作**：(買進/觀望/放空)
    - **進場舒適區**與**停損價**。

    ### 5. ⚖️ 評分 (0-100)
    - 如果缺乏基本面數據，請給出一個基於技術面的保守分數，並說明理由。
    """

def call_gemini(prompt):
    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        if "404" in str(e):
            try:
                fallback_model = genai.GenerativeModel('gemini-pro')
                response = fallback_model.generate_content(prompt)
                return f"⚠️ (自動切換至標準版 Gemini) \n\n{response.text}"
            except Exception as e2:
                return f"Gemini 失敗: {e2}"
        if "429" in str(e):
            return "⚠️ Gemini 休息中 (免費額度暫時用完)，請過 1 分鐘後再試。"
        return f"Gemini 思考失敗: {e}"

def call_groq(prompt):
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Groq 失敗: {e}"

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
        pct = (chg / df['Close'].iloc[-2]) * 100
        
        pe_str, roe_str, eps_val, peg_str = get_smart_fundamentals(info, last['Close'])
        last_date = last.name.strftime('%Y-%m-%d')
        
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("股價", f"{last['Close']:.2f}", f"{pct:.2f}%")
        c2.metric("資料日期", f"{last_date}")
        c3.metric("PE", pe_str)
        c4.metric("ROE", roe_str)
        c5.metric("EPS", f"{eps_val:.2f}" if eps_val else "N/A")
        c6.metric("PEG", peg_str)

        tab1, tab2, tab3 = st.tabs(["📊 技術圖表", "⚡ 雙 AI 觀點", "🏢 財報數據"])

        with tab1:
            rows = 2
            if show_macd: rows += 1
            if show_obv: rows += 1
            fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, row_heights=[0.5] + [0.15]*(rows-1))
            
            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='K線'), row=1, col=1)
            if show_ma:
                fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], line=dict(color='orange'), name='月線'), row=1, col=1)
            
            curr_row = 2
            fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='量'), row=curr_row, col=1); curr_row+=1
            
            if show_macd:
                colors = ['red' if h > 0 else 'green' for h in df['MACD_Hist']]
                fig.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'], marker_color=colors, name='MACD'), row=curr_row, col=1); curr_row+=1
            if show_obv:
                fig.add_trace(go.Scatter(x=df.index, y=df['OBV'], line=dict(color='purple'), name='OBV', fill='tozeroy'), row=curr_row, col=1)
                
            fig.update_layout(height=800, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.subheader(f"⚡ {symbol} 投資論戰 (Google vs Meta)")
            
            data_str = df.tail(5).to_string()
            # 將數據傳入 Prompt
            prompt = get_prompt(symbol, pe_str, roe_str, peg_str, data_str)
            
            col_gemini, col_groq = st.columns(2)
            
            with col_gemini:
                st.markdown("### 🔵 Gemini (Google)")
                if gemini_ok:
                    with st.spinner("Gemini 首席分析師思考中..."):
                        res_g = call_gemini(prompt)
                        st.info(res_g)
                else:
                    st.error("請設定 GEMINI_API_KEY")

            with col_groq:
                st.markdown("### 🟠 Llama 3.3 (Meta)")
                if groq_ok:
                    with st.spinner("Llama 3.3 首席分析師運算中..."):
                        res_l = call_groq(prompt)
                        st.warning(res_l) 
                else:
                    st.error("請設定 GROQ_API_KEY")

        with tab3:
            if not financials.empty:
                st.dataframe(financials)
            else:
                st.warning("無財報資料")
