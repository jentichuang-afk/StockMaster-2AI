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
gemini_ok = False
try:
    gemini_key = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=gemini_key)
    gemini_model = genai.GenerativeModel('gemini-flash-latest') 
    gemini_ok = True
except:
    gemini_ok = False

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
    
    # MACD
    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['Signal']

    # OBV
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()

    # KD
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
    
    # 布林
    std = df['Close'].rolling(20).std()
    df['BB_Upper'] = df['MA20'] + (std * 2)
    df['BB_Lower'] = df['MA20'] - (std * 2)

    return df

# --- 4. 數據抓取函數 ---
@st.cache_data(ttl=3600)
def get_stock_data(symbol, days):
    end = datetime.now()
    start = end - timedelta(days=days + 100)
    try:
        df = yf.download(symbol, start=start, end=end, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df.empty: return None, "Yahoo 回傳空資料"
        return df, None
    except Exception as e:
        return None, str(e)

# --- 5. 🆕 智慧基本面修復函數 ---
def get_smart_fundamentals(info, current_price):
    # 1. 處理 PE (本益比)
    pe = info.get('trailingPE')
    eps = info.get('trailingEps')
    
    if pe is not None:
        pe_str = f"{pe:.2f}"
    elif eps is not None:
        # 如果沒有 PE 但有 EPS，嘗試手動計算
        if eps > 0:
            manual_pe = current_price / eps
            pe_str = f"{manual_pe:.2f} (估)"
        else:
            pe_str = "虧損 (EPS<0)"
    else:
        pe_str = "N/A"

    # 2. 處理 ROE (股東權益報酬率)
    roe = info.get('returnOnEquity')
    if roe is not None:
        roe_str = f"{roe*100:.2f}%"
    else:
        roe_str = "N/A"
        
    return pe_str, roe_str, eps

# --- 6. AI 分析函數 ---
def get_prompt(symbol, pe, roe, peg, recent_data):
    return f"""
    你是一位華爾街頂級避險基金經理人。請分析股票 {symbol}。
    
    【基本面】PE: {pe}, ROE: {roe}, PEG: {peg}
    【技術面數據(近5日)】
    {recent_data}
    
    請簡潔有力地回答以下重點：
    1. 🎯 **多空判斷**：直接說「看多」、「看空」還是「觀望」。
    2. 🔑 **關鍵理由**：用 3 點說明為什麼（結合技術面與籌碼）。
    3. 🛑 **操作價位**：給出建議的「進場價」與「停損價」。
    4. 💯 **評分**：給出 0-100 分。
    
    請用繁體中文，語氣專業且自信。
    """

def call_gemini(prompt):
    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        if "404" in str(e):
            try:
                fallback = genai.GenerativeModel('gemini-pro')
                return f"⚠️ (自動切換標準版) \n{fallback.generate_content(prompt).text}"
            except: pass
        if "429" in str(e): return "⚠️ Gemini 休息中 (額度滿)，請稍後再試。"
        return f"Gemini 失敗: {e}"

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
    
    with st.spinner(f"正在連線 Yahoo Finance 抓取 {symbol} ..."):
        df_raw, error_msg = get_stock_data(symbol, days_input)
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            financials = stock.financials
        except:
            info = {}
            financials = pd.DataFrame()

    if df_raw is None or df_raw.empty:
        st.error(f"❌ 找不到資料。錯誤訊息: {error_msg}")
    else:
        df = calculate_indicators(df_raw).iloc[-days_input:]
        last = df.iloc[-1]
        chg = last['Close'] - df['Close'].iloc[-2]
        pct = (chg / df['Close'].iloc[-2]) * 100
        
        # 使用新的修復函數
        pe_str, roe_str, eps_val = get_smart_fundamentals(info, last['Close'])
        
        c1, c2, c3, c4, c5 = st.columns(5) # 改成 5 欄，多顯示 EPS
        c1.metric("股價", f"{last['Close']:.2f}", f"{pct:.2f}%")
        c2.metric("成交量", f"{int(last['Volume']/1000)}張")
        c3.metric("PE (本益比)", pe_str)
        c4.metric("ROE (股東權益)", roe_str)
        c5.metric("EPS (每股盈餘)", f"{eps_val:.2f}" if eps_val else "N/A")

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
            # 傳遞修復後的數據給 AI
            prompt = get_prompt(symbol, pe_str, roe_str, info.get('pegRatio','N/A'), data_str)
            
            col_gemini, col_groq = st.columns(2)
            
            with col_gemini:
                st.markdown("### 🔵 Gemini (Google)")
                if gemini_ok:
                    with st.spinner("Gemini 深度思考中..."):
                        res_g = call_gemini(prompt)
                        st.info(res_g)
                else:
                    st.error("請設定 GEMINI_API_KEY")

            with col_groq:
                st.markdown("### 🟠 Llama 3.3 (Meta)")
                if groq_ok:
                    with st.spinner("Llama 3.3 急速運算中..."):
                        res_l = call_groq(prompt)
                        st.warning(res_l) 
                else:
                    st.error("請設定 GROQ_API_KEY")

        with tab3:
            if not financials.empty:
                st.dataframe(financials)
            else:
                st.warning("無財報資料")
