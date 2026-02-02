import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import google.generativeai as genai
from groq import Groq

# --- 1. 頁面設定 ---
st.set_page_config(page_title="股票大師：純技術戰情室", layout="wide", page_icon="📈")
st.title("📈 股票大師：純技術面操盤 (Technical Only)")

# --- 安全性設定 ---
gemini_ok = False
try:
    gemini_key = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=gemini_key)
    gemini_model = genai.GenerativeModel('gemini-flash-latest') 
    gemini_ok = True
except: pass

groq_ok = False
try:
    groq_key = st.secrets["GROQ_API_KEY"]
    groq_client = Groq(api_key=groq_key)
    groq_ok = True
except: pass

# --- 2. 側邊欄 ---
st.sidebar.header("⚙️ 參數設定")
ticker_input = st.sidebar.text_input("輸入股票代碼", value="2330", help="台股請輸入如 2330, 8155")
days_input = st.sidebar.slider("K線觀察天數", 60, 730, 180)

if st.sidebar.button("🔄 刷新圖表"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.subheader("📊 指標開關")
show_ma = st.sidebar.checkbox("顯示均線 (MA)", value=True)
show_macd = st.sidebar.checkbox("顯示 MACD", value=True)
show_obv = st.sidebar.checkbox("顯示 OBV", value=True)

run_btn = st.sidebar.button("🚀 AI 技術分析", type="primary")

# --- 3. 核心數據處理 (只抓 K 線) ---
@st.cache_data(ttl=300)
def get_stock_data(symbol, days):
    try:
        # yfinance 抓 K 線非常穩定，不易被擋
        # 為了包含「今天」的盤中數據，end 日期往後推 1 天
        end_date = datetime.now() + timedelta(days=1)
        start_date = end_date - timedelta(days=days+100) # 多抓一些算指標
        
        df = yf.download(symbol, start=start_date, end=end_date, progress=False)
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        if df.empty: return None
        return df
    except: return None

# --- 4. 技術指標計算 (由 Python 算出精準數值) ---
def add_indicators(df):
    # 均線
    df['MA5'] = df['Close'].rolling(5).mean()
    df['MA20'] = df['Close'].rolling(20).mean()
    df['MA60'] = df['Close'].rolling(60).mean()
    
    # MACD
    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['Signal']

    # KD (Stochastic Oscillator)
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
    
    # OBV
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    
    return df

# --- 5. AI Prompt (專注於線圖解讀) ---
def get_prompt(symbol, last_close, technical_data):
    from datetime import datetime
    now = datetime.now().strftime("%Y-%m-%d")
    
    return f"""
    角色：你是一位精通「技術分析 (Technical Analysis)」的華爾街操盤手。
    
    標的：{symbol}
    現價：{last_close:.2f}
    日期：{now}
    
    請根據下方提供的【近 5 日技術指標數據】，進行純技術面判讀。
    (數據包含：收盤價, MA均線, KD值, MACD, OBV能量潮)
    
    {technical_data}
    
    請撰寫一份【技術操作策略】：
    1. 🕵️‍♂️ **趨勢判讀**：
       - 目前是多頭排列、空頭排列，還是盤整？
       - 股價相對於 MA20 (月線) 與 MA60 (季線) 的位置？
    
    2. ⚔️ **指標訊號**：
       - **KD 指標**：是黃金交叉(買點)、死亡交叉(賣點)，還是高/低檔鈍化？
       - **MACD**：柱狀體變化與多空力道。
       - **OBV**：量能是否支持股價？
       
    3. 🎯 **關鍵價位與策略**：
       - **支撐位 (Support)**：跌破哪裡要停損？
       - **壓力位 (Resistance)**：突破哪裡會噴出？
       - **操作建議**：(強力買進 / 拉回買進 / 觀望 / 反彈空 / 強力賣出)
    """

def call_ai(model, prompt):
    try:
        if model == 'gemini' and gemini_ok:
            return gemini_model.generate_content(prompt).text
        elif model == 'groq' and groq_ok:
            return groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.3-70b-versatile"
            ).choices[0].message.content
    except Exception as e:
        return f"AI 忙碌中: {str(e)}"
    return "API Key 未設定"

from datetime import datetime, timedelta # 補上這行避免報錯

# --- 6. 主程式 ---
if run_btn and ticker_input:
    raw_ticker = ticker_input.strip().upper()
    
    # 1. 智慧搜尋 (上市/上櫃)
    final_symbol = raw_ticker
    df = None
    
    with st.spinner(f"正在繪製 {raw_ticker} 技術線圖..."):
        if raw_ticker.isdigit():
            for s in ['.TW', '.TWO']:
                df = get_stock_data(raw_ticker + s, days_input)
                if df is not None:
                    final_symbol = raw_ticker + s
                    break
        else:
            df = get_stock_data(raw_ticker, days_input)
    
    if df is None:
        st.error(f"❌ 查無代碼 {raw_ticker}")
    else:
        # 2. 計算指標
        df = add_indicators(df)
        df_display = df.iloc[-days_input:] # 只顯示設定的天數
        
        last = df.iloc[-1]
        chg = last['Close'] - df['Close'].iloc[-2]
        pct = (chg / df['Close'].iloc[-2]) * 100
        color = "green" if pct > 0 else "red"
        
        # 3. 顯示看板 (純價格資訊)
        st.markdown(f"## 🔥 {final_symbol} 技術戰情室")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("收盤價", f"{last['Close']:.2f}", f"{pct:.2f}%")
        c2.metric("MA5 (短線)", f"{last['MA5']:.2f}")
        c3.metric("MA20 (月線)", f"{last['MA20']:.2f}")
        c4.metric("K值 / D值", f"{last['K']:.1f} / {last['D']:.1f}")

        # 4. 繪製互動圖表 (K線 + MA + MACD)
        tab1, tab2 = st.tabs(["📈 技術分析圖表", "🤖 AI 操盤建議"])
        
        with tab1:
            # 設定圖表列數 (依據開關)
            rows = 2
            if show_macd: rows += 1
            if show_obv: rows += 1
            row_heights = [0.6] + [0.4/(rows-1)] * (rows-1)
            
            fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, row_heights=row_heights, vertical_spacing=0.03)
            
            # 主圖 (K線 + MA)
            fig.add_trace(go.Candlestick(x=df_display.index, open=df_display['Open'], high=df_display['High'], 
                                         low=df_display['Low'], close=df_display['Close'], name='K線'), row=1, col=1)
            if show_ma:
                fig.add_trace(go.Scatter(x=df_display.index, y=df_display['MA5'], line=dict(color='yellow', width=1), name='MA5'), row=1, col=1)
                fig.add_trace(go.Scatter(x=df_display.index, y=df_display['MA20'], line=dict(color='orange', width=1.5), name='MA20'), row=1, col=1)
                fig.add_trace(go.Scatter(x=df_display.index, y=df_display['MA60'], line=dict(color='purple', width=1.5), name='MA60'), row=1, col=1)
            
            # 副圖 1 (成交量)
            curr_row = 2
            colors = ['red' if c >= o else 'green' for c, o in zip(df_display['Close'], df_display['Open'])]
            fig.add_trace(go.Bar(x=df_display.index, y=df_display['Volume'], marker_color=colors, name='成交量'), row=curr_row, col=1)
            curr_row += 1
            
            # 副圖 2 (MACD)
            if show_macd:
                hist_color = ['red' if v >= 0 else 'green' for v in df_display['MACD_Hist']]
                fig.add_trace(go.Bar(x=df_display.index, y=df_display['MACD_Hist'], marker_color=hist_color, name='MACD柱'), row=curr_row, col=1)
                fig.add_trace(go.Scatter(x=df_display.index, y=df_display['MACD'], line=dict(color='orange', width=1), name='DIF'), row=curr_row, col=1)
                fig.add_trace(go.Scatter(x=df_display.index, y=df_display['Signal'], line=dict(color='blue', width=1), name='DEM'), row=curr_row, col=1)
                curr_row += 1
                
            # 副圖 3 (OBV)
            if show_obv:
                fig.add_trace(go.Scatter(x=df_display.index, y=df_display['OBV'], line=dict(color='cyan', width=1), name='OBV', fill='tozeroy'), row=curr_row, col=1)
            
            fig.update_layout(height=800, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

        # 5. AI 分析
        with tab2:
            # 準備數據給 AI (只給最近 5 天的精準數據)
            tech_data_str = df.tail(5)[['Close', 'MA5', 'MA20', 'K', 'D', 'MACD', 'MACD_Hist', 'OBV']].to_string()
            prompt = get_prompt(final_symbol, last['Close'], tech_data_str)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### 🔵 Gemini (技術派)")
                if gemini_ok:
                    with st.spinner("Gemini 正在計算支撐壓力..."):
                        st.info(call_ai('gemini', prompt))
                else: st.error("請設定 Gemini Key")
            
            with col2:
                st.markdown("### 🟠 Llama 3 (動能派)")
                if groq_ok:
                    with st.spinner("Llama 正在分析多空訊號..."):
                        st.warning(call_ai('groq', prompt))
                else: st.error("請設定 Groq Key")
