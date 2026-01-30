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
ticker_input = st.sidebar.text_input("輸入股票代碼", value="8155", help="台股請輸入如 2330, 美股如 NVDA")
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
@st.cache_data(ttl=300)
def get_stock_price_history(symbol, days):
    end = datetime.now() + timedelta(days=1) 
    start = end - timedelta(days=days + 100)
    try:
        df = yf.download(symbol, start=start, end=end, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df.empty: return None, "Empty"
        return df, None
    except Exception as e:
        return None, str(e)

@st.cache_data(ttl=43200)
def get_stock_fundamentals(symbol):
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        financials = stock.financials
        stock_name = info.get('longName', symbol) 
        return info, financials, stock_name
    except Exception as e:
        return {}, pd.DataFrame(), symbol

# --- 5. 智慧基本面修復函數 ---
def get_smart_fundamentals(info, financials, current_price):
    pe = info.get('trailingPE') or info.get('forwardPE')
    eps = info.get('trailingEps') or info.get('forwardEps')
    
    manual_pe_val = None
    
    if pe is not None:
        pe_str = f"{pe:.2f}"
        manual_pe_val = pe
    elif eps is not None:
        if eps > 0:
            manual_pe_val = current_price / eps
            pe_str = f"{manual_pe_val:.2f} (估)"
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
        try:
            eps_row = None
            if not financials.empty:
                for idx in financials.index:
                    if 'Basic EPS' in str(idx) or 'Diluted EPS' in str(idx):
                        eps_row = financials.loc[idx]
                        break
            
            if eps_row is not None and len(eps_row) >= 2:
                eps_this_year = eps_row.iloc[0]
                eps_last_year = eps_row.iloc[1]
                
                if eps_last_year != 0:
                    growth_rate = ((eps_this_year - eps_last_year) / abs(eps_last_year)) * 100
                    if growth_rate > 0 and manual_pe_val is not None:
                        calc_peg = manual_pe_val / growth_rate
                        peg_str = f"{calc_peg:.2f} (估)"
                    elif growth_rate <= 0:
                        peg_str = "N/A (EPS衰退)"
                    else:
                        peg_str = "N/A"
                else:
                    peg_str = "N/A"
            else:
                peg_str = "N/A"
        except Exception as e:
            peg_str = "N/A"
        
    return pe_str, roe_str, eps, peg_str

# --- 6. AI 分析函數 ---
def get_prompt(symbol, stock_name, pe, roe, peg, recent_data):
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    return f"""
    角色設定：你是一位擁有 20 年經驗的華爾街「首席投資長 (CIO)」。
    現在時間是 {now_str}。
    
    分析標的：**{stock_name}** (股票代號：{symbol})
    ⚠️ 注意：請務必針對「{stock_name}」這家公司的產業特性與新聞進行分析，切勿混淆成其他同業。

    【📊 財務體質數據】
    - PE (本益比): {pe}
    - ROE (股東權益報酬率): {roe}
    - PEG (成長估值): {peg} 

    【📈 近五日技術數據】
    {recent_data}

    請撰寫一份【深度投資報告】，章節如下：

    ### 1. 🕵️‍♂️ 盤勢與籌碼 (Context)
    - 解讀 **OBV** 與 **MACD** 趨勢。
    - 簡單描述 {stock_name} 的市場地位。

    ### 2. 🏢 估值診斷 (Valuation)
    - **重點分析 PEG**：(低於 1 為低估，高於 2 為高估)
    - 結合 PE 與 ROE 判斷。

    ### 3. ⚔️ 劇本推演 (Scenarios)
    - **多頭劇本**：關鍵突破價。
    - **回檔劇本**：關鍵支撐價。

    ### 4. 🎯 操作策略 (Action)
    - **建議**：(買進/觀望/賣出)
    - **進場舒適區**與**停損價**。

    ### 5. ⚖️ 評分 (0-100)
    - 請給出一個綜合評分。
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

# --- 7. 主程式 (智慧上市/上櫃偵測) ---
if run_btn and ticker_input:
    raw_symbol = ticker_input.strip().upper()
    
    # 變數初始化
    final_symbol = raw_symbol
    df_raw = None
    error_msg = ""
    
    with st.spinner(f"正在搜尋 {raw_symbol} 的正確掛牌市場 (上市.TW / 上櫃.TWO)..."):
        # 1. 如果是數字，啟動智慧偵測
        if raw_symbol.isdigit():
            # 先試試看 .TW (上市)
            try_tw = raw_symbol + ".TW"
            df_test, err = get_stock_price_history(try_tw, days_input)
            
            if df_test is not None and not df_test.empty:
                final_symbol = try_tw
                df_raw = df_test
            else:
                # 失敗了？那一定是 .TWO (上櫃)
                try_two = raw_symbol + ".TWO"
                df_test, err = get_stock_price_history(try_two, days_input)
                if df_test is not None and not df_test.empty:
                    final_symbol = try_two
                    df_raw = df_test
                else:
                    error_msg = "上市(.TW)與上櫃(.TWO)皆查無資料"
        else:
            # 美股或已帶後綴，直接查
            final_symbol = raw_symbol
            df_raw, error_msg = get_stock_price_history(final_symbol, days_input)

    # 2. 開始顯示資料
    if df_raw is None or df_raw.empty:
        st.error(f"❌ 找不到 {raw_symbol} 的資料。請確認代碼是否正確。")
    else:
        # 使用 final_symbol (正確的後綴) 去抓基本面
        info, financials, stock_name = get_stock_fundamentals(final_symbol)
        
        df = calculate_indicators(df_raw).iloc[-days_input:]
        last = df.iloc[-1]
        chg = last['Close'] - df['Close'].iloc[-2]
        pct = (chg / df['Close'].iloc[-2]) * 100
        
        pe_str, roe_str, eps_val, peg_str = get_smart_fundamentals(info, financials, last['Close'])
        last_date = last.name.strftime('%Y-%m-%d')
        
        # 標題顯示正確的公司名稱與代碼
        st.header(f"🔥 {stock_name} ({final_symbol}) 即時戰情室")

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
            st.subheader(f"⚡ {stock_name} 投資論戰 (Google vs Meta)")
            data_str = df.tail(5).to_string()
            prompt = get_prompt(final_symbol, stock_name, pe_str, roe_str, peg_str, data_str)
            
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
