import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import google.generativeai as genai
from groq import Groq
import requests
from gnews import GNews

# --- 1. 頁面設定 ---
st.set_page_config(page_title="股票大師：全方位戰情室", layout="wide", page_icon="⚡")
st.title("⚡ 股票大師：技術 x 財報 x Google 新聞")

# --- 安全性設定 ---
gemini_ok = False
try:
    gemini_key = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=gemini_key)
    gemini_model = genai.GenerativeModel('gemini-flash-latest') 
    gemini_ok = True
except:
    pass

groq_ok = False
try:
    groq_key = st.secrets["GROQ_API_KEY"]
    groq_client = Groq(api_key=groq_key)
    groq_ok = True
except:
    pass

# --- 2. 側邊欄參數 ---
st.sidebar.header("⚙️ 參數設定")
ticker_input = st.sidebar.text_input("輸入股票代碼", value="2330", help="台股請輸入如 2330, 美股如 NVDA")
days_input = st.sidebar.slider("K線觀察天數", 60, 730, 180)

if st.sidebar.button("🔄 強制刷新"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.subheader("📊 技術指標開關")
show_ma = st.sidebar.checkbox("顯示均線", value=True)
show_macd = st.sidebar.checkbox("顯示 MACD", value=True)
show_obv = st.sidebar.checkbox("顯示 OBV", value=True)

run_btn = st.sidebar.button("🚀 啟動全方位分析", type="primary")

# --- 3. 核心函數：計算指標 ---
def calculate_indicators(df):
    # 確保資料是數值型態
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
    
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

# --- 4. 數據抓取函數 (極致偽裝版) ---

# 4.1 新聞抓取 (Google News)
def fetch_google_news(query_name):
    try:
        google_news = GNews(language='zh-Hant', country='TW', period='7d', max_results=5)
        news_json = google_news.get_news(query_name)
        news_data = []
        for n in news_json:
            title = n.get('title', '無標題')
            publisher = n.get('publisher', {}).get('title', 'Google News')
            url = n.get('url', '#')
            news_data.append(f"- [{title}]({url}) ({publisher})")
        return "\n".join(news_data) if news_data else "無近期重大新聞"
    except Exception as e:
        return f"新聞抓取失敗: {str(e)}"

# 4.2 股價抓取 (使用 yfinance 但不依賴 session，避免過度複雜)
@st.cache_data(ttl=300)
def get_stock_price_history(symbol, days):
    end = datetime.now() + timedelta(days=1) 
    start = end - timedelta(days=days + 100)
    try:
        # yfinance 的 download 在新版中比較穩定
        df = yf.download(symbol, start=start, end=end, progress=False)
        
        # 修正 MultiIndex 問題 (這是很多錯誤的根源)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        if df.empty: return None, "Empty Data"
        return df, None
    except Exception as e:
        return None, str(e)

# 4.3 基本面抓取 (高容錯模式)
@st.cache_data(ttl=43200)
def get_stock_fundamentals(symbol):
    info = {}
    financials = pd.DataFrame()
    stock_name = symbol # 預設名稱為代碼
    
    try:
        # 使用 Session 偽裝
        session = requests.Session()
        session.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0 Safari/537.36'})
        
        stock = yf.Ticker(symbol, session=session)
        
        # 嘗試抓 info
        try:
            info = stock.info
            if 'longName' in info:
                stock_name = info['longName'].replace("Inc.", "").replace("Co., Ltd.", "").strip()
        except:
            pass # 抓不到 info 也不要當機
            
        # 嘗試抓 financials
        try:
            financials = stock.financials
        except:
            pass

        # 準備新聞搜尋關鍵字
        search_key = stock_name
        if symbol.replace(".TW", "").replace(".TWO", "").isdigit():
             base_code = symbol.split(".")[0]
             # 台股策略：代碼 + 名稱 (例如 "2330 台積電") 準確度最高
             search_key = f"{base_code} {stock_name}" if stock_name != symbol else f"{base_code} 股票"

        news_text = fetch_google_news(search_key)
            
        return info, financials, stock_name, news_text
    except Exception as e:
        # 最慘的情況，回傳預設值，讓程式繼續跑
        return {}, pd.DataFrame(), symbol, "無法取得新聞"

# --- 5. 智慧數值處理 (處理 N/A) ---
def get_smart_fundamentals(info, financials, current_price):
    # 預設值
    pe_str, roe_str, eps_val, peg_str, rev_str = "N/A", "N/A", None, "N/A", "N/A"
    
    # 1. 嘗試讀取 PE/EPS
    pe = info.get('trailingPE') or info.get('forwardPE')
    eps = info.get('trailingEps') or info.get('forwardEps')
    
    if pe: 
        pe_str = f"{pe:.2f}"
    elif eps and eps > 0:
        pe_str = f"{current_price/eps:.2f} (估)"
    elif eps and eps <= 0:
        pe_str = "虧損"
        
    if eps: eps_val = eps

    # 2. ROE
    if info.get('returnOnEquity'):
        roe_str = f"{info['returnOnEquity']*100:.2f}%"

    # 3. 營收成長
    if info.get('revenueGrowth'):
        rev_str = f"{info['revenueGrowth']*100:.2f}%"

    # 4. PEG 計算
    if info.get('pegRatio'):
        peg_str = f"{info['pegRatio']:.2f}"
    else:
        # 手動算
        try:
            if not financials.empty:
                # 模糊搜尋 EPS 列
                eps_row = financials[financials.index.str.contains('EPS', case=False, na=False)]
                if not eps_row.empty and len(eps_row.columns) >= 2:
                    e_now = eps_row.iloc[0, 0]
                    e_prev = eps_row.iloc[0, 1]
                    if e_prev != 0:
                        g = (e_now - e_prev) / abs(e_prev)
                        if g > 0 and pe:
                            peg_str = f"{pe/ (g*100):.2f} (估)"
        except:
            pass

    return pe_str, roe_str, eps_val, peg_str, rev_str

# --- 6. AI 分析 (容錯版 Prompt) ---
def get_prompt(symbol, stock_name, pe, roe, peg, rev, recent_data, news_text):
    now_str = datetime.now().strftime("%Y-%m-%d")
    
    return f"""
    角色：華爾街首席分析師。時間：{now_str}。
    標的：**{stock_name} ({symbol})**。

    【📰 新聞情報】
    {news_text}
    
    【📊 基本面數據】(若為 N/A 代表資料暫缺，請改用你的知識庫分析)
    - PE: {pe}, ROE: {roe}, PEG: {peg}, 營收成長: {rev}

    【📈 技術數據 (近5日)】
    {recent_data}

    請撰寫【全方位投資報告】：
    1. **新聞解讀**：市場情緒是樂觀/悲觀？有什麼大事？
    2. **基本面診斷**：若有數據請分析估值；**若無數據(N/A)，請憑知識簡述該公司產業地位與護城河**。
    3. **技術面判讀**：解讀趨勢、OBV、MACD。
    4. **操作建議**：多空劇本與進出場點位。
    """

def call_ai(model_name, prompt):
    try:
        if model_name == 'gemini' and gemini_ok:
            return gemini_model.generate_content(prompt).text
        elif model_name == 'groq' and groq_ok:
            return groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.3-70b-versatile"
            ).choices[0].message.content
    except Exception as e:
        return f"AI 思考中斷: {str(e)}"
    return "API Key 未設定"

# --- 7. 主程式 ---
if run_btn and ticker_input:
    raw_symbol = ticker_input.strip().upper()
    
    final_symbol = raw_symbol
    df_raw = None
    
    with st.spinner(f"正在搜尋 {raw_symbol}..."):
        # 智慧搜尋上市/上櫃
        if raw_symbol.isdigit():
            for suffix in [".TW", ".TWO"]:
                test_sym = raw_symbol + suffix
                df_test, _ = get_stock_price_history(test_sym, days_input)
                if df_test is not None and not df_test.empty:
                    final_symbol = test_sym
                    df_raw = df_test
                    break
        else:
            df_raw, _ = get_stock_price_history(final_symbol, days_input)

    if df_raw is None or df_raw.empty:
        st.error(f"❌ 查無 {raw_symbol} 資料。請稍後再試或檢查代碼。")
    else:
        # 取得基本面 (即使失敗也不會當機)
        info, financials, stock_name, news_text = get_stock_fundamentals(final_symbol)
        
        # 指標運算
        df = calculate_indicators(df_raw).iloc[-days_input:]
        last = df.iloc[-1]
        chg = last['Close'] - df['Close'].iloc[-2]
        pct = (chg / df['Close'].iloc[-2]) * 100
        
        # 數值格式化
        pe, roe, eps, peg, rev = get_smart_fundamentals(info, financials, last['Close'])
        
        st.header(f"🔥 {stock_name} ({final_symbol}) 戰情室")

        # 顯示看板 (容錯處理: 沒資料顯示 N/A)
        cols = st.columns(6)
        cols[0].metric("股價", f"{last['Close']:.2f}", f"{pct:.2f}%")
        cols[1].metric("營收成長", rev)
        cols[2].metric("PE", pe)
        cols[3].metric("ROE", roe)
        cols[4].metric("EPS", f"{eps:.2f}" if eps else "N/A")
        cols[5].metric("PEG", peg)

        tab1, tab2, tab3 = st.tabs(["📊 線圖", "⚡ AI 報告", "📰 新聞"])

        with tab1:
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='K線'), row=1, col=1)
            if show_ma: fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], line=dict(color='orange'), name='MA20'), row=1, col=1)
            fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='量'), row=2, col=1)
            fig.update_layout(height=600, xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            data_str = df.tail(5).to_string()
            # 確保變數正確傳入
            prompt = get_prompt(final_symbol, stock_name, pe, roe, peg, rev, data_str, news_text)
            
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("### 🔵 Gemini")
                if gemini_ok:
                    st.info(call_ai('gemini', prompt))
                else: st.error("無 Gemini Key")
            with c2:
                st.markdown("### 🟠 Llama 3")
                if groq_ok:
                    st.warning(call_ai('groq', prompt))
                else: st.error("無 Groq Key")

        with tab3:
            st.markdown(news_text)
