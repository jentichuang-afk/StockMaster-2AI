import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import google.generativeai as genai
from groq import Groq
from duckduckgo_search import DDGS # 🆕 引入搜尋引擎

# --- 1. 頁面設定 ---
st.set_page_config(page_title="股票大師：AI 搜尋戰情室", layout="wide", page_icon="🔍")
st.title("⚡ 股票大師：AI 網路搜尋 x 深度解析")

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

if st.sidebar.button("🔄 強制刷新資料"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.subheader("📊 技術指標")
show_ma = st.sidebar.checkbox("顯示均線", value=True)
show_macd = st.sidebar.checkbox("顯示 MACD", value=True)
show_obv = st.sidebar.checkbox("顯示 OBV", value=True)

run_btn = st.sidebar.button("🚀 啟動 AI 網路肉搜", type="primary")

# --- 3. 指標計算 ---
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

# --- 4. 關鍵功能：DuckDuckGo 網路搜尋 ---
# 這將取代 Yahoo Info，直接去網路找數字
@st.cache_data(ttl=600)
def search_stock_info(symbol):
    results_text = ""
    news_text = ""
    
    # 判斷是否為台股，優化搜尋關鍵字
    if symbol.replace('.TW','').replace('.TWO','').isdigit():
        code = symbol.split('.')[0]
        # 搜尋 1: 基本面數據
        query_data = f"{code} 股票 本益比 EPS 營收成長率 股利"
        # 搜尋 2: 新聞
        query_news = f"{code} 個股新聞"
    else:
        query_data = f"{symbol} stock PE ratio EPS revenue growth"
        query_news = f"{symbol} stock news"

    try:
        with DDGS() as ddgs:
            # 抓取基本面搜尋結果
            r1 = list(ddgs.text(query_data, region='tw-tzh', max_results=5))
            for i in r1:
                results_text += f"- {i['title']}: {i['body']}\n"
            
            # 抓取新聞
            r2 = list(ddgs.news(query_news, region='tw-tzh', max_results=5))
            for i in r2:
                title = i.get('title','')
                source = i.get('source','')
                date = i.get('date','')
                url = i.get('url','')
                news_text += f"- [{title}]({url}) ({source} - {date})\n"
                
    except Exception as e:
        results_text = f"搜尋失敗: {e}"
        news_text = "無法取得新聞"

    return results_text, news_text

# --- 5. 抓股價 (只用 Yahoo 抓 K 線) ---
@st.cache_data(ttl=300)
def get_stock_data(symbol, days):
    end = datetime.now() + timedelta(days=1) 
    start = end - timedelta(days=days + 100)
    try:
        df = yf.download(symbol, start=start, end=end, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df.empty: return None
        return df
    except: return None

# --- 6. AI 分析 (負責從搜尋結果提取數字) ---
def get_prompt(symbol, price, search_results, news_text, tech_data):
    now = datetime.now().strftime("%Y-%m-%d")
    return f"""
    角色：華爾街王牌分析師。日期：{now}。
    
    分析對象：**{symbol}** (現價：{price:.2f})
    
    ⚠️ **重要任務：你是「數據探勘者」**。
    Yahoo API 已失效，但我幫你從 Google/DuckDuckGo 搜尋到了最新網頁資訊。
    請閱讀下方的【搜尋結果摘要】，**自行從文字中找出** 本益比(PE)、EPS、ROE 或 營收成長率。
    (例如看到 "台積電本益比約 20 倍"，請自行判讀 PE=20)。
    
    【🔍 網路搜尋結果摘要 (請從這裡找基本面數字)】
    {search_results}
    
    【📰 最新新聞】
    {news_text}
    
    【📈 技術指標數據】
    {tech_data}
    
    請撰寫分析報告 (若搜尋結果真的找不到某個數字，請誠實說「查無數據」，不要瞎掰)：
    
    ### 1. 🕵️‍♂️ 關鍵數據掃描 (AI 讀取結果)
    * **本益比 (PE)**: [請從搜尋結果填寫，若無則填 N/A]
    * **EPS / 獲利能力**: [請從搜尋結果摘要]
    * **營收表現**: [請從搜尋結果摘要]
    
    ### 2. 📰 市場情緒解讀
    * 新聞利多/利空判斷。
    
    ### 3. 📉 技術面與籌碼
    * 解讀 MACD, OBV, 均線位置。
    
    ### 4. 🎯 操作建議與評分 (0-100)
    * 給出明確的進出建議。
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
        return f"AI 錯誤: {e}"
    return "API Key 未設定"

# --- 7. 主程式 ---
if run_btn and ticker_input:
    raw_ticker = ticker_input.strip().upper()
    
    with st.spinner(f"正在啟動 AI 網路肉搜 {raw_ticker} ..."):
        # 1. 處理代碼 (上市櫃偵測)
        final_symbol = raw_ticker
        df = None
        if raw_ticker.isdigit():
            for s in ['.TW', '.TWO']:
                df = get_stock_data(raw_ticker + s, days_input)
                if df is not None:
                    final_symbol = raw_ticker + s
                    break
        else:
            df = get_stock_data(raw_ticker, days_input)
            
        if df is None:
            st.error("❌ 找不到股價資料 (Yahoo API 連線失敗)")
        else:
            # 2. 進行網路搜尋 (DuckDuckGo)
            search_data, news_data = search_stock_info(final_symbol)
            
            # 3. 計算技術指標
            df = calculate_indicators(df).iloc[-days_input:]
            last_price = df['Close'].iloc[-1]
            change = (last_price - df['Close'].iloc[-2]) / df['Close'].iloc[-2] * 100
            
            # 4. 顯示看板
            st.header(f"🔥 {final_symbol} AI 戰情室")
            c1, c2, c3 = st.columns(3)
            c1.metric("最新股價", f"{last_price:.2f}", f"{change:.2f}%")
            c2.metric("資料日期", str(df.index[-1].date()))
            c3.info("💡 基本面數據將由 AI 從網路搜尋結果中提取")

            # 5. 圖表
            tab1, tab2, tab3 = st.tabs(["📈 K線圖", "🤖 AI 報告", "🔍 搜尋結果源"])
            
            with tab1:
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
                fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='K'), row=1, col=1)
                fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Vol'), row=2, col=1)
                st.plotly_chart(fig, use_container_width=True)
                
            with tab3:
                st.markdown("### AI 讀到的網路資訊")
                st.text(search_data)
                st.markdown("### 最新新聞")
                st.markdown(news_data)
                
            with tab2:
                tech_str = df.tail(5).to_string()
                prompt = get_prompt(final_symbol, last_price, search_data, news_data, tech_str)
                
                c_gem, c_groq = st.columns(2)
                with c_gem:
                    st.subheader("🔵 Gemini")
                    if gemini_ok:
                        st.write(call_ai('gemini', prompt))
                    else: st.error("請設定 Gemini Key")
                with c_groq:
                    st.subheader("🟠 Llama 3")
                    if groq_ok:
                        st.write(call_ai('groq', prompt))
                    else: st.error("請設定 Groq Key")
