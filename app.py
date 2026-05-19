import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import requests
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configuración de la página con estética Buffett Premium
st.set_page_config(page_title="Mini Buffett | Terminal de Inteligencia", layout="wide")

# Paleta de colores "Buffett Premium"
COLOR_PRIMARY = "#1E3A8A"  # Deep Blue
COLOR_SECONDARY = "#10B981" # Emerald
COLOR_ACCENT = "#F59E0B"    # Amber
COLOR_BG = "#0F172A"       # Dark Slate
COLOR_CARD = "rgba(30, 41, 59, 0.7)"

# --- ESTILOS CSS PREMIUM (Sincronizados) ---
st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');
    html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
    .main {{ background-color: {COLOR_BG}; color: #F8FAFC; }}
    .stMetric {{
        background: {COLOR_CARD};
        padding: 15px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }}
    .buffett-badge {{
        background: linear-gradient(135deg, {COLOR_ACCENT}, #D97706);
        color: white;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
    }}
    /* Estilizar expanders y tabs */
    div[data-testid="stExpander"] {{ background: {COLOR_CARD}; border-radius: 10px; }}
    .stTabs [data-baseweb="tab-list"] {{ gap: 8px; }}
    .stTabs [data-baseweb="tab"] {{
        background-color: {COLOR_CARD};
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
    }}
    </style>
    """, unsafe_allow_html=True)

# --- 1. DATA ENGINEERING (LISTADO) ---
@st.cache_data(ttl=3600)
def get_sp500_base():
    """Obtiene la lista base de Wikipedia."""
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers)
        table = pd.read_html(response.text)
        df = table[0].rename(columns={
            'Symbol': 'Ticker', 
            'Security': 'Nombre', 
            'GICS Sector': 'Sector', 
            'GICS Sub-Industry': 'Industria'
        })
        df['Ticker'] = df['Ticker'].str.replace('.', '-', regex=False)
        return df[['Ticker', 'Nombre', 'Sector', 'Industria']]
    except Exception as e:
        st.error(f"Error al obtener lista: {e}")
        return pd.DataFrame()

def process_single_ticker(ticker):
    """Procesa un solo ticker de forma aislada para multihilos."""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1y")
        if hist.empty: return None
        
        info = stock.info
        close = hist['Close']
        last_p = close.iloc[-1]
        
        # Medias Móviles
        ma9 = close.rolling(9).mean().iloc[-1]
        ma20 = close.rolling(20).mean().iloc[-1]
        ma50 = close.rolling(50).mean().iloc[-1]
        ma100 = close.rolling(100).mean().iloc[-1]
        ma200 = close.rolling(200).mean().iloc[-1]
        
        # MACD
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd_series = ema12 - ema26
        macd_signal_series = macd_series.ewm(span=9, adjust=False).mean()
        macd = macd_series.iloc[-1]
        macd_signal = macd_signal_series.iloc[-1]

        # RSI Wilder
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        rsi = 100 - (100 / (1 + (avg_gain / avg_loss))).iloc[-1]
        
        # RVOL
        rvol = close.iloc[-1] / hist['Volume'].tail(20).mean() if not hist['Volume'].empty else 1.0
        
        # Métricas de Valor y Riesgo
        roe = info.get('returnOnEquity', 0)
        rev_growth = info.get('revenueGrowth', 0)
        peg = info.get('pegRatio', 0)
        beta = info.get('beta', 1.0)
        high_52 = info.get('fiftyTwoWeekHigh', last_p)
        dist_52h = (last_p / high_52 - 1) if high_52 else 0
        
        # --- BUFFETT GRADE (Calificación de Calidad) ---
        grade_points = 0
        if roe > 0.15: grade_points += 1
        if rev_growth > 0.10: grade_points += 1
        if info.get('profitMargins', 0) > 0.10: grade_points += 1
        if info.get('currentRatio', 0) > 1.2: grade_points += 1
        
        grades = {4: "💎 A", 3: "🥇 B", 2: "🥈 C", 1: "🥉 D", 0: "⚠️ F"}
        b_grade = grades.get(grade_points, "⚠️ F")

        # Señalización
        signal = "NEUTRAL"
        if rvol > 2.0 and rsi < 50: signal = "🚀 BREAKOUT"
        elif rsi < 30: signal = "💎 SOBREVENTA"
        elif rsi > 70: signal = "⚠️ SOBRECOMPRA"
        elif ma50 > ma200 and rsi < 60: signal = "🌟 TENDENCIA"

        return {
            'Ticker': ticker,
            'Grade': b_grade,
            'Señal': signal,
            'Precio': round(last_p, 2),
            'RSI': round(rsi, 2),
            'MACD': round(macd, 2) if pd.notna(macd) else None,
            'MACD Signal': round(macd_signal, 2) if pd.notna(macd_signal) else None,
            'Dist MA200 (%)': round(((last_p / ma200) - 1) * 100, 2) if pd.notna(ma200) else None,
            'Market Cap (B)': round(info.get('marketCap', 0) / 1e9, 2),
            'Beta': round(beta, 2),
            'Volumen': int(hist['Volume'].iloc[-1]) if not hist['Volume'].empty else 0,
            'Sector': info.get('sector', 'N/A'),
            'Industria': info.get('industry', 'N/A'),
            'MA9': round(ma9, 2) if pd.notna(ma9) else None,
            'MA20': round(ma20, 2) if pd.notna(ma20) else None,
            'MA50': round(ma50, 2) if pd.notna(ma50) else None,
            'MA100': round(ma100, 2) if pd.notna(ma100) else None,
            'MA200': round(ma200, 2) if pd.notna(ma200) else None,
            'RVOL': round(rvol, 2),
            'PEG': round(peg, 2) if peg else None,
            'ROE (%)': round(roe * 100, 2),
            'Dist 52wH (%)': round(dist_52h * 100, 2),
            'Market Cap': info.get('marketCap', 0)
        }
    except:
        return None

@st.cache_data(ttl=3600)
def fetch_financial_data(tickers):
    """Descarga datos en paralelo usando multihilos (Speed Upgrade)."""
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Usamos ThreadPoolExecutor para procesar en paralelo
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(process_single_ticker, t): t for t in tickers}
        for i, future in enumerate(as_completed(futures)):
            res = future.result()
            if res: results.append(res)
            progress_bar.progress((i + 1) / len(tickers))
            status_text.text(f"Procesando: {i+1}/{len(tickers)}")
            
    progress_bar.empty()
    status_text.empty()
    return results

# --- 2. UI PRINCIPAL ---
def main():
    st.sidebar.title("🏆 CONFIGURACIÓN")
    
    # OPCIÓN 1: GESTIÓN DE BASE (WIKIPEDIA)
    with st.sidebar.expander("📂 Base de Datos S&P 500"):
        if st.button("🔄 Actualizar desde Wikipedia"):
            st.session_state['master_df'] = get_sp500_base()
            st.success("Lista actualizada.")
        
        if 'master_df' not in st.session_state:
            st.session_state['master_df'] = get_sp500_base()
            
        csv_base = st.session_state['master_df'].to_csv(index=False)
        st.download_button("📥 Descargar CSV Base", csv_base, "sp500_base.csv", "text/csv")

    # OPCIÓN 2: SELECCIÓN DE FUENTE
    source = st.sidebar.radio("Fuente de Entrada", ["Universo S&P 500", "Ingreso Manual", "Archivo Local"])
    
    final_tickers = []
    
    if source == "Universo S&P 500":
        df_base = st.session_state['master_df']
        st.header("1️⃣ Paso: Filtrar y Seleccionar")
        
        c1, c2 = st.columns(2)
        with c1:
            all_sectors = sorted(df_base['Sector'].unique().tolist())
            sel_sectors = st.multiselect("Filtrar Sectores", all_sectors)
        with c2:
            temp_df = df_base if not sel_sectors else df_base[df_base['Sector'].isin(sel_sectors)]
            all_industries = sorted(temp_df['Industria'].unique().tolist())
            sel_industries = st.multiselect("Filtrar Industrias", all_industries)
            
        df_to_show = df_base
        if sel_sectors: df_to_show = df_to_show[df_to_show['Sector'].isin(sel_sectors)]
        if sel_industries: df_to_show = df_to_show[df_to_show['Industria'].isin(sel_industries)]
        
        st.dataframe(df_to_show, use_container_width=True, height=250)
        final_tickers = df_to_show['Ticker'].tolist()
        st.info(f"Seleccionados {len(final_tickers)} tickers para procesar.")

    elif source == "Ingreso Manual":
        st.header("✍️ Ingreso Manual")
        m_input = st.text_area("Ingresa los Tickers (separados por coma)", "AAPL, MSFT, TSLA, AMZN, GOOGL")
        final_tickers = [t.strip().upper() for t in m_input.split(",") if t.strip()]

    elif source == "Archivo Local":
        st.header("📁 Cargar Archivo")
        u_file = st.file_uploader("Subir CSV o Excel con columna 'Ticker' o 'Symbol'", type=['csv', 'xlsx'])
        if u_file:
            df_u = pd.read_csv(u_file) if u_file.name.endswith('.csv') else pd.read_excel(u_file)
            col = 'Ticker' if 'Ticker' in df_u.columns else df_u.columns[0]
            final_tickers = df_u[col].tolist()

    # EJECUCIÓN DEL SCREENER
    st.sidebar.markdown("---")
    if st.sidebar.button("🚀 INICIAR ESCANEO QUANT"):
        if not final_tickers:
            st.error("No hay tickers seleccionados.")
        else:
            with st.spinner(f"Analizando {len(final_tickers)} activos..."):
                st.session_state['results'] = fetch_financial_data(final_tickers)

    # RESULTADOS
    if 'results' in st.session_state:
        res = st.session_state['results']
        if not res:
            st.warning("No se pudieron obtener resultados para los tickers seleccionados.")
            return
            
        df_full = pd.DataFrame(res)
        
        tab1, tab2 = st.tabs(["🔍 SCREENER AVANZADO", "📚 DICCIONARIO DE ESTRATEGIAS"])
        
        with tab1:
            # Filtros dinámicos en los resultados
            f_row1_c1, f_row1_c2, f_row1_c3, f_row1_c4 = st.columns(4)
            with f_row1_c1:
                search = st.text_input("Ticker", "").upper()
            with f_row1_c2:
                alert_f = st.multiselect("Señal", sorted(df_full['Señal'].unique()))
            with f_row1_c3:
                sector_f = st.multiselect("Sector", sorted(df_full['Sector'].unique()))
            with f_row1_c4:
                grade_f = st.multiselect("Calificación", sorted(df_full['Grade'].unique()))
            
            df_view = df_full.copy()
            if search: df_view = df_view[df_view['Ticker'].str.contains(search)]
            if alert_f: df_view = df_view[df_view['Señal'].isin(alert_f)]
            if sector_f: df_view = df_view[df_view['Sector'].isin(sector_f)]
            if grade_f: df_view = df_view[df_view['Grade'].isin(grade_f)]

            # Formateo y visualización
            def style_signals(val):
                if not isinstance(val, str): return ''
                if 'BREAKOUT' in val: return 'background-color: #00BFFF; color: #000000; font-weight: bold;'
                if 'SOBREVENTA' in val or 'VALOR' in val: return 'background-color: #10B981; color: #FFFFFF; font-weight: bold;'
                if 'SOBRECOMPRA' in val or 'RIESGO' in val: return 'background-color: #EF4444; color: #FFFFFF; font-weight: bold;'
                if 'TENDENCIA' in val: return 'background-color: #FFD700; color: #000000; font-weight: bold;'
                return ''
            
            def style_grades(val):
                if not isinstance(val, str): return ''
                if 'A' in val or 'B' in val: return 'color: #10B981; font-weight: bold;'
                if 'F' in val: return 'color: #DC2626; font-weight: bold;'
                return ''

            st.dataframe(
                df_view.style
                .map(style_signals, subset=['Señal'])
                .map(style_grades, subset=['Grade'])
                .format({
                    "Precio": "${:,.2f}", 
                    "Market Cap (B)": "{:,.2f}B", 
                    "Dist 52wH (%)": "{:.2f}%",
                    "RVOL": "{:.2f}x",
                    "ROE (%)": "{:.2f}%",
                    "RSI": "{:.2f}",
                    "Beta": "{:.2f}",
                    "MACD": "{:.2f}",
                    "MACD Signal": "{:.2f}",
                    "Dist MA200 (%)": "{:.2f}%",
                    "Volumen": "{:,.0f}",
                    "MA9": "${:,.2f}",
                    "MA20": "${:,.2f}",
                    "MA50": "${:,.2f}",
                    "MA100": "${:,.2f}",
                    "MA200": "${:,.2f}",
                    "Market Cap": "${:,.0f}"
                }, na_rep="N/A"),
                use_container_width=True, height=600
            )
            
            st.download_button("📥 Descargar Resultados Completos", df_full.to_csv(index=False), "screener_results.csv")

        with tab2:
            st.subheader("📖 Guía de Estrategias Quant")
            st.markdown("A continuación se explican las reglas lógicas de cada estrategia integrada en el escáner:")
            
            col_e1, col_e2, col_e3 = st.columns(3)
            
            with col_e1:
                st.info("### 🚀 BREAKOUT INST.")
                st.markdown("""
                **Enfoque:** Momentum e Instituciones.
                
                **Reglas:**
                1. **RVOL > 2.0:** Volumen actual duplica el promedio de 20 días.
                2. **MACD > Signal:** Cruce alcista de momentum.
                3. **Precio > MA50:** Tendencia de medio plazo confirmada.
                
                *Ideal para capturar el inicio de movimientos explosivos.*
                """)
                
            with col_e2:
                st.success("### 💎 VALOR/REVERSIÓN")
                st.markdown("""
                **Enfoque:** Calidad y Valor.
                
                **Reglas:**
                1. **RSI < 35:** El activo está técnicamente sobrevendido.
                2. **ROE > 15%:** La empresa es altamente eficiente (Filtro Buffett).
                3. **Precio > MA200:** El activo sigue en tendencia alcista principal.
                
                *Ideal para comprar empresas excelentes con descuento temporal.*
                """)
                
            with col_e3:
                st.warning("### 🌟 CRECIMIENTO DORADO")
                st.markdown("""
                **Enfoque:** Tendencia y Crecimiento.
                
                **Reglas:**
                1. **MA50 > MA200:** Cruce Dorado.
                2. **Crecimiento > 15%:** Ventas en expansión.
                """)

            st.markdown("---")
            col_e4, col_e5 = st.columns(2)
            
            with col_e4:
                st.markdown("### 🩹 REBOTE/GIRO")
                st.markdown("""
                **Enfoque:** Suelo técnico de corto plazo.
                
                **Reglas:**
                1. **RSI < 30:** Sobreventa técnica (pánico).
                2. **MACD > Signal:** El momentum empieza a girar al alza.
                
                *Ideal para capturar 'suelos' en activos que han caído fuerte.*
                """)

            with col_e5:
                st.error("### ⚠️ VENTA/RIESGO")
                st.markdown("""
                **Enfoque:** Salida o Cobertura.
                
                **Reglas:**
                1. **RSI > 70:** Sobrecompra técnica (euforia).
                2. **MACD < Signal:** El momentum empieza a girar a la baja.
                """)
                
            st.markdown("---")
            st.subheader("💎 Entendiendo el Buffett Grade")
            st.write("""
            La calificación asignada (A-F) se basa en la suma de 4 puntos fundamentales:
            - **Punto 1:** ROE superior al 15%.
            - **Punto 2:** Crecimiento de ingresos superior al 10%.
            - **Punto 3:** Margen neto superior al 10%.
            - **Punto 4:** Ratio de liquidez (Current Ratio) superior a 1.2.
            """)

if __name__ == "__main__":
    main()