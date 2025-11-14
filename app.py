import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import json
from datetime import datetime, timedelta
import ta  # Biblioteca para análise técnica

# --- Configuração da Página ---
st.set_page_config(layout="wide", page_title="Dashboard Avançado de Análise de Ações", page_icon="📈")

# --- Funções Auxiliares ---

def carregar_tickers():
    """Carrega a lista de tickers de ações do arquivo config.json."""
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
        return config.get('tickers', [])
    except FileNotFoundError:
        st.error("Arquivo 'config.json' não encontrado.")
        return []

@st.cache_data
def baixar_dados_completos(tickers, start_date, end_date):
    """Baixa dados completos (OHLCV) para análise técnica."""
    dados_completos = {}
    for ticker in tickers:
        try:
            ticker_data = yf.Ticker(ticker).history(start=start_date, end=end_date, raise_errors=False)
            if not ticker_data.empty:
                dados_completos[ticker] = ticker_data
            else:
                st.warning(f"Dados não disponíveis para {ticker}")
        except Exception as e:
            st.error(f"Erro ao baixar {ticker}: {str(e)}")
    return dados_completos

def calcular_indicadores_tecnicos(df):
    """Calcula indicadores técnicos usando a biblioteca ta."""
    df = df.copy()
    
    # Médias móveis
    df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
    df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
    df['EMA_12'] = ta.trend.ema_indicator(df['Close'], window=12)
    df['EMA_26'] = ta.trend.ema_indicator(df['Close'], window=26)
    
    # Bandas de Bollinger
    bollinger = ta.volatility.BollingerBands(df['Close'])
    df['BB_High'] = bollinger.bollinger_hband()
    df['BB_Low'] = bollinger.bollinger_lband()
    df['BB_Mid'] = bollinger.bollinger_mavg()
    
    # RSI
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
    
    # MACD
    macd = ta.trend.MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Hist'] = macd.macd_diff()
    
    # Volume médio
    df['Volume_SMA'] = ta.volume.sma_ease_of_movement(df['High'], df['Low'], df['Volume'])
    
    return df

def calcular_metricas_performance(df):
    """Calcula métricas de performance e risco."""
    returns = df['Close'].pct_change().dropna()
    
    metricas = {
        'Retorno Total (%)': ((df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1) * 100,
        'Retorno Anualizado (%)': (returns.mean() * 252) * 100,
        'Volatilidade Anualizada (%)': (returns.std() * np.sqrt(252)) * 100,
        'Sharpe Ratio': (returns.mean() * 252) / (returns.std() * np.sqrt(252)) if returns.std() != 0 else 0,
        'Máximo Drawdown (%)': ((df['Close'] / df['Close'].expanding().max()) - 1).min() * 100,
        'Dias de Alta': (returns > 0).sum(),
        'Dias de Baixa': (returns < 0).sum()
    }
    
    return metricas

# --- Interface do Usuário ---

# Sidebar
with st.sidebar:
    st.title("🎛️ Painel de Controle")
    
    tickers_disponiveis = carregar_tickers()
    
    tickers_selecionados = st.multiselect(
        "📈 Selecione as Ações",
        options=tickers_disponiveis,
        default=['PETR4.SA']
    )
    
    data_final_padrao = datetime.now().date()
    data_inicial_padrao = (datetime.now() - timedelta(days=365)).date()
    
    data_inicial = st.date_input("📅 Data Inicial", value=data_inicial_padrao)
    data_final = st.date_input("📅 Data Final", value=data_final_padrao)
    
    st.markdown("---")
    
    # Opções de análise
    st.subheader("🔧 Opções de Análise")
    mostrar_indicadores = st.checkbox("Indicadores Técnicos", value=True)
    mostrar_candlestick = st.checkbox("Gráfico Candlestick", value=True)
    mostrar_volume = st.checkbox("Volume", value=True)
    mostrar_correlacao = st.checkbox("Matriz de Correlação", value=len(tickers_selecionados) > 1)

# Validação de datas
hoje = datetime.now().date()
if data_inicial > data_final:
    st.error("❌ A Data Inicial não pode ser maior que a Data Final.")
    st.stop()
elif data_final > hoje:
    st.warning(f"⚠️ Ajustando data final para hoje ({hoje.strftime('%d/%m/%Y')})")
    data_final = hoje

# --- Dashboard Principal ---

st.title("📊 Dashboard Avançado de Análise de Ações")
st.markdown(f"Analisando **{len(tickers_selecionados)}** ação(ões) de *{data_inicial.strftime('%d/%m/%Y')}* até *{data_final.strftime('%d/%m/%Y')}*")

if not tickers_selecionados:
    st.warning("⚠️ Selecione pelo menos uma ação na barra lateral.")
    st.stop()

# Baixar dados
dados_completos = baixar_dados_completos(tickers_selecionados, data_inicial, data_final)

if not dados_completos:
    st.error("❌ Não foi possível obter dados para as ações selecionadas.")
    st.stop()

# --- Seção 1: Resumo Executivo ---
st.header("📋 Resumo Executivo")

cols = st.columns(len(tickers_selecionados))
for i, (ticker, df) in enumerate(dados_completos.items()):
    with cols[i]:
        preco_atual = df['Close'].iloc[-1]
        preco_anterior = df['Close'].iloc[-2] if len(df) > 1 else preco_atual
        variacao = ((preco_atual / preco_anterior) - 1) * 100
        
        st.metric(
            label=f"{ticker}",
            value=f"R$ {preco_atual:.2f}",
            delta=f"{variacao:.2f}%"
        )

# --- Seção 2: Gráficos Principais ---
for ticker, df in dados_completos.items():
    st.header(f"📈 Análise Detalhada: {ticker}")
    
    # Calcular indicadores técnicos
    if mostrar_indicadores:
        df_com_indicadores = calcular_indicadores_tecnicos(df)
    else:
        df_com_indicadores = df
    
    # Criar subplots
    if mostrar_candlestick:
        fig = make_subplots(
            rows=3 if mostrar_volume else 2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=['Preço e Indicadores', 'RSI', 'Volume'] if mostrar_volume else ['Preço e Indicadores', 'RSI'],
            row_width=[0.2, 0.1, 0.1] if mostrar_volume else [0.2, 0.1]
        )
        
        # Candlestick
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name=ticker
            ),
            row=1, col=1
        )
        
        # Indicadores técnicos
        if mostrar_indicadores:
            fig.add_trace(go.Scatter(x=df_com_indicadores.index, y=df_com_indicadores['SMA_20'], 
                                   name='SMA 20', line=dict(color='orange')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_com_indicadores.index, y=df_com_indicadores['SMA_50'], 
                                   name='SMA 50', line=dict(color='red')), row=1, col=1)
            
            # Bandas de Bollinger
            fig.add_trace(go.Scatter(x=df_com_indicadores.index, y=df_com_indicadores['BB_High'], 
                                   name='BB High', line=dict(color='gray', dash='dash')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_com_indicadores.index, y=df_com_indicadores['BB_Low'], 
                                   name='BB Low', line=dict(color='gray', dash='dash'), 
                                   fill='tonexty', fillcolor='rgba(128,128,128,0.1)'), row=1, col=1)
        
        # RSI
        if mostrar_indicadores:
            fig.add_trace(go.Scatter(x=df_com_indicadores.index, y=df_com_indicadores['RSI'], 
                                   name='RSI', line=dict(color='purple')), row=2, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
        # Volume
        if mostrar_volume:
            fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', 
                               marker_color='lightblue'), row=3, col=1)
        
        fig.update_layout(height=800, title=f"Análise Técnica Completa - {ticker}")
        st.plotly_chart(fig, use_container_width=True)
    
    else:
        # Gráfico de linha simples
        fig = px.line(df, y='Close', title=f'Preço de Fechamento - {ticker}')
        st.plotly_chart(fig, use_container_width=True)
    
    # Métricas de Performance
    st.subheader("📊 Métricas de Performance")
    metricas = calcular_metricas_performance(df)
    
    cols_metricas = st.columns(4)
    for i, (metrica, valor) in enumerate(metricas.items()):
        with cols_metricas[i % 4]:
            if isinstance(valor, (int, float)):
                st.metric(metrica, f"{valor:.2f}")
            else:
                st.metric(metrica, valor)

# --- Seção 3: Análise Comparativa ---
if len(dados_completos) > 1:
    st.header("🔄 Análise Comparativa")
    
    # Normalizar preços para comparação
    precos_normalizados = pd.DataFrame()
    for ticker, df in dados_completos.items():
        precos_normalizados[ticker] = (df['Close'] / df['Close'].iloc[0]) * 100
    
    fig = px.line(precos_normalizados, title="Performance Relativa (Base 100)")
    fig.update_layout(yaxis_title="Performance (%)")
    st.plotly_chart(fig, use_container_width=True)
    
    # Matriz de Correlação
    if mostrar_correlacao:
        st.subheader("🔗 Matriz de Correlação")
        returns_df = pd.DataFrame()
        for ticker, df in dados_completos.items():
            returns_df[ticker] = df['Close'].pct_change()
        
        correlation_matrix = returns_df.corr()
        fig = px.imshow(correlation_matrix, text_auto=True, aspect="auto",
                       title="Correlação entre Retornos Diários")
        st.plotly_chart(fig, use_container_width=True)

# --- Seção 4: Simulador de Carteira ---
st.header("💼 Simulador de Carteira")

if len(dados_completos) > 1:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Pesos da Carteira")
        pesos = {}
        for ticker in dados_completos.keys():
            pesos[ticker] = st.slider(f"{ticker}", 0.0, 1.0, 1.0/len(dados_completos), 0.05)
        
        total_peso = sum(pesos.values())
        if abs(total_peso - 1.0) > 0.01:
            st.warning(f"⚠️ Soma dos pesos: {total_peso:.2f} (deve ser 1.0)")
    
    with col2:
        if abs(total_peso - 1.0) <= 0.01:
            # Calcular performance da carteira
            carteira_returns = pd.Series(0, index=list(dados_completos.values())[0].index)
            for ticker, peso in pesos.items():
                returns = dados_completos[ticker]['Close'].pct_change().fillna(0)
                carteira_returns += returns * peso
            
            carteira_valor = (1 + carteira_returns).cumprod() * 10000  # Base R$ 10.000
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=carteira_valor.index, y=carteira_valor, 
                                   name='Carteira', line=dict(color='green', width=3)))
            fig.update_layout(title="Evolução da Carteira (Base: R$ 10.000)", 
                            yaxis_title="Valor (R$)")
            st.plotly_chart(fig, use_container_width=True)
            
            # Métricas da carteira
            metricas_carteira = calcular_metricas_performance(pd.DataFrame({'Close': carteira_valor}))
            
            cols_carteira = st.columns(3)
            for i, (metrica, valor) in enumerate(list(metricas_carteira.items())[:3]):
                with cols_carteira[i]:
                    st.metric(metrica, f"{valor:.2f}")

# --- Dados Tabulares ---
with st.expander("📋 Ver Dados Tabulares"):
    for ticker, df in dados_completos.items():
        st.subheader(f"Dados - {ticker}")
        st.dataframe(df.tail(10))
