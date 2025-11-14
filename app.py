import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import json
from datetime import datetime, timedelta
import ta
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score

# Deep Learning Libraries
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    st.warning("⚠️ TensorFlow não disponível. Modelos LSTM/GRU desabilitados.")

# Optimization Libraries
from scipy.optimize import minimize
try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False
    st.warning("⚠️ CVXPY não disponível. Otimização Markowitz desabilitada.")

# --- Configuração da Página ---
st.set_page_config(layout="wide", page_title="Dashboard AI de Análise Financeira", page_icon="🤖")

# --- Funções de Machine Learning ---

def preparar_dados_ml(df, lookback=60):
    """Prepara dados para modelos de ML/Deep Learning."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1, 1))
    
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i, 0])
        y.append(scaled_data[i, 0])
    
    return np.array(X), np.array(y), scaler

def criar_modelo_lstm(input_shape, units=50):
    """Cria modelo LSTM para previsão de preços."""
    if not TENSORFLOW_AVAILABLE:
        return None
    
    model = Sequential([
        LSTM(units, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(units, return_sequences=True),
        Dropout(0.2),
        LSTM(units),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

def criar_modelo_gru(input_shape, units=50):
    """Cria modelo GRU para previsão de preços."""
    if not TENSORFLOW_AVAILABLE:
        return None
    
    model = Sequential([
        GRU(units, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        GRU(units, return_sequences=True),
        Dropout(0.2),
        GRU(units),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

@st.cache_data
def treinar_modelo_ml(df, modelo_tipo='Random Forest', lookback=60, epochs=50):
    """Treina modelo de ML e retorna previsões."""
    X, y, scaler = preparar_dados_ml(df, lookback)
    
    if len(X) < 50:
        return None
    
    # Split train/test
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    try:
        if modelo_tipo == 'LSTM' and TENSORFLOW_AVAILABLE:
            X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
            X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
            
            model = criar_modelo_lstm((X_train.shape[1], 1))
            if model is None:
                return None
            
            with st.spinner(f'Treinando modelo {modelo_tipo}... Isso pode levar alguns minutos.'):
                history = model.fit(X_train, y_train, epochs=epochs, batch_size=32, 
                                  validation_data=(X_test, y_test), verbose=0)
            
            predictions = model.predict(X_test, verbose=0)
            
            # Previsões futuras
            future_predictions = []
            current_sequence = X[-1].reshape(1, lookback, 1)
            
            for _ in range(30):
                next_pred = model.predict(current_sequence, verbose=0)
                future_predictions.append(next_pred[0, 0])
                # Atualizar sequência
                current_sequence = np.roll(current_sequence, -1, axis=1)
                current_sequence[0, -1, 0] = next_pred[0, 0]
            
        elif modelo_tipo == 'GRU' and TENSORFLOW_AVAILABLE:
            X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
            X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
            
            model = criar_modelo_gru((X_train.shape[1], 1))
            if model is None:
                return None
            
            with st.spinner(f'Treinando modelo {modelo_tipo}... Isso pode levar alguns minutos.'):
                history = model.fit(X_train, y_train, epochs=epochs, batch_size=32, 
                                  validation_data=(X_test, y_test), verbose=0)
            
            predictions = model.predict(X_test, verbose=0)
            
            # Previsões futuras
            future_predictions = []
            current_sequence = X[-1].reshape(1, lookback, 1)
            
            for _ in range(30):
                next_pred = model.predict(current_sequence, verbose=0)
                future_predictions.append(next_pred[0, 0])
                current_sequence = np.roll(current_sequence, -1, axis=1)
                current_sequence[0, -1, 0] = next_pred[0, 0]
            
        elif modelo_tipo == 'Random Forest':
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            
            with st.spinner('Treinando Random Forest...'):
                rf_model.fit(X_train, y_train)
            
            predictions = rf_model.predict(X_test).reshape(-1, 1)
            
            # Previsões futuras
            future_predictions = []
            current_sequence = X[-1].copy()
            
            for _ in range(30):
                next_pred = rf_model.predict(current_sequence.reshape(1, -1))[0]
                future_predictions.append(next_pred)
                current_sequence = np.roll(current_sequence, -1)
                current_sequence[-1] = next_pred
            
        elif modelo_tipo == 'SVR':
            svr_model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
            
            with st.spinner('Treinando SVR...'):
                svr_model.fit(X_train, y_train)
            
            predictions = svr_model.predict(X_test).reshape(-1, 1)
            
            # Previsões futuras
            future_predictions = []
            current_sequence = X[-1].copy()
            
            for _ in range(30):
                next_pred = svr_model.predict(current_sequence.reshape(1, -1))[0]
                future_predictions.append(next_pred)
                current_sequence = np.roll(current_sequence, -1)
                current_sequence[-1] = next_pred
        
        else:
            return None
        
        # Desnormalizar previsões
        predictions = scaler.inverse_transform(predictions)
        y_test_real = scaler.inverse_transform(y_test.reshape(-1, 1))
        future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
        
        # Calcular métricas
        mse = mean_squared_error(y_test_real, predictions)
        mae = mean_absolute_error(y_test_real, predictions)
        rmse = np.sqrt(mse)
        
        return {
            'predictions': predictions.flatten(),
            'y_test': y_test_real.flatten(),
            'future_predictions': future_predictions.flatten(),
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'test_dates': df.index[-len(predictions):]
        }
        
    except Exception as e:
        st.error(f"Erro no treinamento do modelo {modelo_tipo}: {str(e)}")
        return None

# --- Funções de Otimização de Carteira ---

def calcular_retornos_esperados(dados_completos):
    """Calcula retornos esperados para otimização."""
    returns = pd.DataFrame()
    for ticker, df in dados_completos.items():
        returns[ticker] = df['Close'].pct_change().dropna()
    
    return returns.mean() * 252  # Anualizado

def calcular_matriz_covariancia(dados_completos):
    """Calcula matriz de covariância simétrica."""
    returns = pd.DataFrame()
    for ticker, df in dados_completos.items():
        returns[ticker] = df['Close'].pct_change().dropna()
    
    cov_matrix = returns.cov() * 252  # Anualizada
    
    # Garantir que a matriz seja simétrica
    cov_matrix = (cov_matrix + cov_matrix.T) / 2
    
    # Verificar se é positiva definida e regularizar se necessário
    eigenvalues = np.linalg.eigvals(cov_matrix.values)
    if np.any(eigenvalues <= 0):
        cov_matrix += np.eye(len(cov_matrix)) * 1e-6
    
    return cov_matrix

def otimizacao_markowitz(retornos_esperados, matriz_cov, risk_free_rate=0.02):
    """Otimização de Markowitz para carteira ótima."""
    if not CVXPY_AVAILABLE:
        return []
    
    n_assets = len(retornos_esperados)
    
    try:
        # Variáveis de decisão (pesos)
        w = cp.Variable(n_assets)
        
        # Função objetivo e constraints
        portfolio_return = retornos_esperados.values @ w
        portfolio_variance = cp.quad_form(w, matriz_cov.values)
        
        # Constraints básicos
        constraints = [
            cp.sum(w) == 1,  # Soma dos pesos = 1
            w >= 0,  # Long-only
        ]
        
        # Resolver para diferentes níveis de risco
        risk_levels = np.linspace(0.05, 0.5, 15)
        efficient_portfolios = []
        
        for risk_level in risk_levels:
            try:
                # Minimizar variância sujeito a retorno mínimo
                constraints_temp = constraints + [portfolio_return >= risk_level]
                
                problem = cp.Problem(cp.Minimize(portfolio_variance), constraints_temp)
                problem.solve(solver=cp.ECOS, verbose=False)
                
                if problem.status == cp.OPTIMAL and w.value is not None:
                    portfolio_vol = np.sqrt(portfolio_variance.value)
                    sharpe = (portfolio_return.value - risk_free_rate) / portfolio_vol if portfolio_vol > 0 else 0
                    
                    efficient_portfolios.append({
                        'weights': w.value.copy(),
                        'return': portfolio_return.value,
                        'risk': portfolio_vol,
                        'sharpe': sharpe
                    })
            except Exception:
                continue
        
        return efficient_portfolios
        
    except Exception as e:
        st.error(f"Erro na otimização Markowitz: {str(e)}")
        return []

def monte_carlo_optimization(dados_completos, num_portfolios=5000):
    """Otimização por Monte Carlo."""
    tickers = list(dados_completos.keys())
    returns = pd.DataFrame()
    for ticker, df in dados_completos.items():
        returns[ticker] = df['Close'].pct_change().dropna()
    
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252
    
    # Garantir que a matriz seja simétrica e positiva definida
    cov_matrix = (cov_matrix + cov_matrix.T) / 2
    eigenvalues = np.linalg.eigvals(cov_matrix)
    if np.any(eigenvalues <= 0):
        cov_matrix += np.eye(len(tickers)) * 1e-6
    
    results = np.zeros((4, num_portfolios))
    weights_array = np.zeros((num_portfolios, len(tickers)))
    
    np.random.seed(42)
    
    for i in range(num_portfolios):
        weights = np.random.random(len(tickers))
        weights /= np.sum(weights)
        weights_array[i] = weights
        
        portfolio_return = np.sum(mean_returns * weights)
        portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
        portfolio_std = np.sqrt(portfolio_variance)
        sharpe_ratio = (portfolio_return - 0.02) / portfolio_std if portfolio_std > 0 else 0
        
        results[0, i] = portfolio_return
        results[1, i] = portfolio_std
        results[2, i] = sharpe_ratio
        results[3, i] = portfolio_variance
    
    return results, weights_array, tickers

# --- Funções Auxiliares ---

def carregar_tickers():
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
        return config.get('tickers', [])
    except FileNotFoundError:
        st.error("Arquivo 'config.json' não encontrado.")
        return []

@st.cache_data
def baixar_dados_completos(tickers, start_date, end_date):
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
    df = df.copy()
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
    
    return df

def calcular_metricas_performance(df):
    """Calcula métricas de performance e risco."""
    returns = df['Close'].pct_change().dropna()
    
    if len(returns) == 0:
        return {}
    
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

# --- Interface Principal ---

st.title("🤖 Dashboard AI de Análise Financeira")
st.markdown("*Machine Learning • Deep Learning • Otimização de Carteiras*")

# Sidebar
with st.sidebar:
    st.title("🎛️ Painel de Controle")
    
    tickers_disponiveis = carregar_tickers()
    tickers_selecionados = st.multiselect(
        "📈 Selecione as Ações",
        options=tickers_disponiveis,
        default=['PETR4.SA', 'AAPL']
    )
    
    data_final_padrao = datetime.now().date()
    data_inicial_padrao = (datetime.now() - timedelta(days=730)).date()  # 2 anos para ML
    
    data_inicial = st.date_input("📅 Data Inicial", value=data_inicial_padrao)
    data_final = st.date_input("📅 Data Final", value=data_final_padrao)
    
    st.markdown("---")
    
    # Opções de ML
    st.subheader("🤖 Machine Learning")
    modelos_disponiveis = ['Random Forest', 'SVR']
    if TENSORFLOW_AVAILABLE:
        modelos_disponiveis.extend(['LSTM', 'GRU'])
    
    modelo_ml = st.selectbox("Escolha o Modelo", options=modelos_disponiveis)
    
    epochs_ml = st.slider("Épocas de Treinamento", 10, 100, 50) if modelo_ml in ['LSTM', 'GRU'] else 50
    
    # Opções de Otimização
    st.subheader("📊 Otimização de Carteira")
    metodos_disponiveis = ['Monte Carlo']
    if CVXPY_AVAILABLE:
        metodos_disponiveis.append('Markowitz')
    
    metodo_otimizacao = st.selectbox("Método de Otimização", options=metodos_disponiveis)

# Validações
hoje = datetime.now().date()
if data_inicial > data_final:
    st.error("❌ A Data Inicial não pode ser maior que a Data Final.")
    st.stop()
elif data_final > hoje:
    st.warning(f"⚠️ Ajustando data final para hoje ({hoje.strftime('%d/%m/%Y')})")
    data_final = hoje

if not tickers_selecionados:
    st.warning("⚠️ Selecione pelo menos uma ação na barra lateral.")
    st.stop()

# Baixar dados
dados_completos = baixar_dados_completos(tickers_selecionados, data_inicial, data_final)

if not dados_completos:
    st.error("❌ Não foi possível obter dados.")
    st.stop()

# --- Seção 1: Resumo Executivo ---
st.header("📋 Resumo Executivo")

cols = st.columns(len(tickers_selecionados))
for i, (ticker, df) in enumerate(dados_completos.items()):
    with cols[i % len(cols)]:
        preco_atual = df['Close'].iloc[-1]
        preco_anterior = df['Close'].iloc[-2] if len(df) > 1 else preco_atual
        variacao = ((preco_atual / preco_anterior) - 1) * 100
        
        st.metric(
            label=f"{ticker}",
            value=f"${preco_atual:.2f}",
            delta=f"{variacao:.2f}%"
        )

# --- Seção 2: Gráficos Principais ---
for ticker, df in dados_completos.items():
    st.header(f"📈 Análise Detalhada: {ticker}")
    
    # Calcular indicadores técnicos
    df_com_indicadores = calcular_indicadores_tecnicos(df)
    
    # Criar subplots
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=['Preço e Indicadores', 'RSI', 'Volume'],
        row_heights=[0.6, 0.2, 0.2]
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
    fig.add_trace(go.Scatter(x=df_com_indicadores.index, y=df_com_indicadores['RSI'], 
                           name='RSI', line=dict(color='purple')), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    # Volume
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', 
                       marker_color='lightblue'), row=3, col=1)
    
    fig.update_layout(height=800, title=f"Análise Técnica Completa - {ticker}")
    st.plotly_chart(fig, use_container_width=True)
    
    # Métricas de Performance
    st.subheader("📊 Métricas de Performance")
    metricas = calcular_metricas_performance(df)
    
    if metricas:
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
    st.subheader("🔗 Matriz de Correlação")
    returns_df = pd.DataFrame()
    for ticker, df in dados_completos.items():
        returns_df[ticker] = df['Close'].pct_change()
    
    correlation_matrix = returns_df.corr()
    fig = px.imshow(correlation_matrix, text_auto=True, aspect="auto",
                   title="Correlação entre Retornos Diários")
    st.plotly_chart(fig, use_container_width=True)

# --- Seção 4: Machine Learning ---
st.header("🔮 Previsão de Preços com Machine Learning")

tab1, tab2, tab3 = st.tabs(["📈 Previsões", "📊 Métricas", "🎯 Análise"])

with tab1:
    if len(dados_completos) >= 1:
        ticker_para_ml = st.selectbox("Escolha a ação para previsão:", list(dados_completos.keys()))
        
        if st.button(f"🚀 Treinar Modelo {modelo_ml}", type="primary"):
            df_ml = dados_completos[ticker_para_ml]
            
            if len(df_ml) < 100:
                st.error("❌ Dados insuficientes para ML (mínimo: 100 dias)")
            else:
                resultados_ml = treinar_modelo_ml(df_ml, modelo_ml, epochs=epochs_ml)
                
                if resultados_ml:
                    # Gráfico de previsões
                    fig = go.Figure()
                    
                    # Dados históricos
                    fig.add_trace(go.Scatter(
                        x=df_ml.index,
                        y=df_ml['Close'],
                        name='Preço Real',
                        line=dict(color='blue')
                    ))
                    
                    # Previsões no período de teste
                    fig.add_trace(go.Scatter(
                        x=resultados_ml['test_dates'],
                        y=resultados_ml['predictions'],
                        name=f'Previsões {modelo_ml}',
                        line=dict(color='red', dash='dash')
                    ))
                    
                    # Previsões futuras
                    future_dates = pd.date_range(start=df_ml.index[-1] + timedelta(days=1), periods=30)
                    fig.add_trace(go.Scatter(
                        x=future_dates,
                        y=resultados_ml['future_predictions'],
                        name='Previsões Futuras',
                        line=dict(color='green', dash='dot')
                    ))
                    
                    fig.update_layout(
                        title=f'Previsões {modelo_ml} - {ticker_para_ml}',
                        xaxis_title='Data',
                        yaxis_title='Preço',
                        height=600
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Métricas do modelo
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("RMSE", f"{resultados_ml['rmse']:.4f}")
                    with col2:
                        st.metric("MAE", f"{resultados_ml['mae']:.4f}")
                    with col3:
                        accuracy = 100 * (1 - resultados_ml['mae'] / np.mean(resultados_ml['y_test']))
                        st.metric("Acurácia", f"{accuracy:.2f}%")
                else:
                    st.error("❌ Falha no treinamento do modelo.")

with tab2:
    st.subheader("📊 Comparação de Modelos")
    st.info("💡 **Dica:** Execute diferentes modelos e compare as métricas para escolher o melhor.")
    
    # Placeholder para comparação de modelos
    st.markdown("""
    **Interpretação das Métricas:**
    - **RMSE**: Raiz do Erro Quadrático Médio (menor é melhor)
    - **MAE**: Erro Absoluto Médio (menor é melhor)  
    - **Acurácia**: Percentual de acerto do modelo (maior é melhor)
    
    **Modelos Disponíveis:**
    - **Random Forest**: Ensemble learning, robusto e interpretável
    - **SVR**: Support Vector Regression, bom para dados não-lineares
    - **LSTM**: Redes neurais recorrentes, ideal para séries temporais
    - **GRU**: Alternativa ao LSTM, mais rápido de treinar
    """)

with tab3:
    st.subheader("🎯 Análise de Tendências")
    
    # Análise de tendência com Random Forest
    if st.button("🔍 Analisar Tendência (Random Forest)"):
        ticker_tendencia = list(dados_completos.keys())[0]
        df_tendencia = dados_completos[ticker_tendencia].copy()
        
        # Preparar features para classificação
        df_tendencia['Returns'] = df_tendencia['Close'].pct_change()
        df_tendencia['SMA_5'] = df_tendencia['Close'].rolling(5).mean()
        df_tendencia['SMA_20'] = df_tendencia['Close'].rolling(20).mean()
        df_tendencia['RSI'] = ta.momentum.rsi(df_tendencia['Close'])
        df_tendencia['Volume_MA'] = df_tendencia['Volume'].rolling(10).mean()
        
        # Target: 1 se preço subiu, 0 se desceu
        df_tendencia['Target'] = (df_tendencia['Returns'] > 0).astype(int)
        
        # Features
        features = ['SMA_5', 'SMA_20', 'RSI', 'Volume_MA']
        df_clean = df_tendencia[features + ['Target']].dropna()
        
        if len(df_clean) > 50:
            X = df_clean[features]
            y = df_clean['Target']
            
            # Split
            train_size = int(len(X) * 0.8)
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
            
            # Treinar classificador
            rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_classifier.fit(X_train, y_train)
            
            # Previsões
            y_pred = rf_classifier.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Feature importance
            importances = pd.DataFrame({
                'Feature': features,
                'Importance': rf_classifier.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Acurácia da Classificação", f"{accuracy:.2%}")
                
                # Gráfico de importância
                fig = px.bar(importances, x='Importance', y='Feature', 
                           title="Importância das Features")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Previsão para próximo período
                last_features = X.iloc[-1:].values
                proxima_tendencia = rf_classifier.predict(last_features)[0]
                probabilidade = rf_classifier.predict_proba(last_features)[0]
                
                if proxima_tendencia == 1:
                    st.success(f"📈 **Tendência Prevista:** ALTA")
                    st.info(f"Probabilidade: {probabilidade[1]:.2%}")
                else:
                    st.error(f"📉 **Tendência Prevista:** BAIXA")
                    st.info(f"Probabilidade: {probabilidade[0]:.2%}")

# --- Seção 5: Otimização de Carteiras ---
if len(dados_completos) > 1:
    st.header("⚖️ Otimização de Carteiras")
    
    tab1, tab2, tab3 = st.tabs(["🎯 Fronteira Eficiente", "🎲 Monte Carlo", "📊 Resultados"])
    
    with tab1:
        if metodo_otimizacao == 'Markowitz' and CVXPY_AVAILABLE:
            if st.button("🔧 Calcular Fronteira Eficiente"):
                retornos_esperados = calcular_retornos_esperados(dados_completos)
                matriz_cov = calcular_matriz_covariancia(dados_completos)
                
                carteiras_eficientes = otimizacao_markowitz(retornos_esperados, matriz_cov)
                
                if carteiras_eficientes:
                    # Extrair dados para plotar
                    returns_ef = [p['return'] for p in carteiras_eficientes]
                    risks_ef = [p['risk'] for p in carteiras_eficientes]
                    sharpes_ef = [p['sharpe'] for p in carteiras_eficientes]
                    
                    # Gráfico da fronteira eficiente
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=risks_ef,
                        y=returns_ef,
                        mode='markers+lines',
                        marker=dict(
                            size=10,
                            color=sharpes_ef,
                            colorscale='Viridis',
                            showscale=True,
                            colorbar=dict(title="Sharpe Ratio")
                        ),
                        name='Fronteira Eficiente'
                    ))
                    
                    fig.update_layout(
                        title='Fronteira Eficiente de Markowitz',
                        xaxis_title='Risco (Volatilidade)',
                        yaxis_title='Retorno Esperado',
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Melhor carteira (maior Sharpe)
                    melhor_carteira = max(carteiras_eficientes, key=lambda x: x['sharpe'])
                    
                    st.subheader("🏆 Carteira Ótima (Maior Sharpe Ratio)")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Retorno Esperado", f"{melhor_carteira['return']:.2%}")
                    with col2:
                        st.metric("Risco (Volatilidade)", f"{melhor_carteira['risk']:.2%}")
                    with col3:
                        st.metric("Sharpe Ratio", f"{melhor_carteira['sharpe']:.3f}")
                    
                    # Pesos da carteira ótima
                    pesos_otimos = pd.DataFrame({
                        'Ativo': list(dados_completos.keys()),
                        'Peso': melhor_carteira['weights']
                    }).sort_values('Peso', ascending=False)
                    
                    fig_pesos = px.pie(pesos_otimos, values='Peso', names='Ativo', 
                                     title="Alocação da Carteira Ótima")
                    st.plotly_chart(fig_pesos, use_container_width=True)
                
                else:
                    st.error("❌ Não foi possível otimizar a carteira.")
        else:
            st.info("💡 Otimização Markowitz não disponível. Use Monte Carlo.")
    
    with tab2:
        if st.button("🎲 Executar Simulação Monte Carlo"):
            with st.spinner("Executando 5.000 simulações..."):
                resultados_mc, pesos_mc, tickers = monte_carlo_optimization(dados_completos)
            
            # Gráfico de dispersão
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=resultados_mc[1],  # Risco
                y=resultados_mc[0],  # Retorno
                mode='markers',
                marker=dict(
                    size=3,
                    color=resultados_mc[2],  # Sharpe ratio
                    colorscale='RdYlBu',
                    showscale=True,
                    colorbar=dict(title="Sharpe Ratio")
                ),
                name='Carteiras Simuladas'
            ))
            
            # Destacar carteira com maior Sharpe
            max_sharpe_idx = np.argmax(resultados_mc[2])
            fig.add_trace(go.Scatter(
                x=[resultados_mc[1, max_sharpe_idx]],
                y=[resultados_mc[0, max_sharpe_idx]],
                mode='markers',
                marker=dict(size=15, color='red', symbol='star'),
                name='Melhor Sharpe'
            ))
            
            # Carteira de menor risco
            min_risk_idx = np.argmin(resultados_mc[1])
            fig.add_trace(go.Scatter(
                x=[resultados_mc[1, min_risk_idx]],
                y=[resultados_mc[0, min_risk_idx]],
                mode='markers',
                marker=dict(size=15, color='blue', symbol='diamond'),
                name='Menor Risco'
            ))
            
            fig.update_layout(
                title='Simulação Monte Carlo - 5.000 Carteiras',
                xaxis_title='Risco (Volatilidade)',
                yaxis_title='Retorno Esperado',
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Estatísticas
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Carteira com Maior Sharpe Ratio")
                melhor_pesos = pesos_mc[max_sharpe_idx]
                
                pesos_df = pd.DataFrame({
                    'Ativo': tickers,
                    'Peso': melhor_pesos
                }).sort_values('Peso', ascending=False)
                
                st.dataframe(pesos_df.style.format({'Peso': '{:.2%}'}))
                
                st.metric("Sharpe Ratio", f"{resultados_mc[2, max_sharpe_idx]:.3f}")
                st.metric("Retorno", f"{resultados_mc[0, max_sharpe_idx]:.2%}")
                st.metric("Risco", f"{resultados_mc[1, max_sharpe_idx]:.2%}")
            
            with col2:
                st.subheader("🛡️ Carteira de Menor Risco")
                menor_risco_pesos = pesos_mc[min_risk_idx]
                
                pesos_min_df = pd.DataFrame({
                    'Ativo': tickers,
                    'Peso': menor_risco_pesos
                }).sort_values('Peso', ascending=False)
                
                st.dataframe(pesos_min_df.style.format({'Peso': '{:.2%}'}))
                
                st.metric("Risco", f"{resultados_mc[1, min_risk_idx]:.2%}")
                st.metric("Retorno", f"{resultados_mc[0, min_risk_idx]:.2%}")
                st.metric("Sharpe Ratio", f"{resultados_mc[2, min_risk_idx]:.3f}")
    
    with tab3:
        st.subheader("📈 Comparação de Estratégias")
        
        # Tabela comparativa
        estrategias_data = {
            'Estratégia': ['Igual Peso', 'Maior Sharpe (MC)', 'Menor Risco (MC)', 'Markowitz'],
            'Descrição': [
                'Pesos iguais para todos os ativos',
                'Maximiza retorno ajustado ao risco',
                'Minimiza volatilidade da carteira',
                'Fronteira eficiente teórica'
            ],
            'Vantagem': [
                'Simplicidade e diversificação',
                'Melhor retorno ajustado',
                'Menor volatilidade',
                'Otimização matemática'
            ],
            'Desvantagem': [
                'Não considera correlações',
                'Pode concentrar em poucos ativos',
                'Menor retorno esperado',
                'Sensível a estimativas'
            ]
        }
        
        df_estrategias = pd.DataFrame(estrategias_data)
        st.dataframe(df_estrategias, use_container_width=True)

# --- Seção 6: Simulador de Carteira ---
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
            
            if metricas_carteira:
                cols_carteira = st.columns(3)
                for i, (metrica, valor) in enumerate(list(metricas_carteira.items())[:3]):
                    with cols_carteira[i]:
                        st.metric(metrica, f"{valor:.2f}")

# --- Dados Tabulares ---
with st.expander("📋 Ver Dados Tabulares"):
    for ticker, df in dados_completos.items():
        st.subheader(f"Dados - {ticker}")
        st.dataframe(df.tail(10))

# Footer
st.markdown("---")
st.markdown("*Dashboard desenvolvido com Streamlit, TensorFlow, Scikit-learn e bibliotecas de otimização financeira*")
