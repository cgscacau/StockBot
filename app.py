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

# Optimization Libraries
from scipy.optimize import minimize
import cvxpy as cp

# --- Configuração da Página ---
st.set_page_config(layout="wide", page_title="Dashboard AI de Análise Financeira", page_icon="🤖")

# --- Funções de Machine Learning ---

def preparar_dados_ml(df, lookback=60):
    """Prepara dados para modelos de ML."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1, 1))
    
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i, 0])
        y.append(scaled_data[i, 0])
    
    return np.array(X), np.array(y), scaler

@st.cache_data
def treinar_modelo_ml(df, modelo_tipo='Random Forest', lookback=60):
    """Treina modelo de ML simplificado."""
    X, y, scaler = preparar_dados_ml(df, lookback)
    
    if len(X) < 50:
        return None
    
    # Split train/test
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    if modelo_tipo == 'Random Forest':
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        
        with st.spinner('Treinando Random Forest...'):
            model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        
        # Previsões futuras
        future_predictions = []
        current_sequence = X[-1].copy()
        
        for _ in range(30):
            next_pred = model.predict(current_sequence.reshape(1, -1))[0]
            future_predictions.append(next_pred)
            # Atualizar sequência
            current_sequence = np.roll(current_sequence, -1)
            current_sequence[-1] = next_pred
        
    elif modelo_tipo == 'SVR':
        model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
        
        with st.spinner('Treinando SVR...'):
            model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        
        # Previsões futuras
        future_predictions = []
        current_sequence = X[-1].copy()
        
        for _ in range(30):
            next_pred = model.predict(current_sequence.reshape(1, -1))[0]
            future_predictions.append(next_pred)
            current_sequence = np.roll(current_sequence, -1)
            current_sequence[-1] = next_pred
    
    # Desnormalizar previsões
    predictions = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
    y_test_real = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1)).flatten()
    
    # Calcular métricas
    mse = mean_squared_error(y_test_real, predictions)
    mae = mean_absolute_error(y_test_real, predictions)
    rmse = np.sqrt(mse)
    
    return {
        'predictions': predictions,
        'y_test': y_test_real,
        'future_predictions': future_predictions,
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'test_dates': df.index[-len(predictions):]
    }

# --- Funções de Otimização de Carteira (Corrigidas) ---

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
    
    return cov_matrix

def otimizacao_markowitz_corrigida(retornos_esperados, matriz_cov, risk_free_rate=0.02):
    """Otimização de Markowitz corrigida."""
    n_assets = len(retornos_esperados)
    
    # Verificar se a matriz é positiva definida
    eigenvalues = np.linalg.eigvals(matriz_cov.values)
    if np.any(eigenvalues <= 0):
        # Adicionar regularização se necessário
        matriz_cov += np.eye(n_assets) * 1e-6
    
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
        except Exception as e:
            continue
    
    return efficient_portfolios

def monte_carlo_optimization(dados_completos, num_portfolios=5000):
    """Otimização por Monte Carlo simplificada."""
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
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
    return df

def calcular_metricas_performance(df):
    """Calcula métricas de performance."""
    returns = df['Close'].pct_change().dropna()
    
    if len(returns) == 0:
        return {}
    
    metricas = {
        'Retorno Total (%)': ((df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1) * 100,
        'Retorno Anualizado (%)': (returns.mean() * 252) * 100,
        'Volatilidade (%)': (returns.std() * np.sqrt(252)) * 100,
        'Sharpe Ratio': (returns.mean() * 252) / (returns.std() * np.sqrt(252)) if returns.std() != 0 else 0,
        'Máximo Drawdown (%)': ((df['Close'] / df['Close'].expanding().max()) - 1).min() * 100,
    }
    
    return metricas

# --- Interface Principal ---

st.title("🤖 Dashboard AI de Análise Financeira")
st.markdown("*Machine Learning • Otimização de Carteiras • Análise Técnica*")

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
    data_inicial_padrao = (datetime.now() - timedelta(days=730)).date()
    
    data_inicial = st.date_input("📅 Data Inicial", value=data_inicial_padrao)
    data_final = st.date_input("📅 Data Final", value=data_final_padrao)
    
    st.markdown("---")
    
    # Opções de ML
    st.subheader("🤖 Machine Learning")
    modelo_ml = st.selectbox(
        "Escolha o Modelo",
        options=['Random Forest', 'SVR']
    )
    
    # Opções de Otimização
    st.subheader("📊 Otimização")
    metodo_otimizacao = st.selectbox(
        "Método",
        options=['Monte Carlo', 'Markowitz']
    )

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

# --- Seção 2: Análise Técnica ---
st.header("📈 Análise Técnica")

for ticker, df in dados_completos.items():
    st.subheader(f"📊 {ticker}")
    
    # Gráfico principal
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=['Preço e Médias Móveis', 'Volume'],
        row_heights=[0.7, 0.3]
    )
    
    # Preço
    fig.add_trace(
        go.Scatter(x=df.index, y=df['Close'], name='Preço', line=dict(color='blue')),
        row=1, col=1
    )
    
    # Médias móveis
    df_indicadores = calcular_indicadores_tecnicos(df)
    fig.add_trace(
        go.Scatter(x=df_indicadores.index, y=df_indicadores['SMA_20'], 
                  name='SMA 20', line=dict(color='orange')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df_indicadores.index, y=df_indicadores['SMA_50'], 
                  name='SMA 50', line=dict(color='red')),
        row=1, col=1
    )
    
    # Volume
    fig.add_trace(
        go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color='lightblue'),
        row=2, col=1
    )
    
    fig.update_layout(height=600, title=f"Análise Técnica - {ticker}")
    st.plotly_chart(fig, use_container_width=True)
    
    # Métricas
    metricas = calcular_metricas_performance(df)
    if metricas:
        cols_metricas = st.columns(len(metricas))
        for i, (metrica, valor) in enumerate(metricas.items()):
            with cols_metricas[i % len(cols_metricas)]:
                st.metric(metrica, f"{valor:.2f}")

# --- Seção 3: Machine Learning ---
st.header("🔮 Previsão com Machine Learning")

if len(dados_completos) >= 1:
    ticker_para_ml = st.selectbox("Escolha a ação para previsão:", list(dados_completos.keys()))
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        if st.button(f"🚀 Treinar {modelo_ml}", type="primary"):
            df_ml = dados_completos[ticker_para_ml]
            
            if len(df_ml) < 100:
                st.error("❌ Dados insuficientes (mínimo: 100 dias)")
            else:
                resultados_ml = treinar_modelo_ml(df_ml, modelo_ml)
                
                if resultados_ml:
                    # Armazenar resultados na sessão
                    st.session_state['resultados_ml'] = resultados_ml
                    st.session_state['ticker_ml'] = ticker_para_ml
                    st.session_state['modelo_ml'] = modelo_ml
                    st.success("✅ Modelo treinado com sucesso!")
                else:
                    st.error("❌ Falha no treinamento")
    
    with col2:
        # Exibir resultados se existirem
        if 'resultados_ml' in st.session_state:
            resultados = st.session_state['resultados_ml']
            ticker_ml = st.session_state['ticker_ml']
            modelo_usado = st.session_state['modelo_ml']
            df_ml = dados_completos[ticker_ml]
            
            # Gráfico de previsões
            fig = go.Figure()
            
            # Dados históricos
            fig.add_trace(go.Scatter(
                x=df_ml.index,
                y=df_ml['Close'],
                name='Preço Real',
                line=dict(color='blue')
            ))
            
            # Previsões no teste
            fig.add_trace(go.Scatter(
                x=resultados['test_dates'],
                y=resultados['predictions'],
                name=f'Previsões {modelo_usado}',
                line=dict(color='red', dash='dash')
            ))
            
            # Previsões futuras
            future_dates = pd.date_range(start=df_ml.index[-1] + timedelta(days=1), periods=30)
            fig.add_trace(go.Scatter(
                x=future_dates,
                y=resultados['future_predictions'],
                name='Previsões Futuras',
                line=dict(color='green', dash='dot')
            ))
            
            fig.update_layout(
                title=f'Previsões {modelo_usado} - {ticker_ml}',
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Métricas
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("RMSE", f"{resultados['rmse']:.4f}")
            with col2:
                st.metric("MAE", f"{resultados['mae']:.4f}")
            with col3:
                accuracy = 100 * (1 - resultados['mae'] / np.mean(resultados['y_test']))
                st.metric("Acurácia", f"{accuracy:.2f}%")

# --- Seção 4: Otimização de Carteiras ---
if len(dados_completos) > 1:
    st.header("⚖️ Otimização de Carteiras")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        if st.button("🔧 Otimizar Carteira", type="primary"):
            if metodo_otimizacao == 'Monte Carlo':
                with st.spinner("Executando simulação Monte Carlo..."):
                    resultados_mc, pesos_mc, tickers = monte_carlo_optimization(dados_completos)
                    st.session_state['otimizacao'] = {
                        'tipo': 'Monte Carlo',
                        'resultados': resultados_mc,
                        'pesos': pesos_mc,
                        'tickers': tickers
                    }
                    st.success("✅ Otimização concluída!")
            
            elif metodo_otimizacao == 'Markowitz':
                try:
                    retornos_esperados = calcular_retornos_esperados(dados_completos)
                    matriz_cov = calcular_matriz_covariancia(dados_completos)
                    
                    with st.spinner("Calculando fronteira eficiente..."):
                        carteiras_eficientes = otimizacao_markowitz_corrigida(retornos_esperados, matriz_cov)
                    
                    if carteiras_eficientes:
                        st.session_state['otimizacao'] = {
                            'tipo': 'Markowitz',
                            'carteiras': carteiras_eficientes,
                            'tickers': list(dados_completos.keys())
                        }
                        st.success("✅ Fronteira eficiente calculada!")
                    else:
                        st.error("❌ Não foi possível calcular a fronteira eficiente")
                        
                except Exception as e:
                    st.error(f"❌ Erro na otimização: {str(e)}")
    
    with col2:
        # Exibir resultados da otimização
        if 'otimizacao' in st.session_state:
            otim = st.session_state['otimizacao']
            
            if otim['tipo'] == 'Monte Carlo':
                resultados_mc = otim['resultados']
                pesos_mc = otim['pesos']
                tickers = otim['tickers']
                
                # Gráfico de dispersão
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=resultados_mc[1],  # Risco
                    y=resultados_mc[0],  # Retorno
                    mode='markers',
                    marker=dict(
                        size=3,
                        color=resultados_mc[2],  # Sharpe
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="Sharpe Ratio")
                    ),
                    name='Carteiras'
                ))
                
                # Melhor Sharpe
                max_sharpe_idx = np.argmax(resultados_mc[2])
                fig.add_trace(go.Scatter(
                    x=[resultados_mc[1, max_sharpe_idx]],
                    y=[resultados_mc[0, max_sharpe_idx]],
                    mode='markers',
                    marker=dict(size=15, color='red', symbol='star'),
                    name='Melhor Sharpe'
                ))
                
                fig.update_layout(
                    title='Monte Carlo - Carteiras Otimizadas',
                    xaxis_title='Risco',
                    yaxis_title='Retorno',
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Carteira ótima
                st.subheader("🏆 Carteira Ótima")
                melhor_pesos = pesos_mc[max_sharpe_idx]
                
                pesos_df = pd.DataFrame({
                    'Ativo': tickers,
                    'Peso': melhor_pesos
                }).sort_values('Peso', ascending=False)
                
                st.dataframe(pesos_df.style.format({'Peso': '{:.2%}'}))
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Retorno", f"{resultados_mc[0, max_sharpe_idx]:.2%}")
                with col2:
                    st.metric("Risco", f"{resultados_mc[1, max_sharpe_idx]:.2%}")
                with col3:
                    st.metric("Sharpe", f"{resultados_mc[2, max_sharpe_idx]:.3f}")
            
            elif otim['tipo'] == 'Markowitz':
                carteiras = otim['carteiras']
                tickers = otim['tickers']
                
                if carteiras:
                    # Fronteira eficiente
                    returns_ef = [p['return'] for p in carteiras]
                    risks_ef = [p['risk'] for p in carteiras]
                    sharpes_ef = [p['sharpe'] for p in carteiras]
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=risks_ef,
                        y=returns_ef,
                        mode='markers+lines',
                        marker=dict(
                            size=8,
                            color=sharpes_ef,
                            colorscale='Viridis',
                            showscale=True,
                            colorbar=dict(title="Sharpe Ratio")
                        ),
                        name='Fronteira Eficiente'
                    ))
                    
                    fig.update_layout(
                        title='Fronteira Eficiente de Markowitz',
                        xaxis_title='Risco',
                        yaxis_title='Retorno',
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Melhor carteira
                    melhor_carteira = max(carteiras, key=lambda x: x['sharpe'])
                    
                    st.subheader("🏆 Carteira Ótima")
                    pesos_otimos = pd.DataFrame({
                        'Ativo': tickers,
                        'Peso': melhor_carteira['weights']
                    }).sort_values('Peso', ascending=False)
                    
                    st.dataframe(pesos_otimos.style.format({'Peso': '{:.2%}'}))
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Retorno", f"{melhor_carteira['return']:.2%}")
                    with col2:
                        st.metric("Risco", f"{melhor_carteira['risk']:.2%}")
                    with col3:
                        st.metric("Sharpe", f"{melhor_carteira['sharpe']:.3f}")

# Footer
st.markdown("---")
st.markdown("*Dashboard com ML, Otimização e Análise Técnica*")
