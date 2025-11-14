import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import json
from datetime import datetime, timedelta

# --- Configuração da Página ---
st.set_page_config(layout="wide", page_title="Dashboard de Ações")

# --- Funções Auxiliares ---

def carregar_tickers():
    """Carrega a lista de tickers de ações do arquivo config.json."""
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
        return config.get('tickers', [])
    except FileNotFoundError:
        st.error("Arquivo 'config.json' não encontrado. Verifique se o arquivo existe.")
        return []

@st.cache_data
def baixar_dados_acoes(tickers, start_date, end_date):
    """Baixa os dados históricos de fechamento para uma lista de tickers."""
    dados = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    return dados

# --- Interface do Usuário (Sidebar) ---

with st.sidebar:
    st.title("Painel de Controle")
    
    tickers_disponiveis = carregar_tickers()
    
    tickers_selecionados = st.multiselect(
        "Selecione as Ações",
        options=tickers_disponiveis,
        default=['PETR4.SA', 'AAPL']
    )
    
    data_final_padrao = datetime.now().date()  # Usa .date() para compatibilidade
    data_inicial_padrao = (datetime.now() - timedelta(days=365)).date()
    
    data_inicial = st.date_input("Data Inicial", value=data_inicial_padrao)
    data_final = st.date_input("Data Final", value=data_final_padrao)

# --- Validação de Datas (Nova Seção para Prevenir Erros) ---
hoje = datetime.now().date()
if data_inicial > data_final:
    st.error("A Data Inicial não pode ser maior que a Data Final. Por favor, ajuste as datas.")
elif data_final > hoje:
    st.warning(f"A Data Final selecionada ({data_final.strftime('%d/%m/%Y')}) está no futuro. Ajustando automaticamente para hoje ({hoje.strftime('%d/%m/%Y')}) para obter dados históricos disponíveis.")
    data_final = hoje  # Ajusta para hoje

# --- Lógica Principal e Exibição no Dashboard ---

st.title("Dashboard de Análise de Ações")
st.markdown(f"Analisando dados de **{len(tickers_selecionados)}** ação(ões) de *{data_inicial.strftime('%d/%m/%Y')}* até *{data_final.strftime('%d/%m/%Y')}*.")

if not tickers_selecionados:
    st.warning("Por favor, selecione pelo menos uma ação na barra lateral.")
else:
    dados_acoes = baixar_dados_acoes(tickers_selecionados, data_inicial, data_final)
    
    if dados_acoes.empty:
        st.error("Não foi possível obter dados para as ações e o período selecionado. "
                 "Verifique: \n- Se os tickers estão corretos (ex: 'PETR4.SA' para ações brasileiras). \n"
                 "- Se o intervalo de datas inclui dias de negociação (evite fins de semana/feriados ou futuro). \n"
                 "- Conexão com a internet ou limites da API do Yahoo Finance.")
    else:
        # --- Seção de Gráficos ---
        st.header("Gráfico de Preços de Fechamento")

        dados_plot = dados_acoes.stack().reset_index()
        dados_plot.columns = ['Data', 'Ticker', 'Preço']

        fig = px.line(
            dados_plot, 
            x='Data', 
            y='Preço', 
            color='Ticker',
            title='Série Histórica de Preços de Fechamento Ajustado'
        )
        fig.update_layout(
            xaxis_title="Data",
            yaxis_title="Preço de Fechamento (em moeda local)",
            legend_title="Ações"
        )
        st.plotly_chart(fig, use_container_width=True)

        # --- Seção de Dados Brutos ---
        with st.expander("Ver Dados Tabulares"):
            st.dataframe(dados_acoes)

        # --- Seção de Informações da Empresa ---
        st.header("Informações das Empresas")
        for ticker in tickers_selecionados:
            try:
                info = yf.Ticker(ticker).info
                st.subheader(f"{info.get('shortName', ticker)} ({ticker})")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Setor", info.get('sector', 'N/A'))
                    st.metric("País", info.get('country', 'N/A'))
                with col2:
                    st.metric("Moeda", info.get('currency', 'N/A'))
                    st.metric("Site", info.get('website', 'N/A'))
                
                st.info(f"**Resumo:** {info.get('longBusinessSummary', 'Resumo não disponível.')}")
                st.markdown("---")

            except Exception as e:
                st.error(f"Não foi possível obter informações para {ticker}. Erro: {e}")
