import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import json
from datetime import datetime, timedelta

# --- Configuração da Página ---
# st.set_page_config define as configurações iniciais da página.
# layout="wide" faz com que o conteúdo ocupe toda a largura da tela.
st.set_page_config(layout="wide", page_title="Dashboard de Ações")

# --- Funções Auxiliares ---

# Função para carregar os tickers do arquivo JSON
def carregar_tickers():
    """Carrega a lista de tickers de ações do arquivo config.json."""
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
        return config.get('tickers', [])
    except FileNotFoundError:
        st.error("Arquivo 'config.json' não encontrado. Verifique se o arquivo existe.")
        return []

# Função para baixar os dados das ações
# O decorador @st.cache_data é uma feature poderosa do Streamlit.
# Ele armazena o resultado da função em cache. Se a função for chamada novamente
# com os mesmos parâmetros (tickers, start_date, end_date), o Streamlit retorna
# o resultado armazenado em vez de executar a função novamente.
# Isso evita downloads repetidos e torna o app muito mais rápido.
@st.cache_data
def baixar_dados_acoes(tickers, start_date, end_date):
    """Baixa os dados históricos de fechamento para uma lista de tickers."""
    # O yfinance pode baixar dados para múltiplos tickers de uma vez
    dados = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    return dados

# --- Interface do Usuário (Sidebar) ---

# st.sidebar cria uma barra lateral na aplicação
with st.sidebar:
    st.title("Painel de Controle")
    
    # Carrega os tickers disponíveis
    tickers_disponiveis = carregar_tickers()
    
    # st.multiselect permite que o usuário selecione múltiplos itens de uma lista
    tickers_selecionados = st.multiselect(
        "Selecione as Ações",
        options=tickers_disponiveis,
        default=['PETR4.SA', 'AAPL'] # Ações padrão ao carregar o app
    )
    
    # Define as datas padrão
    data_final_padrao = datetime.now()
    data_inicial_padrao = data_final_padrao - timedelta(days=365) # Um ano atrás
    
    # st.date_input cria um seletor de data
    data_inicial = st.date_input("Data Inicial", value=data_inicial_padrao)
    data_final = st.date_input("Data Final", value=data_final_padrao)

# --- Lógica Principal e Exibição no Dashboard ---

st.title("Dashboard de Análise de Ações")
st.markdown(f"Analisando dados de **{len(tickers_selecionados)}** ação(ões) de *{data_inicial.strftime('%d/%m/%Y')}* até *{data_final.strftime('%d/%m/%Y')}*.")

# Verifica se o usuário selecionou alguma ação
if not tickers_selecionados:
    st.warning("Por favor, selecione pelo menos uma ação na barra lateral.")
else:
    # Baixa os dados apenas se houver ações selecionadas
    dados_acoes = baixar_dados_acoes(tickers_selecionados, data_inicial, data_final)
    
    if dados_acoes.empty:
        st.error("Não foi possível obter dados para as ações e o período selecionado. Verifique os tickers ou o intervalo de datas.")
    else:
        # --- Seção de Gráficos ---
        st.header("Gráfico de Preços de Fechamento")

        # O Pandas DataFrame retornado pelo yfinance tem os tickers como colunas.
        # Para usar com o Plotly Express de forma eficiente, é melhor "derreter" (melt) o dataframe
        # para ter uma coluna para o ticker, uma para a data e uma para o preço.
        dados_plot = dados_acoes.stack().reset_index()
        dados_plot.columns = ['Data', 'Ticker', 'Preço']

        # Cria o gráfico de linha interativo com Plotly Express
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
        # st.plotly_chart exibe um gráfico Plotly no Streamlit
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
                
                # Usamos colunas para organizar melhor as informações
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
