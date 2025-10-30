"""
Monte Carlo Portfolio Simulator - Streamlit App
Simulatore Monte Carlo per Portafoglio (Investitori Europei)

Deploy to Streamlit Cloud:
1. Create GitHub repo with this file as app.py
2. Add requirements.txt with: streamlit, yfinance, numpy, pandas, matplotlib
3. Go to share.streamlit.io and deploy

Performance: Streamlit can handle calculations well. For 10K simulations it takes 5-10 seconds.
The app runs on shared infrastructure but it's sufficient for Monte Carlo sims.
"""

import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import time

# Page configuration
st.set_page_config(
    page_title="Simulatore Portafoglio Monte Carlo",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stRadio > label {
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# FUNCTIONS
# ============================================================================

@st.cache_data(ttl=3600, show_spinner=False)
def get_etf_stats(ticker, period='10y'):
    """Get historical ETF statistics with better error handling"""
    try:
        # Download data
        etf = yf.Ticker(ticker)
        hist = etf.history(period=period)
        
        if hist.empty or len(hist) < 60:  # Need at least ~2 months of data
            return None
        
        # Calculate monthly returns
        monthly_prices = hist['Close'].resample('ME').last()
        monthly_returns = monthly_prices.pct_change().dropna()
        
        if len(monthly_returns) < 12:  # Need at least 1 year
            return None
        
        # Annualize statistics
        annual_return = (1 + monthly_returns.mean()) ** 12 - 1
        annual_vol = monthly_returns.std() * np.sqrt(12)
        
        # Get additional info
        info = etf.info
        name = info.get('longName', info.get('shortName', ticker))
        
        return {
            'ticker': ticker,
            'name': name,
            'annual_return': annual_return,
            'annual_volatility': annual_vol,
            'data': monthly_returns,
            'data_points': len(monthly_returns)
        }
    except Exception as e:
        return None

@st.cache_data(ttl=3600, show_spinner=False)
def get_correlation(ticker1, ticker2, period='10y'):
    """Calculate correlation between two ETFs"""
    try:
        # Download data for both
        data1 = yf.Ticker(ticker1).history(period=period)['Close'].resample('ME').last().pct_change()
        data2 = yf.Ticker(ticker2).history(period=period)['Close'].resample('ME').last().pct_change()
        
        # Align dates
        combined = pd.concat([data1, data2], axis=1).dropna()
        
        if len(combined) < 12:
            return 0.2  # Default
        
        corr = combined.corr().iloc[0, 1]
        return corr if not pd.isna(corr) else 0.2
    except:
        return 0.2  # Default fallback

def monte_carlo_dca(initial_investment, monthly_contribution, years, num_simulations,
                    stock_allocation, stock_return, stock_vol,
                    bond_return, bond_vol, correlation=0.2):
    """Run Monte Carlo simulation with DCA - Optimized version"""
    
    months = years * 12
    bond_allocation = 1 - stock_allocation
    
    # Pre-calculate monthly parameters
    monthly_stock_return = stock_return / 12
    monthly_stock_vol = stock_vol / np.sqrt(12)
    monthly_bond_return = bond_return / 12
    monthly_bond_vol = bond_vol / np.sqrt(12)
    
    # Generate all random numbers at once (much faster)
    z1_all = np.random.normal(0, 1, (num_simulations, months))
    z2_all = np.random.normal(0, 1, (num_simulations, months))
    
    # Calculate correlated returns for all simulations at once
    stock_returns = monthly_stock_return + monthly_stock_vol * z1_all
    bond_returns = monthly_bond_return + monthly_bond_vol * (
        correlation * z1_all + np.sqrt(1 - correlation**2) * z2_all
    )
    
    # Portfolio returns
    portfolio_returns = stock_allocation * stock_returns + bond_allocation * bond_returns
    
    # Calculate cumulative values
    results = np.zeros((num_simulations, months + 1))
    results[:, 0] = initial_investment
    
    for month in range(1, months + 1):
        results[:, month] = (results[:, month-1] + monthly_contribution) * (1 + portfolio_returns[:, month-1])
    
    return results

# ============================================================================
# HEADER
# ============================================================================

st.markdown('<div class="main-header">📊 Simulatore Monte Carlo Portafoglio</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Analizza diversi scenari di investimento - Per investitori europei</div>', unsafe_allow_html=True)

# ============================================================================
# SIDEBAR - PARAMETERS
# ============================================================================

st.sidebar.header("⚙️ Parametri di Simulazione")

# ETF Selection
st.sidebar.subheader("📈 Selezione ETF")

# European-focused ETFs
stock_etfs_popular = {
    'VWCE.DE': 'Vanguard FTSE All-World (EUR) - Accumulation',
    'SWDA.MI': 'iShares MSCI World (EUR) - Milano',
    'VWCE.MI': 'Vanguard FTSE All-World (EUR) - Milano',
    'EUNL.DE': 'iShares MSCI World (EUR) - Accumulation',
    'IWDA.AS': 'iShares MSCI World (EUR) - Amsterdam',
    'SPYL.MI': 'SPDR S&P 500 (EUR) - Milano',
    'VUAA.L': 'Vanguard S&P 500 (USD) - London',
    'VUSA.L': 'Vanguard S&P 500 (USD) - London',
}

bond_etfs_popular = {
    'AGGH.MI': 'iShares Global Aggregate Bond (EUR) - Milano',
    'VGEA.L': 'Vanguard Global Aggregate Bond (EUR) - London',
    'IEAG.L': 'iShares Global Aggregate Bond (EUR) - London',
    'VAGF.DE': 'Vanguard Global Aggregate Bond (EUR) - XETRA',
    'EUNA.MI': 'iShares Euro Aggregate Bond - Milano',
    'SEGA.MI': 'SPDR Bloomberg Euro Aggregate - Milano',
}

# Stock ETF selection with radio buttons
stock_mode = st.sidebar.radio(
    "Seleziona ETF Azionario",
    ["📋 Lista Europa", "🔍 Cerca ticker"],
    key="stock_mode",
    horizontal=True
)

if stock_mode == "📋 Lista Europa":
    stock_etf = st.sidebar.selectbox(
        "ETF Azionario",
        options=list(stock_etfs_popular.keys()),
        format_func=lambda x: f"{x} - {stock_etfs_popular[x].split(' - ')[0]}",
        index=0,
        key="stock_select"
    )
else:
    stock_etf_input = st.sidebar.text_input(
        "Ticker ETF Azionario",
        value="VWCE.DE",
        help="Es: VWCE.DE, SWDA.MI, IWDA.AS, EUNL.DE",
        placeholder="Inserisci ticker...",
        key="stock_input"
    )
    stock_etf = stock_etf_input.upper().strip()
    
    if stock_etf and len(stock_etf) > 0:
        # Validate ticker (only when button clicked or on change)
        if 'last_stock_check' not in st.session_state or st.session_state.last_stock_check != stock_etf:
            with st.spinner(f"🔍 Verifica {stock_etf}..."):
                test = get_etf_stats(stock_etf, period='1y')
                st.session_state.last_stock_check = stock_etf
                st.session_state.stock_valid = test is not None
                if test:
                    st.session_state.stock_name = test['name']
        
        if st.session_state.get('stock_valid', False):
            st.sidebar.success(f"✅ {stock_etf}: {st.session_state.get('stock_name', '')[:40]}")
        else:
            st.sidebar.error(f"❌ {stock_etf} non trovato")

st.sidebar.markdown("---")

# Bond ETF selection
bond_mode = st.sidebar.radio(
    "Seleziona ETF Obbligazionario",
    ["📋 Lista Europa", "🔍 Cerca ticker"],
    key="bond_mode",
    horizontal=True
)

if bond_mode == "📋 Lista Europa":
    bond_etf = st.sidebar.selectbox(
        "ETF Obbligazionario",
        options=list(bond_etfs_popular.keys()),
        format_func=lambda x: f"{x} - {bond_etfs_popular[x].split(' - ')[0]}",
        index=0,
        key="bond_select"
    )
else:
    bond_etf_input = st.sidebar.text_input(
        "Ticker ETF Obbligazionario",
        value="AGGH.MI",
        help="Es: AGGH.MI, VGEA.L, IEAG.L, VAGF.DE",
        placeholder="Inserisci ticker...",
        key="bond_input"
    )
    bond_etf = bond_etf_input.upper().strip()
    
    if bond_etf and len(bond_etf) > 0:
        if 'last_bond_check' not in st.session_state or st.session_state.last_bond_check != bond_etf:
            with st.spinner(f"🔍 Verifica {bond_etf}..."):
                test = get_etf_stats(bond_etf, period='1y')
                st.session_state.last_bond_check = bond_etf
                st.session_state.bond_valid = test is not None
                if test:
                    st.session_state.bond_name = test['name']
        
        if st.session_state.get('bond_valid', False):
            st.sidebar.success(f"✅ {bond_etf}: {st.session_state.get('bond_name', '')[:40]}")
        else:
            st.sidebar.error(f"❌ {bond_etf} non trovato")

use_historical = st.sidebar.checkbox("Usa dati storici reali", value=True, 
                                     help="Se disattivato, usa parametri standard di mercato")

st.sidebar.markdown("---")

# Investment Parameters
st.sidebar.subheader("💰 Parametri Investimento")

initial_investment = st.sidebar.number_input(
    "Investimento Iniziale (€)",
    min_value=0,
    max_value=1000000,
    value=10000,
    step=1000,
    format="%d"
)

monthly_contribution = st.sidebar.number_input(
    "Versamento Mensile (€)",
    min_value=0,
    max_value=10000,
    value=500,
    step=50,
    format="%d"
)

years = st.sidebar.slider(
    "Orizzonte Temporale (anni)",
    min_value=1,
    max_value=30,
    value=10,
    step=1
)

stock_allocation = st.sidebar.slider(
    "Allocazione Azioni (%)",
    min_value=0,
    max_value=100,
    value=80,
    step=5,
    help="Resto sarà allocato in obbligazioni"
)

st.sidebar.markdown("---")

# Simulation Parameters
st.sidebar.subheader("🎲 Parametri Simulazione")

num_simulations = st.sidebar.select_slider(
    "Numero di Simulazioni",
    options=[1000, 2500, 5000, 10000, 20000],
    value=5000,
    help="Più simulazioni = risultati più precisi (ma più lenti)"
)

st.sidebar.info(f"⏱️ Tempo stimato: ~{num_simulations/1000:.0f}-{num_simulations/500:.0f} secondi")

run_simulation = st.sidebar.button("▶️ Avvia Simulazione", type="primary", use_container_width=True)

# ============================================================================
# MAIN CONTENT
# ============================================================================

if run_simulation:
    
    # Validate tickers before running
    if stock_mode == "🔍 Cerca ticker" and not st.session_state.get('stock_valid', False):
        st.error("❌ Ticker azionario non valido. Verifica il ticker e riprova.")
        st.stop()
    
    if bond_mode == "🔍 Cerca ticker" and not st.session_state.get('bond_valid', False):
        st.error("❌ Ticker obbligazionario non valido. Verifica il ticker e riprova.")
        st.stop()
    
    start_time = time.time()
    
    # Get ETF data
    if use_historical:
        col1, col2 = st.columns(2)
        
        with col1:
            with st.spinner(f"📥 Caricamento dati {stock_etf}..."):
                stock_stats = get_etf_stats(stock_etf, period='max')
        
        with col2:
            with st.spinner(f"📥 Caricamento dati {bond_etf}..."):
                bond_stats = get_etf_stats(bond_etf, period='max')
        
        if stock_stats and bond_stats:
            stock_return = stock_stats['annual_return']
            stock_vol = stock_stats['annual_volatility']
            bond_return = bond_stats['annual_return']
            bond_vol = bond_stats['annual_volatility']
            correlation = get_correlation(stock_etf, bond_etf, period='max')
            
            # Display historical stats
            st.success("✅ Dati storici caricati con successo!")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    f"📈 {stock_etf}",
                    stock_stats['name'][:30],
                    help=f"Dati basati su {stock_stats['data_points']} mesi di storia"
                )
                st.metric("Rendimento Annuo", f"{stock_return:.2%}")
                st.metric("Volatilità", f"{stock_vol:.2%}")
            with col2:
                st.metric(
                    f"📉 {bond_etf}",
                    bond_stats['name'][:30],
                    help=f"Dati basati su {bond_stats['data_points']} mesi di storia"
                )
                st.metric("Rendimento Annuo", f"{bond_return:.2%}")
                st.metric("Volatilità", f"{bond_vol:.2%}")
            with col3:
                st.metric("🔗 Correlazione", f"{correlation:.3f}",
                          help="Correlazione tra i due asset (-1 a +1)")
                st.metric("📊 Dati Stock", f"{stock_stats['data_points']} mesi")
                st.metric("📊 Dati Bond", f"{bond_stats['data_points']} mesi")
            
        else:
            st.warning("⚠️ Impossibile caricare dati storici. Uso parametri standard.")
            use_historical = False
    
    if not use_historical:
        stock_return = 0.08
        stock_vol = 0.17
        bond_return = 0.03
        bond_vol = 0.05
        correlation = 0.2
        
        st.info("📊 Parametri Standard: Azioni 8%/17% vol, Obbligazioni 3%/5% vol")
    
    # Run simulation
    st.markdown("---")
    st.subheader(f"🎲 Esecuzione {num_simulations:,} Simulazioni")
    
    progress_bar = st.progress(0, text="Preparazione simulazione...")
    
    # Update progress
    progress_bar.progress(10, text="Generazione numeri casuali...")
    
    results = monte_carlo_dca(
        initial_investment=initial_investment,
        monthly_contribution=monthly_contribution,
        years=years,
        num_simulations=num_simulations,
        stock_allocation=stock_allocation / 100,
        stock_return=stock_return,
        stock_vol=stock_vol,
        bond_return=bond_return,
        bond_vol=bond_vol,
        correlation=correlation
    )
    
    progress_bar.progress(90, text="Calcolo statistiche...")
    
    # Calculate statistics
    months = years * 12
    final_values = results[:, -1]
    total_invested = initial_investment + (monthly_contribution * months)
    
    elapsed_time = time.time() - start_time
    progress_bar.progress(100, text=f"✅ Completato in {elapsed_time:.1f} secondi!")
    time.sleep(0.5)
    progress_bar.empty()
    
    # ====================================================================
    # RESULTS DISPLAY
    # ====================================================================
    
    st.markdown("---")
    st.header("📊 Risultati della Simulazione")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "💰 Investimento Totale",
            f"€{total_invested:,.0f}",
            help="Capitale investito nel periodo"
        )
    
    with col2:
        median_value = np.median(final_values)
        median_return = (median_value / total_invested - 1) * 100
        st.metric(
            "🎯 Scenario Mediano",
            f"€{median_value:,.0f}",
            f"{median_return:+.1f}%",
            help="50% probabilità di fare meglio/peggio"
        )
    
    with col3:
        p10_value = np.percentile(final_values, 10)
        p10_return = (p10_value / total_invested - 1) * 100
        st.metric(
            "😟 Scenario Pessimistico",
            f"€{p10_value:,.0f}",
            f"{p10_return:+.1f}%",
            delta_color="inverse",
            help="10% probabilità di fare peggio"
        )
    
    with col4:
        p90_value = np.percentile(final_values, 90)
        p90_return = (p90_value / total_invested - 1) * 100
        st.metric(
            "😊 Scenario Ottimistico",
            f"€{p90_value:,.0f}",
            f"{p90_return:+.1f}%",
            help="10% probabilità di fare meglio"
        )
    
    # ====================================================================
    # VISUALIZATIONS
    # ====================================================================
    
    st.markdown("---")
    
    # Chart 1 & 2: Time evolution and distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Evoluzione nel Tempo")
        
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        
        percentiles = [10, 25, 50, 75, 90]
        colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd']
        
        time_years = np.arange(months + 1) / 12
        
        for p, color in zip(percentiles, colors):
            values = np.percentile(results, p, axis=0)
            ax1.plot(time_years, values, label=f'{p}° percentile', 
                     color=color, linewidth=2.5, alpha=0.8)
        
        ax1.axhline(y=total_invested, color='black', linestyle='--', 
                    linewidth=2, alpha=0.6, label='Totale investito')
        
        ax1.set_xlabel('Anni', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Valore Portafoglio (€)', fontsize=12, fontweight='bold')
        ax1.legend(loc='upper left', fontsize=10)
        ax1.grid(alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'€{x/1000:.0f}K'))
        
        st.pyplot(fig1)
    
    with col2:
        st.subheader("📊 Distribuzione Risultati Finali")
        
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        
        ax2.hist(final_values, bins=60, alpha=0.7, color='steelblue', edgecolor='black')
        ax2.axvline(np.median(final_values), color='green', linestyle='-', 
                    linewidth=3, label=f'Mediana: €{np.median(final_values):,.0f}')
        ax2.axvline(total_invested, color='red', linestyle='--', 
                    linewidth=3, label=f'Investito: €{total_invested:,.0f}')
        
        ax2.set_xlabel('Valore Finale Portafoglio (€)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Frequenza', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(alpha=0.3, axis='y')
        ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'€{x/1000:.0f}K'))
        
        st.pyplot(fig2)
    
    # Chart 3: Probability of outcomes
    st.markdown("---")
    st.subheader("🎯 Probabilità di Raggiungere Obiettivi")
    
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    
    thresholds = [
        total_invested * 0.8,
        total_invested * 0.9,
        total_invested * 1.0,
        total_invested * 1.5,
        total_invested * 2.0,
        total_invested * 3.0
    ]
    
    threshold_labels = [
        'Evitare perdita >20%',
        'Evitare perdita >10%',
        'Break-even (pareggio)',
        'Guadagno +50%',
        'Raddoppio (+100%)',
        'Triplicare (+200%)'
    ]
    
    probabilities = [(final_values >= t).mean() * 100 for t in thresholds]
    colors_prob = ['#d62728', '#ff7f0e', '#ffdd57', '#90ee90', '#2ca02c', '#1f77b4']
    
    bars = ax3.barh(threshold_labels, probabilities, color=colors_prob, edgecolor='black', linewidth=1.5)
    ax3.set_xlabel('Probabilità (%)', fontsize=12, fontweight='bold')
    ax3.set_xlim(0, 105)
    ax3.grid(alpha=0.3, axis='x')
    
    for bar, prob in zip(bars, probabilities):
        width = bar.get_width()
        ax3.text(width + 1, bar.get_y() + bar.get_height()/2, 
                 f'{prob:.1f}%', ha='left', va='center', fontweight='bold', fontsize=11)
    
    st.pyplot(fig3)
    
    # ====================================================================
    # DETAILED STATISTICS
    # ====================================================================
    
    st.markdown("---")
    st.subheader("📋 Statistiche Dettagliate")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 💰 Riepilogo Investimento")
        st.markdown(f"""
        - **Investimento iniziale:** €{initial_investment:,.0f}
        - **Versamento mensile:** €{monthly_contribution:,.0f}
        - **Periodo:** {years} anni ({months} mesi)
        - **Totale investito:** €{total_invested:,.0f}
        - **Allocazione:** {stock_allocation}% azioni / {100-stock_allocation}% obbligazioni
        """)
        
        st.markdown("### 📊 Parametri di Mercato Utilizzati")
        st.markdown(f"""
        - **Azioni ({stock_etf}):**
          - Rendimento atteso: {stock_return:.2%}
          - Volatilità: {stock_vol:.2%}
        - **Obbligazioni ({bond_etf}):**
          - Rendimento atteso: {bond_return:.2%}
          - Volatilità: {bond_vol:.2%}
        - **Correlazione:** {correlation:.3f}
        """)
    
    with col2:
        st.markdown("### 📈 Range di Risultati")
        percentiles_detail = [5, 10, 25, 50, 75, 90, 95]
        
        data = []
        for p in percentiles_detail:
            value = np.percentile(final_values, p)
            return_pct = (value / total_invested - 1) * 100
            data.append({
                'Percentile': f'{p}°',
                'Valore Finale': f'€{value:,.0f}',
                'Rendimento': f'{return_pct:+.1f}%'
            })
        
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        st.markdown("### ⚠️ Probabilità di Perdita")
        loss_prob = (final_values < total_invested).mean() * 100
        loss_10_prob = (final_values < total_invested * 0.9).mean() * 100
        loss_20_prob = (final_values < total_invested * 0.8).mean() * 100
        
        st.markdown(f"""
        - **Qualsiasi perdita:** {loss_prob:.2f}%
        - **Perdita > 10%:** {loss_10_prob:.2f}%
        - **Perdita > 20%:** {loss_20_prob:.2f}%
        """)
    
    # ====================================================================
    # FOOTER NOTES
    # ====================================================================
    
    st.markdown("---")
    st.info(f"""
    **⚠️ Note Importanti:**
    - Simulazione completata in {elapsed_time:.1f} secondi con {num_simulations:,} scenari
    - Basata su dati storici reali (quando disponibili) e modelli probabilistici
    - I rendimenti passati non garantiscono risultati futuri
    - Non considera tasse, inflazione o costi di gestione (TER degli ETF)
    - Consulta sempre un consulente finanziario per decisioni di investimento
    """)

else:
    # Welcome message
    st.info("👈 Configura i parametri nella barra laterale e clicca su '▶️ Avvia Simulazione' per iniziare")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 📖 Come Funziona
        
        Questo simulatore utilizza il **metodo Monte Carlo** per generare migliaia di scenari possibili per il tuo portafoglio.
        
        #### Caratteristiche:
        - ✅ **ETF Europei** - Focus su ETF quotati in EUR
        - ✅ **Dati Storici Reali** - Da Yahoo Finance
        - ✅ **DCA (Dollar Cost Averaging)** - Versamenti mensili
        - ✅ **Performance** - Gestisce 20.000 simulazioni
        
        #### Perché è utile:
        - Ti mostra **l'intera distribuzione** dei possibili risultati
        - Evidenzia i **rischi** e le **opportunità**
        - Aiuta a capire se l'allocazione è adatta alla tua **tolleranza al rischio**
        """)
    
    with col2:
        st.markdown("""
        ### 🇪🇺 ETF Europei Disponibili
        
        **Azionari Globali:**
        - `VWCE.DE` - Vanguard FTSE All-World (più popolare)
        - `SWDA.MI` - iShares MSCI World (Milano)
        - `EUNL.DE` - iShares MSCI World Acc
        - `IWDA.AS` - iShares MSCI World (Amsterdam)
        
        **Obbligazionari Globali:**
        - `AGGH.MI` - iShares Global Aggregate (Milano)
        - `VGEA.L` - Vanguard Global Aggregate (London)
        - `VAGF.DE` - Vanguard Global Aggregate (XETRA)
		
		**Azionari S&P 500 (per confronto):**
		- `SPYL.MI` - SPDR S&P 500 (Milano, EUR)
		- `VUAA.L` - Vanguard S&P 500 (London, Acc, USD)
		""")

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; font-size: 0.9rem;'>
© 2024 | Creato con Streamlit | Dati da Yahoo Finance.<br>
<strong>Disclaimer:</strong> Questo strumento è solo a scopo educativo e non costituisce consulenza finanziaria.
</div>
""", unsafe_allow_html=True)