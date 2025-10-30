"""
Monte Carlo Portfolio Simulator - Streamlit App
Simulatore Monte Carlo per Portafoglio

Deploy to Streamlit Cloud:
1. Create GitHub repo with this file as app.py
2. Add requirements.txt with: streamlit, yfinance, numpy, pandas, matplotlib
3. Go to share.streamlit.io and deploy
"""

import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

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
    .metric-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# FUNCTIONS
# ============================================================================

@st.cache_data(ttl=3600)
def get_etf_stats(ticker, period='10y'):
    """Get historical ETF statistics"""
    try:
        etf = yf.Ticker(ticker)
        hist = etf.history(period=period)
        
        if len(hist) < 20:
            return None
        
        monthly_returns = hist['Close'].resample('M').last().pct_change().dropna()
        annual_return = (1 + monthly_returns.mean()) ** 12 - 1
        annual_vol = monthly_returns.std() * np.sqrt(12)
        
        return {
            'ticker': ticker,
            'annual_return': annual_return,
            'annual_volatility': annual_vol,
            'data': monthly_returns
        }
    except Exception as e:
        st.warning(f"Impossibile caricare {ticker}: {str(e)}")
        return None

@st.cache_data(ttl=3600)
def get_correlation(ticker1, ticker2, period='10y'):
    """Calculate correlation between two ETFs"""
    try:
        data1 = yf.Ticker(ticker1).history(period=period)['Close'].resample('M').last().pct_change()
        data2 = yf.Ticker(ticker2).history(period=period)['Close'].resample('M').last().pct_change()
        
        combined = pd.concat([data1, data2], axis=1).dropna()
        if len(combined) < 20:
            return 0.2
        return combined.corr().iloc[0, 1]
    except:
        return 0.2

@st.cache_data
def monte_carlo_dca(initial_investment, monthly_contribution, years, num_simulations,
                    stock_allocation, stock_return, stock_vol,
                    bond_return, bond_vol, correlation=0.2, _progress_bar=None):
    """Run Monte Carlo simulation with DCA"""
    
    months = years * 12
    bond_allocation = 1 - stock_allocation
    results = np.zeros((num_simulations, months + 1))
    results[:, 0] = initial_investment
    
    for sim in range(num_simulations):
        portfolio_value = initial_investment
        
        for month in range(1, months + 1):
            portfolio_value += monthly_contribution
            
            z1 = np.random.normal(0, 1)
            z2 = np.random.normal(0, 1)
            
            monthly_stock_return = stock_return / 12
            monthly_stock_vol = stock_vol / np.sqrt(12)
            monthly_bond_return = bond_return / 12
            monthly_bond_vol = bond_vol / np.sqrt(12)
            
            stock_rand_return = monthly_stock_return + monthly_stock_vol * z1
            bond_rand_return = monthly_bond_return + monthly_bond_vol * (
                correlation * z1 + np.sqrt(1 - correlation**2) * z2)
            
            portfolio_return = (stock_allocation * stock_rand_return + 
                               bond_allocation * bond_rand_return)
            
            portfolio_value *= (1 + portfolio_return)
            results[sim, month] = portfolio_value
        
        if _progress_bar and sim % 100 == 0:
            _progress_bar.progress((sim + 1) / num_simulations)
    
    return results

# ============================================================================
# HEADER
# ============================================================================

st.markdown('<div class="main-header">📊 Simulatore Monte Carlo Portafoglio</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Analizza diversi scenari di investimento con dati storici reali</div>', unsafe_allow_html=True)

# ============================================================================
# SIDEBAR - PARAMETERS
# ============================================================================

st.sidebar.header("⚙️ Parametri di Simulazione")

# ETF Selection
st.sidebar.subheader("📈 Selezione ETF")

stock_etfs = {
    'VT': 'Vanguard Total World Stock',
    'VWCE.DE': 'Vanguard FTSE All-World (EUR)',
    'SWDA.MI': 'iShares MSCI World (Milano)',
    'ACWI': 'iShares MSCI ACWI'
}

bond_etfs = {
    'BNDW': 'Vanguard Total World Bond',
    'AGG': 'iShares Core US Aggregate',
    'AGGH.MI': 'iShares Global Aggregate (Milano)',
    'VWRL.L': 'Vanguard Global Bond Index'
}

stock_etf = st.sidebar.selectbox(
    "ETF Azionario",
    options=list(stock_etfs.keys()),
    format_func=lambda x: f"{x} - {stock_etfs[x]}",
    index=0
)

bond_etf = st.sidebar.selectbox(
    "ETF Obbligazionario",
    options=list(bond_etfs.keys()),
    format_func=lambda x: f"{x} - {bond_etfs[x]}",
    index=0
)

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
    options=[1000, 2500, 5000, 7500, 10000],
    value=5000,
    help="Più simulazioni = risultati più precisi (ma più lenti)"
)

run_simulation = st.sidebar.button("▶️ Avvia Simulazione", type="primary", use_container_width=True)

# ============================================================================
# MAIN CONTENT
# ============================================================================

if run_simulation:
    with st.spinner("🔄 Caricamento dati e avvio simulazione..."):
        
        # Get ETF data
        if use_historical:
            col1, col2 = st.columns(2)
            
            with col1:
                with st.spinner(f"Caricamento {stock_etf}..."):
                    stock_stats = get_etf_stats(stock_etf)
            
            with col2:
                with st.spinner(f"Caricamento {bond_etf}..."):
                    bond_stats = get_etf_stats(bond_etf)
            
            if stock_stats and bond_stats:
                stock_return = stock_stats['annual_return']
                stock_vol = stock_stats['annual_volatility']
                bond_return = bond_stats['annual_return']
                bond_vol = bond_stats['annual_volatility']
                correlation = get_correlation(stock_etf, bond_etf)
                
                # Display historical stats
                st.success("✅ Dati storici caricati con successo!")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(f"📈 {stock_etf} - Rendimento", f"{stock_return:.2%}")
                    st.metric(f"📊 {stock_etf} - Volatilità", f"{stock_vol:.2%}")
                with col2:
                    st.metric(f"📉 {bond_etf} - Rendimento", f"{bond_return:.2%}")
                    st.metric(f"📊 {bond_etf} - Volatilità", f"{bond_vol:.2%}")
                with col3:
                    st.metric("🔗 Correlazione", f"{correlation:.3f}")
                
            else:
                st.warning("⚠️ Impossibile caricare dati storici. Uso parametri standard.")
                use_historical = False
        
        if not use_historical:
            stock_return = 0.10
            stock_vol = 0.18
            bond_return = 0.04
            bond_vol = 0.06
            correlation = 0.2
            
            st.info("📊 Parametri Standard: Azioni 10%/18% vol, Obbligazioni 4%/6% vol")
        
        # Run simulation
        st.markdown("---")
        st.subheader(f"🎲 Esecuzione {num_simulations:,} Simulazioni")
        
        progress_bar = st.progress(0)
        
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
            correlation=correlation,
            _progress_bar=progress_bar
        )
        
        progress_bar.empty()
        
        # Calculate statistics
        months = years * 12
        final_values = results[:, -1]
        total_invested = initial_investment + (monthly_contribution * months)
        
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
            - **Azioni - Rendimento atteso:** {stock_return:.2%}
            - **Azioni - Volatilità:** {stock_vol:.2%}
            - **Obbligazioni - Rendimento atteso:** {bond_return:.2%}
            - **Obbligazioni - Volatilità:** {bond_vol:.2%}
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
        st.info("""
        **⚠️ Note Importanti:**
        - Queste simulazioni sono basate su modelli probabilistici e dati storici
        - I rendimenti passati non garantiscono risultati futuri
        - Non considerano tasse, inflazione o costi di gestione
        - Consulta sempre un consulente finanziario per decisioni di investimento
        """)

else:
    # Welcome message when no simulation has been run
    st.info("👈 Configura i parametri nella barra laterale e clicca su '▶️ Avvia Simulazione' per iniziare")
    
    st.markdown("---")
    st.markdown("""
    ### 📖 Come Funziona
    
    Questo simulatore utilizza il **metodo Monte Carlo** per generare migliaia di scenari possibili per il tuo portafoglio.
    
    #### Cosa puoi fare:
    1. **Seleziona gli ETF** che vuoi analizzare (azioni e obbligazioni)
    2. **Configura i parametri** di investimento (importo iniziale, versamenti mensili, durata)
    3. **Imposta l'allocazione** tra azioni e obbligazioni
    4. **Avvia la simulazione** per vedere i possibili risultati
    
    #### Perché è utile:
    - Ti mostra **l'intera distribuzione** dei possibili risultati, non solo la media
    - Evidenzia i **rischi** e le **opportunità** della tua strategia
    - Usa **dati storici reali** degli ETF che scegli
    - Ti aiuta a capire se la tua allocazione è adatta alla tua **tolleranza al rischio**
    
    #### Interpretazione dei Risultati:
    - **Scenario Mediano (50° percentile):** Il risultato "tipico" - metà delle simulazioni fa meglio, metà peggio
    - **Scenario Pessimistico (10° percentile):** C'è solo il 10% di probabilità di fare peggio di questo
    - **Scenario Ottimistico (90° percentile):** C'è solo il 10% di probabilità di fare meglio di questo
    """)
    
    st.markdown("---")
    st.markdown("""
    ### 🎯 ETF Disponibili
    
    **Azionari:**
    - VT: Vanguard Total World Stock (tutto il mondo)
    - VWCE.DE: Vanguard FTSE All-World quotato in Euro
    - SWDA.MI: iShares MSCI World quotato a Milano
    - ACWI: iShares MSCI ACWI
    
    **Obbligazionari:**
    - BNDW: Vanguard Total World Bond
    - AGG: iShares Core US Aggregate Bond
    - AGGH.MI: iShares Global Aggregate Bond quotato a Milano
    - VWRL.L: Vanguard Global Bond Index
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem 0;'>
    <p>📊 Simulatore Monte Carlo Portafoglio | Creato per analisi di investimento a lungo termine</p>
    <p style='font-size: 0.8rem;'>⚠️ Solo a scopo educativo - Non costituisce consulenza finanziaria</p>
</div>
""", unsafe_allow_html=True)