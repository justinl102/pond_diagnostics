import streamlit as st
import pandas as pd 
import numpy as np
if 'language' not in st.session_state:
    st.session_state['language'] = 'ESP'

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.write("Active Cycles:")

option = st.sidebar.selectbox(
    'Language',
    ("ENG", "ESP"),
    key = "language"
)

active_cycles = pd.read_csv('active_cycles.csv')

active_cycles = active_cycles[active_cycles['PesoPromedio2'] >= 2]
active_cycles.sort_values('PesoPromedio2', inplace = True)
active_cycles[['pondName', 'PesoPromedio2', 'cycle_days', 'density_ha']]