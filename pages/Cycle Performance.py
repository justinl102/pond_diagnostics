import plotly.figure_factory as ff
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, least_squares
import seaborn as sns
from datetime import datetime
sns.set_style('whitegrid')

@st.cache_data
def get_dataframe(file_name):
    return pd.read_csv(file_name)

cycles = get_dataframe('cycles_cleaned.csv')
cycles = cycles[cycles['PesoPromedio2'] >19]

top10 = (cycles['MnProveedor'].value_counts()[cycles['MnProveedor'].value_counts()> 10]).index
cycles['pct_animals_harvested'] = cycles['partial_harvest_qty'] / cycles['CantidadCosechada_harvested']
print(cycles.columns)
cycles['MnProveedor_category'] = np.where(cycles['MnProveedor'].isin(top10), cycles['MnProveedor'],'Other')
#cycles['MnProveedor_category'] = cycles.loc[cycles['MnProveedor'].isin((cycles['MnProveedor'].value_counts()[cycles['MnProveedor'].value_counts() < 10]).index), 'MnProveedor'] = 'other'
cycles['qty_harvested_ha'] = round(cycles['CantidadCosechada_harvested'] / cycles['Hectareas'])

cycles['avg_price_kg'] = round(cycles['VentaUSDReal'] / cycles['final_biomass_harvested'],3)

cycles['FechaSiembra'] = pd.to_datetime(cycles['FechaSiembra'])


max_date = cycles['FechaSiembra'].max().date()
min_date = cycles['FechaSiembra'].min().date()

#profit calculation 

labels_dict = {
    'avg_price_kg': 'Survival Rate',
    'rev_ha_day':'Revenue/Ha/Day',
    'cycle_feed_ha_day': 'Feed/Ha/Day',
    'cycle_weekly_growth_rate':'Avg. Weekly Growth Rate',
    'MnProveedor_category':'Supplier',
    'cycle_fcr':'FCR',
    'avg_price_kg': 'Avg. Price KG',
    'density_ha':'Density',
    'PesoPromedio2':'Average Weight',
    'Supervivencia': 'Survival Rate',
    'cycle_days': 'Cycle Days',
    'cycle_profit_ha_day': 'Profit/Ha/Day',
    'cycle_total_profit_usd':'Total Profit',
    'pct_animals_harvested': 'Percent of animals partially harvested',
    'qty_harvested_ha':'Animals Harvested/Ha'

}

category_dict = {
    'MnProveedor_category':'Supplier',
    'density_ha':'Density',
    'Supervivencia': 'Survival Rate',
}

def bin_continuous(series, n_bins):
    return pd.qcut(series, n_bins, precision = 0)

labels_reverse_dict = dict((v,k) for k,v in labels_dict.items())
category_reverse_dict = dict((v,k) for k,v in category_dict.items())
 
x_var1 = st.sidebar.selectbox(
    "X Variable",
    ['Average Weight', 'Cycle Days', 'Density', "FCR",'Animals Harvested/Ha'],
    placeholder="Metric #1",
    )
objective_var2 = st.sidebar.selectbox(
    "Y Variable",
    ['FCR', 'Avg. Weekly Growth Rate','Revenue/Ha/Day','Avg. Price KG','Profit/Ha/Day','Total Profit', 'Survival Rate','Animals Harvested/Ha'],
    placeholder="Objective",
    )
bin_str = st.sidebar.selectbox(
    "Grouping Variable",
    ['Density','Supplier'],
    index = None,
    placeholder="Select Variable",
    )

start_time, end_time = st.sidebar.slider(
        "Stocking Date",
        value=[min_date,max_date],
        format="MM/DD/YY")
fixed_cost_day = st.sidebar.number_input('Fixed Cost/Ha/Day')


cycles['cycle_fixed_cost'] = cycles['cycle_days'] * fixed_cost_day * cycles['Hectareas']

cycles['cycle_total_profit_usd'] = cycles['VentaUSDReal'] - cycles['cycle_fixed_cost'] - cycles['cycle_feed_cost']

cycles['cycle_profit_ha_day'] = cycles['cycle_total_profit_usd'] / cycles['Hectareas']/ cycles['cycle_days']

show_trendlines_only = st.sidebar.toggle('Show Trendline Only', value = False)
remove_outliers = st.sidebar.toggle('Remove Outliers', value = False)

def get_outlier_threshold(cycle_df, y_variable, magnitude):

    data = cycle_df[y_variable].values

    if len(data) < 3:
        return cycle_df # No outliers to remove for small datasets

    median = np.median(data)
    deviation = np.abs(data - median)
    median_absolute_deviation = np.median(deviation)
    threshold = magnitude * median_absolute_deviation

    is_outlier = np.abs(data - median) > threshold
    filtered_df = cycle_df[~is_outlier]

    return filtered_df


if bin_str:
    bin_str_original = category_reverse_dict[bin_str]
    bin_str_binned = bin_str_original + '_bin'
else: 
    bin_str = None


cycles['FechaMuestreo'] = pd.to_datetime(cycles['FechaMuestreo'])
cycles['FechaSiembra'] = pd.to_datetime(cycles['FechaSiembra'])
cycles['FechaCosecha'] = pd.to_datetime(cycles['FechaCosecha'])
    
cycles['avg_price_kg'] = round(cycles['VentaUSDReal'] / cycles['final_biomass_harvested'],3)

cycles_df = cycles[(cycles['FechaSiembra'].dt.date >= start_time)
                   & (cycles['FechaSiembra'].dt.date <= end_time)
                   ]
if bin_str:
    if bin_str in ['Density',
                'Survival Rate']:
        cycles_df[bin_str_binned] = bin_continuous(cycles_df[bin_str_original],3)
    else:
        cycles_df[bin_str_binned] = cycles_df[bin_str_original]

x_variable = labels_reverse_dict[x_var1]
objective_variable_str = labels_reverse_dict[objective_var2]
if remove_outliers:
    plot_df = get_outlier_threshold(cycles_df,objective_variable_str,8 )
    plot_df = get_outlier_threshold(plot_df,x_variable,8 )
else: 
    plot_df = cycles_df

if bin_str:

    fig = px.scatter(plot_df, 
                        x=x_variable, 
                        y=objective_variable_str, 
                        color=bin_str_binned,
                    hover_data=['IDPiscina'],
                    trendline="ols"
                    )
else:
    fig = px.scatter(plot_df, 
                        x=x_variable, 
                        y=objective_variable_str, 
                        color=None,
                    hover_data=['IDPiscina'],
                    trendline="ols"
                    
                    )
fig.update_xaxes(title_text=x_var1)
fig.update_yaxes(title_text=objective_var2)
if show_trendlines_only:
    fig.data = [t for t in fig.data if t.mode == "lines"]
    fig.update_traces(showlegend=True)
st.plotly_chart(fig, use_container_width=True)
if bin_str:
    table_df = plot_df.groupby(bin_str_binned)[objective_variable_str].describe().round(2)
    table_df.sort_values('50%',inplace = True )
else:
    table_df = plot_df[objective_variable_str].describe().T.round(2)

table_df


    