"""
# My first app
Here's our first attempt at using data to create a table:
"""
import plotly.figure_factory as ff
import streamlit as st
import numpy as np
import math
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, least_squares
import seaborn as sns
from datetime import datetime

import plotly.graph_objects as go
from plotly.subplots import make_subplots
sns.set_style('whitegrid')

@st.cache_data
def get_dataframe(file_name):
    return pd.read_csv(file_name)

monitorings = get_dataframe('monitoring_cleaned.csv')
harvests = get_dataframe('bravito_harvests.csv')
active_cycles = get_dataframe('active_cycles.csv')


monitorings['FechaMuestreo'] = pd.to_datetime(monitorings['FechaMuestreo']).dt.date

monitorings['feed_percent_biomass'] =   (monitorings['KilosAlimento']/7) / monitorings['live_biomass']

max_date = monitorings['FechaMuestreo'].max()
min_date = monitorings['FechaMuestreo'].min()


def get_decreasing(df):
    data = (
        df.groupby([
            'PKCiclo', pd.Grouper(
                key='FechaMuestreo', freq='1W',
            ),
        ])
        .PesoPromedio2.median()
        .reset_index()
    )

    window_diff = (
        data.groupby('PKCiclo')
        .rolling(window=2)
        .PesoPromedio2.apply(np.diff)
        .dropna()
        .reset_index()
        .drop(columns=['level_1'])
    )
    return window_diff

def clip_extreme_change(df, mini, maxi):
    d = get_decreasing(df)
    normal_population = d[
        (d.PesoPromedio2 < mini) | (d.PesoPromedio2 > maxi)
    ].PKCiclo.unique()
    return df.loc[~df.PKCiclo.isin(normal_population)].copy()

def dropna(df, subset=['PesoPromedio2', 'cycle_days']):
  return df.dropna(subset = subset)

def clip_extreme_growth(df, mini, maxi):
  df['daily_growth'] = df['PesoPromedio2'] / df['cycle_days']
  growth_median = df['daily_growth'].median()
  mini_growth = growth_median * mini
  maxi_growth = growth_median * maxi
  return df[(df['daily_growth']>=mini_growth) & (df['daily_growth']<=maxi_growth)]

def print_shape(df):
  print(df.shape)
  return df


def growth_model(time, Linf, K, position, w0):
    return Linf * np.exp(-np.exp(position - K * time)) + w0





def exponential_fit_3d(x, a,b,c,d):
  return a*x**3 + b*x**2 + c*x + d


def exponential_fit_2d(x, b,c,d):
  return b*x**2 + c*x + d
def exponential_decay(x, a, b):
    return a * np.exp(b * x)

#dictionaries -------------------------------------------------------------
param_dict = {
    'Supervivencia': {'lower_factor':0.7,'upper_factor': 1.5, 'min_value': 40, 'max_value': 100},
    'biomass_ha': {'lower_factor':0.7,'upper_factor': 1.5, 'min_value': 0, 'max_value': 10000},
    'PesoPromedio2':{'lower_factor':0.5,'upper_factor': 1.5, 'min_value': 0, 'max_value': 50},
    'cumulative_fcr':{'lower_factor':0.5,'upper_factor': 1.5, 'min_value': 0, 'max_value': 6},
    'weekly_fcr':{'lower_factor':0.5,'upper_factor': 1.5, 'min_value': 0, 'max_value': 6},
    '1week_growth_rate':{'lower_factor':0.5,'upper_factor': 2, 'min_value': -0.5, 'max_value': 5},
    '2week_growth_rate':{'lower_factor':0.5,'upper_factor': 2, 'min_value': -0.5, 'max_value': 5},
    'kg/ha/day':{'lower_factor':0.5,'upper_factor': 2, 'min_value': 0, 'max_value': 250},
    'feed_percent_biomass':{'lower_factor':0.2,'upper_factor': 3, 'min_value': 0, 'max_value': 1},
    'mlResultWeightCv':{'lower_factor':0.25,'upper_factor': 3, 'min_value': 0, 'max_value': 1},
}
model_dict = {
    'Supervivencia': {"model":exponential_decay,"p0":[99.17700584718783,-0.005269]},
    'biomass_ha': {"model": exponential_fit_3d,"p0":None},
    'PesoPromedio2':{"model": exponential_fit_3d,"p0":None},
    'cumulative_fcr':{"model": exponential_fit_3d,"p0":None},
    'weekly_fcr':{"model": exponential_fit_3d,"p0":None},
    '1week_growth_rate':{"model": exponential_fit_3d,"p0":None},
    '2week_growth_rate':{"model": exponential_fit_3d,"p0":None},
    'kg/ha/day':{"model": exponential_fit_3d,"p0":None},
    'feed_percent_biomass':{"model": exponential_fit_3d,"p0":None},
    'mlResultWeightCv':{"model": exponential_fit_3d,"p0":None},
}

labels_dict = {
    'Supervivencia': 'Survival Rate',
    'biomass_ha': 'Biomass/Ha',
    'PesoPromedio2':'Average Weight',
    'cumulative_fcr':'Cumulative FCR',
    'weekly_fcr':'Weekly FCR',
    '1week_growth_rate':'Growth Rate - 1 Week',
    '2week_growth_rate':'Growth Rate - 2 Week',
    'kg/ha/day': 'KG/Ha/Day',
    'feed_percent_biomass': "Feed - % of biomass",
    'mlResultWeightCv':"CV"
}
labels_reverse_dict = dict((v,k) for k,v in labels_dict.items())
active_cycles = pd.read_csv('active_cycles.csv')

active_cycles.sort_values('pondName', inplace = True)
pond_cycle_dict = active_cycles.set_index('pondName')['PKCiclo'].to_dict()


def remove_outliers(df, x_column, y_column,  upper_factor=4, lower_factor = 0.25, num_intervals=5):
    # Sort the DataFrame by the time_column
    df = df.sort_values(by=x_column)
    # Generate time intervals using qcut
    df['time_intervals'] = pd.qcut(df[x_column], num_intervals, labels=False)
    df_list = []
    # Iterate through each time interval
    for interval_num in range(num_intervals):
        interval_df = df[df['time_intervals'] == interval_num]
        interval_median = interval_df[y_column].median()
        print(interval_median)
        upper_threshold = upper_factor * interval_median
        lower_threshold = lower_factor * interval_median
        interval_df = interval_df[(interval_df[y_column]< upper_threshold) & (interval_df[y_column]> lower_threshold) ]
        df_list.append(interval_df)
    training_df = pd.concat(df_list)
    return training_df

def set_min_max(df, y_column,  min_value, max_value):
    # Sort the DataFrame by the time_column
    return df.loc[(df[y_column]> min_value) & (df[y_column]< max_value)]

def remove_na(df,columns):
    # Sort the DataFrame by the time_column
    return df.dropna(subset = columns).copy()





def clean_df(df, x_variable, y_variable, start_date,end_date):
  
  y_lower_threshold = param_dict[y_variable]['lower_factor']
  y_upper_threshold = param_dict[y_variable]['upper_factor']

  y_min = param_dict[y_variable]['min_value']
  y_max = param_dict[y_variable]['max_value']


  new_df = (df.pipe(remove_na,[x_variable, y_variable])
  .pipe(set_min_max,y_variable, y_min, y_max)
  .pipe(set_min_max, x_variable, 0,120)
  .pipe(set_min_max, 'FechaMuestreo', start_date,end_date)
  .pipe(remove_outliers, x_variable, y_variable, upper_factor=y_upper_threshold, lower_factor = y_lower_threshold)
 )
  return new_df



def get_curve_params(df,y_variable,  x_variable = 'cycle_days'):
  x_train = df[x_variable]
 
  y_train = df[y_variable]
  
  model = model_dict[y_variable]['model']
  p0 = model_dict[y_variable]['p0']
  curve_params, covariance = curve_fit(model,
                                     x_train,
                                     y_train,
                                     p0 = p0,
                                     maxfev = 100000
                                          )
  return curve_params 


def plot_benchmark(curve_params, model, x_min, x_max, increment, x_label, y_label):
  x_plot = np.arange(x_min, x_max, increment)

  y_plot = np.round(model(x_plot, *curve_params),3)

  return pd.DataFrame({x_label:x_plot,
                       y_label:y_plot
                       })

def get_variable_df(df, y_variable, start_time, end_time):
    cleaned_df = clean_df(df, 'cycle_days', y_variable,start_time, end_time)

    model = model_dict[y_variable]['model']

    
    curve_params = get_curve_params(cleaned_df, y_variable,)
    plot_df = plot_benchmark(curve_params,model, 5, 91, 1, 'Cycle_Day', y_variable)


    return plot_df 

def get_title(cycle_id, active_cycles):
    current_cycle_density = active_cycles.loc[active_cycles['PKCiclo'] == cycle_id, 'density_ha'].iloc[0]
    current_cycle_ha = active_cycles.loc[active_cycles['PKCiclo'] == cycle_id, 'Hectareas'].iloc[0]
    title = str(sidebar_cycle) + " - " + str(round(current_cycle_ha,2))+ " Ha" + " - " + str(current_cycle_density)+ " animals/ha" 
    return title#models ---------------------------------------------------
def growth_model(time, Linf, K, position, w0):
    return Linf * np.exp(-np.exp(position - K * time)) + w0



#sidebar ---------------------------------------------------
cycle_options = pond_cycle_dict.keys() 

sidebar_var1 = st.sidebar.selectbox(
    "Metric #1",
    list(labels_reverse_dict.keys()),
 
    placeholder="Metric #1",
    )

sidebar_var2 = st.sidebar.selectbox(
    "Metric #2",
    list(labels_reverse_dict.keys()),

    placeholder="Metric #2",
    )


sidebar_cycle = st.sidebar.selectbox(
    "Pond",
    cycle_options,
    index=None,
    placeholder="Select Pond",
    )

start_time, end_time = st.sidebar.slider(
        "Benchmark Window",
        value=[min_date,max_date],
        format="MM/DD/YY")

show_benchmarks = st.sidebar.toggle('Show Benchmarks', value = True)

second_graph = st.sidebar.toggle('Show Second Graph')
show_raleos = st.sidebar.toggle('Show Raleos', value = False)

if second_graph:
    sidebar_var3 = st.sidebar.selectbox(
        "Metric #3",
        list(labels_reverse_dict.keys()),
        placeholder="Metric #3",
        )

    sidebar_var4 = st.sidebar.selectbox(
        "Metric #4",
        list(labels_reverse_dict.keys()),
        placeholder="Metric #4",
        )


def generate_graph(y_variable_label1, y_variable_label2, show_benchmarks, show_cycles, harvests, start_time, end_time, active_cycles, colors = ["#83c9ff","#0068c9"], show_title = False):
    y_variable1 = labels_reverse_dict[y_variable_label1]
    y_variable2 = labels_reverse_dict[y_variable_label2]

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    title = ""

    if show_cycles: 
        cycle_id = int(pond_cycle_dict[sidebar_cycle])
        
    
        plot_current_cycle = monitorings.loc[monitorings['PKCiclo'] == cycle_id, ['cycle_days', y_variable1]]


        plot_current_cycle2 = monitorings.loc[monitorings['PKCiclo'] == cycle_id, ['cycle_days', y_variable2]]
        if show_title:
            title = get_title(cycle_id, active_cycles)
        cycle_raleos =  harvests.loc[
                        (harvests['Parcial'] == 1) & 
                        (harvests['PKCiclo'] == cycle_id)]



    # Set x-axis title
        if show_raleos & len(cycle_raleos)>0:
            for i in cycle_raleos['cycle_days']:
                fig.add_vline(x =i, line_width = 2, line_dash = "dash", line_color = 'red', annotation_text= 'Raleo',)
        fig.add_trace(
            go.Scatter(x=plot_current_cycle['cycle_days'], 
                    y=plot_current_cycle[y_variable1], 
                    name= "Current Cycle " + labels_dict[y_variable1],
                    line=dict(color=colors[0])
                    ),
            secondary_y=False,
            
        )
        fig.add_trace(
            go.Scatter(x=plot_current_cycle2['cycle_days'], 
                    y=plot_current_cycle2[y_variable2], 
                    name= "Current Cycle " + labels_dict[y_variable2],
                    line=dict(color=colors[1])
                    
                    
                    ),
            secondary_y=True,
            
        )

    if show_benchmarks:
        variable1_df = get_variable_df(monitorings, y_variable1, start_time, end_time)
        variable2_df = get_variable_df(monitorings, y_variable2,  start_time, end_time)
        fig.add_trace(
            go.Scatter(x=variable1_df['Cycle_Day'], 
                    y=variable1_df[y_variable1], 
                    name=labels_dict[y_variable1],
                    line=dict(color=colors[0], dash="dash")),
            secondary_y=False,
            
        )
        fig.add_trace(
            go.Scatter(x=variable2_df['Cycle_Day'], 
                    y=variable2_df[y_variable2], 
                    name=labels_dict[y_variable2],
                    line=dict(color=colors[1],dash="dash")
                    ),
            secondary_y=True,
        )

        
    
    fig.update_layout(
    title_text=title,
    yaxis2=dict(
        side="right",
        tickmode="sync",
    ))
    fig.update_xaxes(title_text="Cycle Days")
    fig.update_yaxes(title_text=labels_dict[y_variable1], secondary_y=False)
    fig.update_yaxes(title_text=labels_dict[y_variable2], secondary_y=True)

    # Set y-axes titles

    return fig
try_chart = generate_graph(sidebar_var1, sidebar_var2, show_benchmarks, sidebar_cycle, harvests, start_time, end_time, active_cycles,show_title = True)
st.plotly_chart(try_chart, use_container_width=True)
if second_graph:
    try_chart2 = generate_graph(sidebar_var3, sidebar_var4, show_benchmarks, sidebar_cycle, harvests, start_time, end_time, active_cycles,colors=["#FFB983","#C900BB"], show_title = False)
    st.plotly_chart(try_chart2, use_container_width=True)




    # Using object notation


