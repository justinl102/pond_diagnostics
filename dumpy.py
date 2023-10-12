dataframe = pd.DataFrame(
    np.random.randn(10, 20),


    columns=('col %d' % i for i in range(20)))
monitorings = pd.read_csv('monitoring_cleaned.csv')

monitorings['FechaMuestreo'] = pd.to_datetime(monitorings['FechaMuestreo'])


st.write(monitorings.shape)

x = st.slider('x')  # 👈 this is a widget
st.write(x, 'squared is', x * x)


st.dataframe(dataframe.style.highlight_max(axis=0))

chart_data = pd.DataFrame(
     np.random.randn(20, 3),
     columns=['a', 'b', 'c'])

st.line_chart(chart_data)


print(clean_df.columns)
x1 = clean_df.loc[clean_df['density_ha_bin'] == '<130K', 'PesoPromedio2']
x2 = clean_df.loc[clean_df['density_ha_bin'] == '130-160k','PesoPromedio2']
x3 = clean_df.loc[clean_df['density_ha_bin'] == '>160k','PesoPromedio2']
print(x1.shape)
print(x2.shape)
print(x3.shape)
# Group data together
hist_data = [x1, x2, x3]

group_labels = ['130K', '130-160K', '160K+']

# Create distplot with custom bin_size
fig = ff.create_distplot(
        hist_data, group_labels)


st.plotly_chart(fig, use_container_width=True)


st.line_chart(data = variable1_df, x ="Cycle_Day", y = y_variable1 )


arr = np.random.normal(1, 1, size=100)
fig, ax = plt.subplots()
sns.distplot(arr, bins=20)

st.pyplot(fig)


def get_benchmark(df, weight_min, weight_max, model):
  train_df = df[(df['PesoPromedio2'] > weight_min) & (df['PesoPromedio2'] < weight_max)]
  x_train = df['cycle_days'] 
  y_train = df['PesoPromedio'] 

  curve_params, covariance = curve_fit(model,
                                     x_train,
                                     y_train,
                                     p0 = [37.728, 0.02348,0.93, 8.008],
                                     maxfev = 100000
                                          )
  return curve_params