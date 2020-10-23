import cudf
gdf = cudf.read_csv('https://data.heatonresearch.com/data/t81-558/iris.csv')
for column in ['sepal_l', 'sepal_w', 'petal_l', 'petal_w']:
    print(gdf[column].mean())