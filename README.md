
<h1>Table of Contents<span class="tocSkip"></span></h1>
<div class="toc"><ul class="toc-item"><li><span><a href="#vishelper.plot-demo" data-toc-modified-id="vishelper.plot-demo-1"><span class="toc-item-num">1&nbsp;&nbsp;</span><code>vishelper.plot</code> demo</a></span><ul class="toc-item"><li><span><a href="#Univariate-plot" data-toc-modified-id="Univariate-plot-1.1"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>Univariate plot</a></span><ul class="toc-item"><li><span><a href="#Basic" data-toc-modified-id="Basic-1.1.1"><span class="toc-item-num">1.1.1&nbsp;&nbsp;</span>Basic</a></span></li><li><span><a href="#Add-labels-and-title" data-toc-modified-id="Add-labels-and-title-1.1.2"><span class="toc-item-num">1.1.2&nbsp;&nbsp;</span>Add labels and title</a></span></li><li><span><a href="#Add-histogram-specific-keywords" data-toc-modified-id="Add-histogram-specific-keywords-1.1.3"><span class="toc-item-num">1.1.3&nbsp;&nbsp;</span>Add histogram specific keywords</a></span></li></ul></li><li><span><a href="#Bivariate" data-toc-modified-id="Bivariate-1.2"><span class="toc-item-num">1.2&nbsp;&nbsp;</span>Bivariate</a></span><ul class="toc-item"><li><span><a href="#Basic-with-labels" data-toc-modified-id="Basic-with-labels-1.2.1"><span class="toc-item-num">1.2.1&nbsp;&nbsp;</span>Basic with labels</a></span></li><li><span><a href="#Multiple-datasets" data-toc-modified-id="Multiple-datasets-1.2.2"><span class="toc-item-num">1.2.2&nbsp;&nbsp;</span>Multiple datasets</a></span></li><li><span><a href="#Change-colors" data-toc-modified-id="Change-colors-1.2.3"><span class="toc-item-num">1.2.3&nbsp;&nbsp;</span>Change colors</a></span></li><li><span><a href="#Remove-legend" data-toc-modified-id="Remove-legend-1.2.4"><span class="toc-item-num">1.2.4&nbsp;&nbsp;</span>Remove legend</a></span></li></ul></li><li><span><a href="#Horizontal-bar-plot" data-toc-modified-id="Horizontal-bar-plot-1.3"><span class="toc-item-num">1.3&nbsp;&nbsp;</span>Horizontal bar plot</a></span><ul class="toc-item"><li><span><a href="#Multiple-values" data-toc-modified-id="Multiple-values-1.3.1"><span class="toc-item-num">1.3.1&nbsp;&nbsp;</span>Multiple values</a></span></li></ul></li><li><span><a href="#Heatmap" data-toc-modified-id="Heatmap-1.4"><span class="toc-item-num">1.4&nbsp;&nbsp;</span>Heatmap</a></span></li><li><span><a href="#Box-plots" data-toc-modified-id="Box-plots-1.5"><span class="toc-item-num">1.5&nbsp;&nbsp;</span>Box plots</a></span></li></ul></li><li><span><a href="#Demo-of-vh.vhDvhisDF" data-toc-modified-id="Demo-of-vh.vhDvhisDF-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Demo of <code>vh.vhDvhisDF</code></a></span><ul class="toc-item"><li><span><a href="#Subplots" data-toc-modified-id="Subplots-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>Subplots</a></span></li><li><span><a href="#Relationship-between-categorical-and-continuous-variables-(e.g.-cluster-assigment-and-feature)" data-toc-modified-id="Relationship-between-categorical-and-continuous-variables-(e.g.-cluster-assigment-and-feature)-2.2"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>Relationship between categorical and continuous variables (e.g. cluster assigment and feature)</a></span><ul class="toc-item"><li><span><a href="#Calculate-lift,-percent-increase,-absolute-difference-from-population-of-group-statistics" data-toc-modified-id="Calculate-lift,-percent-increase,-absolute-difference-from-population-of-group-statistics-2.2.1"><span class="toc-item-num">2.2.1&nbsp;&nbsp;</span>Calculate lift, percent increase, absolute difference from population of group statistics</a></span></li><li><span><a href="#Heatmap-of-variable-means-by-category" data-toc-modified-id="Heatmap-of-variable-means-by-category-2.2.2"><span class="toc-item-num">2.2.2&nbsp;&nbsp;</span>Heatmap of variable means by category</a></span></li><li><span><a href="#Heatmap-of-lift" data-toc-modified-id="Heatmap-of-lift-2.2.3"><span class="toc-item-num">2.2.3&nbsp;&nbsp;</span>Heatmap of lift</a></span></li></ul></li><li><span><a href="#Box-plots" data-toc-modified-id="Box-plots-2.3"><span class="toc-item-num">2.3&nbsp;&nbsp;</span>Box plots</a></span></li></ul></li></ul></div>


```python
# must go first
%matplotlib inline
%config InlineBackend.figure_format='retina'

# Reloads functions each time so you can edit a script 
# and not need to restart the kernel
%load_ext autoreload
%autoreload 2

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
```


```python
import vishelper as vh
```

    /Users/chloemawer/miniconda3/envs/eda3/lib/python3.6/site-packages/IPython/html.py:14: ShimWarning: The `IPython.html` package has been deprecated since IPython 4.0. You should import from `notebook` instead. `IPython.html.widgets` has moved to `ipywidgets`.
      "`IPython.html.widgets` has moved to `ipywidgets`.", ShimWarning)



```python
data = datasets.load_boston()
```


```python
df = pd.DataFrame(data.data, columns=data.feature_names)
```


```python
print(data.DESCR)
```

    Boston House Prices dataset
    ===========================
    
    Notes
    ------
    Data Set Characteristics:  
    
        :Number of Instances: 506 
    
        :Number of Attributes: 13 numeric/categorical predictive
        
        :Median Value (attribute 14) is usually the target
    
        :Attribute Information (in order):
            - CRIM     per capita crime rate by town
            - ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
            - INDUS    proportion of non-retail business acres per town
            - CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
            - NOX      nitric oxides concentration (parts per 10 million)
            - RM       average number of rooms per dwelling
            - AGE      proportion of owner-occupied units built prior to 1940
            - DIS      weighted distances to five Boston employment centres
            - RAD      index of accessibility to radial highways
            - TAX      full-value property-tax rate per $10,000
            - PTRATIO  pupil-teacher ratio by town
            - B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
            - LSTAT    % lower status of the population
            - MEDV     Median value of owner-occupied homes in $1000's
    
        :Missing Attribute Values: None
    
        :Creator: Harrison, D. and Rubinfeld, D.L.
    
    This is a copy of UCI ML housing dataset.
    http://archive.ics.uci.edu/ml/datasets/Housing
    
    
    This dataset was taken from the StatLib library which is maintained at Carnegie Mellon University.
    
    The Boston house-price data of Harrison, D. and Rubinfeld, D.L. 'Hedonic
    prices and the demand for clean air', J. Environ. Economics & Management,
    vol.5, 81-102, 1978.   Used in Belsley, Kuh & Welsch, 'Regression diagnostics
    ...', Wiley, 1980.   N.B. Various transformations are used in the table on
    pages 244-261 of the latter.
    
    The Boston house-price data has been used in many machine learning papers that address regression
    problems.   
         
    **References**
    
       - Belsley, Kuh & Welsch, 'Regression diagnostics: Identifying Influential Data and Sources of Collinearity', Wiley, 1980. 244-261.
       - Quinlan,R. (1993). Combining Instance-Based and Model-Based Learning. In Proceedings on the Tenth International Conference of Machine Learning, 236-243, University of Massachusetts, Amherst. Morgan Kaufmann.
       - many more! (see http://archive.ics.uci.edu/ml/datasets/Housing)
    



```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CRIM</th>
      <th>ZN</th>
      <th>INDUS</th>
      <th>CHAS</th>
      <th>NOX</th>
      <th>RM</th>
      <th>AGE</th>
      <th>DIS</th>
      <th>RAD</th>
      <th>TAX</th>
      <th>PTRATIO</th>
      <th>B</th>
      <th>LSTAT</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.00632</td>
      <td>18.0</td>
      <td>2.31</td>
      <td>0.0</td>
      <td>0.538</td>
      <td>6.575</td>
      <td>65.2</td>
      <td>4.0900</td>
      <td>1.0</td>
      <td>296.0</td>
      <td>15.3</td>
      <td>396.90</td>
      <td>4.98</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.02731</td>
      <td>0.0</td>
      <td>7.07</td>
      <td>0.0</td>
      <td>0.469</td>
      <td>6.421</td>
      <td>78.9</td>
      <td>4.9671</td>
      <td>2.0</td>
      <td>242.0</td>
      <td>17.8</td>
      <td>396.90</td>
      <td>9.14</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.02729</td>
      <td>0.0</td>
      <td>7.07</td>
      <td>0.0</td>
      <td>0.469</td>
      <td>7.185</td>
      <td>61.1</td>
      <td>4.9671</td>
      <td>2.0</td>
      <td>242.0</td>
      <td>17.8</td>
      <td>392.83</td>
      <td>4.03</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.03237</td>
      <td>0.0</td>
      <td>2.18</td>
      <td>0.0</td>
      <td>0.458</td>
      <td>6.998</td>
      <td>45.8</td>
      <td>6.0622</td>
      <td>3.0</td>
      <td>222.0</td>
      <td>18.7</td>
      <td>394.63</td>
      <td>2.94</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.06905</td>
      <td>0.0</td>
      <td>2.18</td>
      <td>0.0</td>
      <td>0.458</td>
      <td>7.147</td>
      <td>54.2</td>
      <td>6.0622</td>
      <td>3.0</td>
      <td>222.0</td>
      <td>18.7</td>
      <td>396.90</td>
      <td>5.33</td>
    </tr>
  </tbody>
</table>
</div>




```python
kmeans = KMeans(n_clusters=5)
kmeans.fit(df.values)
df['cluster'] = kmeans.labels_
```

# `vishelper.plot` demo

```fig, ax = vh.plot(x, y=None, kind=None, plot_function=None, ax=None,
                     xlabel=None, ylabel=None, title=None, legend=True,
                     labels=None, color=None, **kwargs)```

* Univariate or bivariate 
* Type of plot given either by `kind` (str) or `plot_function` (function that takes and returns `ax`)
* `kwargs` can be any keyword that can be taken by the plot function

## Univariate plot

### Basic


```python
fig, ax = vh.plot(df.RM, kind='hist')
```


![png](figures/output_13_0.png)


### Add labels and title


```python
fig, ax = vh.plot(df.RM, kind='hist', 
                 xlabel='Average number of rooms per dwelling',
                 ylabel='Count',
                 title='Distribution of average number of rooms per dwelling')
```


![png](figures/output_15_0.png)


### Add histogram specific keywords


```python
fig, ax = vh.plot(df.RM, kind='hist', 
                 xlabel='Average number of rooms per dwelling',
                 ylabel='Count',
                 title='Distribution of average number of rooms per dwelling',
                 bins=20)
```


![png](figures/output_17_0.png)


## Bivariate

### Basic with labels


```python
fig, ax =vh.plot(df.RM, df.CRIM, 
                 kind='scatter', xlabel='Average number of rooms per dwelling',
                 ylabel='Per-capita crime rate', 
                 title='Average number of rooms per dwelling vs per-capita crime rate')
```


![png](figures/output_20_0.png)


### Multiple datasets 


```python
X = df.groupby('cluster').RM.apply(list).tolist()
```


```python
Y = df.groupby('cluster').CRIM.apply(list).tolist()
```


```python
fig, ax =vh.plot(X,Y, kind='scatter', xlabel='Average number of rooms per dwelling',
                 ylabel='Per-capita crime rate', 
                 title='Average number of rooms per dwelling vs per-capita crime rate',
                alpha=0.4, legend=True)
```


![png](figures/output_24_0.png)


### Change colors


```python
fig, ax = vh.plot(X, Y, kind='scatter', xlabel='Average number of rooms per dwelling',
                 ylabel='Per-capita crime rate', 
                 title='Average number of rooms per dwelling vs per-capita crime rate',
                alpha=0.4)
```


![png](figures/output_26_0.png)


### Remove legend


```python
fig, ax =vh.plot(X, Y, 
                 kind='scatter', xlabel='Average number of rooms per dwelling',
                 ylabel='Per-capita crime rate', 
                 title='Average number of rooms per dwelling vs per-capita crime rate',
                 alpha=0.4, 
                 legend=False)
```


![png](figures/output_28_0.png)


## Horizontal bar plot


```python
means = df.groupby('cluster').mean()
```


```python
fig, ax = vh.plot(means.index, means.CRIM.values, 
                  kind='barh', ylabel='Cluster number', 
                  xlabel='Per-capita crime rate',
                  title='Average per-capita crime rate of towns by cluster')
```


![png](figures/output_31_0.png)


### Multiple values


```python
river_means = df.groupby(['CHAS', 'cluster']).mean()
```


```python
river_means
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>CRIM</th>
      <th>ZN</th>
      <th>INDUS</th>
      <th>NOX</th>
      <th>RM</th>
      <th>AGE</th>
      <th>DIS</th>
      <th>RAD</th>
      <th>TAX</th>
      <th>PTRATIO</th>
      <th>B</th>
      <th>LSTAT</th>
    </tr>
    <tr>
      <th>CHAS</th>
      <th>cluster</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="5" valign="top">0.0</th>
      <th>0</th>
      <td>1.806027</td>
      <td>0.000000</td>
      <td>16.421000</td>
      <td>0.691400</td>
      <td>5.892500</td>
      <td>92.740000</td>
      <td>2.381510</td>
      <td>4.700000</td>
      <td>385.300000</td>
      <td>17.230000</td>
      <td>197.500000</td>
      <td>17.432000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.235937</td>
      <td>17.572917</td>
      <td>6.588208</td>
      <td>0.484357</td>
      <td>6.470321</td>
      <td>55.181250</td>
      <td>4.912630</td>
      <td>4.287500</td>
      <td>275.204167</td>
      <td>17.901250</td>
      <td>388.667917</td>
      <td>9.272792</td>
    </tr>
    <tr>
      <th>2</th>
      <td>11.347501</td>
      <td>0.000000</td>
      <td>18.612766</td>
      <td>0.667415</td>
      <td>5.928723</td>
      <td>89.825532</td>
      <td>2.095984</td>
      <td>22.936170</td>
      <td>668.393617</td>
      <td>20.194681</td>
      <td>371.665532</td>
      <td>18.567021</td>
    </tr>
    <tr>
      <th>3</th>
      <td>16.060371</td>
      <td>0.000000</td>
      <td>18.100000</td>
      <td>0.666829</td>
      <td>6.076000</td>
      <td>90.125714</td>
      <td>1.988334</td>
      <td>24.000000</td>
      <td>666.000000</td>
      <td>20.200000</td>
      <td>55.669714</td>
      <td>21.007429</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.545340</td>
      <td>13.722826</td>
      <td>11.538696</td>
      <td>0.550761</td>
      <td>6.196489</td>
      <td>67.581522</td>
      <td>3.747732</td>
      <td>4.717391</td>
      <td>402.260870</td>
      <td>17.960870</td>
      <td>383.498370</td>
      <td>12.267826</td>
    </tr>
    <tr>
      <th rowspan="4" valign="top">1.0</th>
      <th>0</th>
      <td>3.535010</td>
      <td>0.000000</td>
      <td>19.580000</td>
      <td>0.871000</td>
      <td>6.152000</td>
      <td>82.600000</td>
      <td>1.745500</td>
      <td>5.000000</td>
      <td>403.000000</td>
      <td>14.700000</td>
      <td>88.010000</td>
      <td>15.020000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.228931</td>
      <td>13.500000</td>
      <td>8.165500</td>
      <td>0.487145</td>
      <td>6.508600</td>
      <td>66.680000</td>
      <td>3.938190</td>
      <td>4.950000</td>
      <td>268.500000</td>
      <td>17.385000</td>
      <td>390.068500</td>
      <td>11.789500</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5.775886</td>
      <td>0.000000</td>
      <td>18.100000</td>
      <td>0.716000</td>
      <td>6.611375</td>
      <td>90.950000</td>
      <td>1.856025</td>
      <td>24.000000</td>
      <td>666.000000</td>
      <td>20.200000</td>
      <td>373.418750</td>
      <td>9.731250</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.747955</td>
      <td>0.000000</td>
      <td>19.580000</td>
      <td>0.738000</td>
      <td>6.495167</td>
      <td>94.783333</td>
      <td>1.780383</td>
      <td>5.000000</td>
      <td>403.000000</td>
      <td>14.700000</td>
      <td>363.030000</td>
      <td>10.800000</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig, ax = vh.plot(
    river_means.loc[0].index,
    [river_means.loc[0.0].CRIM.values, river_means.loc[1.0].CRIM.values],
    kind='barh',
    ylabel='Cluster',
    xlabel='Per-capita crime rate',
    title='Average per-capita crime rate of towns along the river and not',
    labels=["On the river", "Not on the river"])
```


![png](figures/output_35_0.png)


## Heatmap


```python
population_mean = df.drop('cluster', axis=1).mean()
lift = means.div(population_mean)
```


```python
fig, ax = vh.heatmap(lift.T, xlabel='Cluster number', 
                     title='Lift')
```


![png](figures/output_38_0.png)


Any `kwargs` for `seaborn.heatmap()` work for the function too, such as `annot` and ` xrotation`


```python
fig, ax = vh.heatmap(lift.T, xlabel='Cluster number', 
                     title='Lift', xrotation=0, annot=True)
```


![png](figures/output_40_0.png)


## Box plots


```python
fig, ax = vh.plot(df.cluster, df.CRIM, kind='boxplot', ylim=(0,40))
```


![png](figures/output_42_0.png)


# Demo of `vh.vhDvhisDF`


```python
df.columns
```




    Index(['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX',
           'PTRATIO', 'B', 'LSTAT', 'cluster'],
          dtype='object')




```python
feature_labels = ['Per-capita crime rate',
                 'Proportion of residential land zoned for lots over 25,000 sq.ft.',
                 'Proportion of non-retail business acres per town',
                 'Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)',
                 'Nitric oxides concentration (parts per 10 million)',
                 'Average number of rooms per dwelling',
                 'Proportion of owner-occupied units built prior to 1940',
                'Weighted distances to five Boston employment centres',
                'Index of accessibility to radial highways',
                "Full-value property-tax rate per $10,000",
                'Pupil-teacher ratio by town',
                '1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town',
                'Percent of lower status of the population',
                "Median value of owner-occupied homes in $1000's"]

column_labels = dict(zip(df.columns, feature_labels))
```


```python
proportion_features = ['ZN','INDUS','AGE','LSTAT']
```


```python
plot_df = vh.VisDF(df, column_labels=column_labels, cluster_label='cluster')
```

## Subplots

Column label mapping will now occur in any figures created through the `vh.VisDF` object, `plot_df`. 


```python
fig, ax = plot_df.subplots(proportion_features, kind='hist', log=True,
                           main_title='Distribution of proportional variables')
```


![png](figures/output_50_0.png)


## Relationship between categorical and continuous variables (e.g. cluster assigment and feature)

### Calculate lift, percent increase, absolute difference from population of group statistics


```python
comparison = plot_df.compare_categories(category='cluster')
```


```python
comparison.keys()
```




    dict_keys(['lift', 'percent_increase', 'absolute_difference', 'actual'])




```python
comparison['actual']
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CRIM</th>
      <th>ZN</th>
      <th>INDUS</th>
      <th>CHAS</th>
      <th>NOX</th>
      <th>RM</th>
      <th>AGE</th>
      <th>DIS</th>
      <th>RAD</th>
      <th>TAX</th>
      <th>PTRATIO</th>
      <th>B</th>
      <th>LSTAT</th>
    </tr>
    <tr>
      <th>cluster</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.963207</td>
      <td>0.000000</td>
      <td>16.708182</td>
      <td>0.090909</td>
      <td>0.707727</td>
      <td>5.916091</td>
      <td>91.818182</td>
      <td>2.323691</td>
      <td>4.727273</td>
      <td>386.909091</td>
      <td>17.000000</td>
      <td>187.546364</td>
      <td>17.212727</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.235398</td>
      <td>17.259615</td>
      <td>6.709538</td>
      <td>0.076923</td>
      <td>0.484572</td>
      <td>6.473265</td>
      <td>56.065769</td>
      <td>4.837673</td>
      <td>4.338462</td>
      <td>274.688462</td>
      <td>17.861538</td>
      <td>388.775654</td>
      <td>9.466385</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10.910511</td>
      <td>0.000000</td>
      <td>18.572549</td>
      <td>0.078431</td>
      <td>0.671225</td>
      <td>5.982265</td>
      <td>89.913725</td>
      <td>2.077164</td>
      <td>23.019608</td>
      <td>668.205882</td>
      <td>20.195098</td>
      <td>371.803039</td>
      <td>17.874020</td>
    </tr>
    <tr>
      <th>3</th>
      <td>16.060371</td>
      <td>0.000000</td>
      <td>18.100000</td>
      <td>0.000000</td>
      <td>0.666829</td>
      <td>6.076000</td>
      <td>90.125714</td>
      <td>1.988334</td>
      <td>24.000000</td>
      <td>666.000000</td>
      <td>20.200000</td>
      <td>55.669714</td>
      <td>21.007429</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.618969</td>
      <td>12.882653</td>
      <td>12.031020</td>
      <td>0.061224</td>
      <td>0.562224</td>
      <td>6.214776</td>
      <td>69.246939</td>
      <td>3.627282</td>
      <td>4.734694</td>
      <td>402.306122</td>
      <td>17.761224</td>
      <td>382.245204</td>
      <td>12.177959</td>
    </tr>
  </tbody>
</table>
</div>




```python
comparison['lift']
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CRIM</th>
      <th>ZN</th>
      <th>INDUS</th>
      <th>CHAS</th>
      <th>NOX</th>
      <th>RM</th>
      <th>AGE</th>
      <th>DIS</th>
      <th>RAD</th>
      <th>TAX</th>
      <th>PTRATIO</th>
      <th>B</th>
      <th>LSTAT</th>
    </tr>
    <tr>
      <th>cluster</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.546282</td>
      <td>0.000000</td>
      <td>1.500271</td>
      <td>1.314286</td>
      <td>1.275885</td>
      <td>0.941358</td>
      <td>1.338947</td>
      <td>0.612296</td>
      <td>0.495033</td>
      <td>0.947756</td>
      <td>0.921133</td>
      <td>0.525820</td>
      <td>1.360360</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.065502</td>
      <td>1.518846</td>
      <td>0.602467</td>
      <td>1.112088</td>
      <td>0.873583</td>
      <td>1.030015</td>
      <td>0.817584</td>
      <td>1.274735</td>
      <td>0.454317</td>
      <td>0.672865</td>
      <td>0.967815</td>
      <td>1.090003</td>
      <td>0.748150</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3.035959</td>
      <td>0.000000</td>
      <td>1.667677</td>
      <td>1.133894</td>
      <td>1.210080</td>
      <td>0.951887</td>
      <td>1.311175</td>
      <td>0.547336</td>
      <td>2.410580</td>
      <td>1.636808</td>
      <td>1.094257</td>
      <td>1.042417</td>
      <td>1.412624</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.468960</td>
      <td>0.000000</td>
      <td>1.625246</td>
      <td>0.000000</td>
      <td>1.202153</td>
      <td>0.966802</td>
      <td>1.314267</td>
      <td>0.523929</td>
      <td>2.513245</td>
      <td>1.631405</td>
      <td>1.094523</td>
      <td>0.156080</td>
      <td>1.660264</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.172234</td>
      <td>1.133673</td>
      <td>1.080296</td>
      <td>0.885131</td>
      <td>1.013574</td>
      <td>0.988884</td>
      <td>1.009800</td>
      <td>0.955795</td>
      <td>0.495810</td>
      <td>0.985472</td>
      <td>0.962379</td>
      <td>1.071693</td>
      <td>0.962451</td>
    </tr>
  </tbody>
</table>
</div>



### Heatmap of variable means by category


```python
fig, ax = plot_df.category_heatmap(category='cluster', metric='actual', 
                                   transpose=True, xrotation=0)
```


![png](figures/output_58_0.png)


### Heatmap of lift

Actual values are usually hard to visualize like this because they may be different ordres of magnitude. We can plot lift as: 

$$ lift = \frac{\bar{v}_c}{\bar{v}_{pop}} $$

where $v$ represents a variable such as the per capita crime rate, $\bar{v}_c$ is the mean of that variable for a particular cluster, $c$, and $\bar{v}_{pop}$ is the mean for the population.


```python
fig, ax = plot_df.category_heatmap(category='cluster', metric='lift', 
                                   transpose=True)
```


![png](figures/output_61_0.png)


## Box plots


```python
fig, ax = plot_df.subplots([["cluster","CRIM"]], kind='boxplot', log=True,
                           main_title='Distribution of per-capita crime rate by cluster')
```


![png](figures/output_63_0.png)



```python

```


```python

```
