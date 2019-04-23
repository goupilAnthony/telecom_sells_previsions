import warnings
import itertools
import numpy as np
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'k'

df = pd.read_csv('CSV/clean_orange.csv',low_memory=False)
cols = list(df.columns)
cols.remove('Journée de la Vente')
cols.remove('Nombre de produits vendus')

df = df.drop(columns=cols)
df['Journée de la Vente'] = pd.to_datetime(df['Journée de la Vente'])
df = df.sort_values('Journée de la Vente')
df = df.set_index('Journée de la Vente')
daily = df['Nombre de produits vendus'].resample('D', how='sum')
weekly = df['Nombre de produits vendus'].resample('W', how='sum')

weekly = weekly[:-1] # remove last week because we only have one day 

weekly.plot(figsize=(15, 6))
plt.show()

from pylab import rcParams
rcParams['figure.figsize'] = 15, 10
decomposition = sm.tsa.seasonal_decompose(weekly, model='additive')
fig = decomposition.plot()
plt.show()
# we can see that weekly sales are a litle stable

# Make a set of parameters to search for the best ARIMA model in our case
p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

# Grid searching the best parameters
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(weekly,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
            results = mod.fit()
            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue

# lowest AIC is : ARIMA(0, 1, 1)x(0, 1, 1, 12)12 - AIC:1656.1218179497578
            
        
mod = sm.tsa.statespace.SARIMAX(weekly,
                                order=(0, 1, 1),
                                seasonal_order=(0, 1, 1, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results = mod.fit()
print(results.summary().tables[1])
        

results.plot_diagnostics(figsize=(16, 8))
plt.show()
        
# Trying to predict 
pred = results.get_prediction(start=70,end=104, dynamic=False)
pred_ci = pred.conf_int()
ax = weekly.plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(18, 9))
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)
ax.set_xlabel('Date')
ax.set_ylabel('Sales')
plt.legend()
plt.show()
        
        
        
        
        
        
        
        
        