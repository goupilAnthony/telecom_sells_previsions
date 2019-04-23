# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""
import pandas as pd

#df = pd.read_csv('orange_clean_v3.csv',low_memory=False)

df = pd.read_csv('CSV/clean_orange.csv',low_memory=False)

#%%
df.memory_usage(deep=True) / 1024 ** 2

df['Mois de la Vente'].memory_usage(deep=True) / 1024 **2

def recap_nan(data):
    """
        Fonction qui renvoie un récapitulatif des features
        
        Input : Dataframe 
        Output : Dataframe
    """
  feature_understanding = pd.DataFrame(columns=['feature_name','value_type','unique_value_number','unique_value_list','nan_percentage','nan_nbr']) 
  cols = list(data.columns)
  for col in cols:
    nb_nan = data[col].isna().sum()
    total = len(data)
    nan_perc = nb_nan / total *100
    type_val = []
    un_val_list = list(data[col].unique()) 
    for val in un_val_list:
      if type(val) not in type_val:
        type_val.append(type(val))
    un_val_nb = len(un_val_list)
    memory = data[col].memory_usage(deep=True) / 1024 ** 2
    line = { 'feature_name' : col, 'value_type' : type_val, 'unique_value_number' : un_val_nb, 'unique_value_list' : un_val_list,'memory_usage':memory, 'nan_percentage' : nan_perc,'nan_nbr':nb_nan}
    feature_understanding = feature_understanding.append(line,ignore_index=True)
  return feature_understanding

recap = recap_nan(df2)

df2.columns
#%%
df = pd.read_csv('CSV/clean_orange.csv',low_memory=False)

def pre_process(df2):
    """
        Fonction de pré-processing qui enlève les EAN des produit taggés 'ALIENATION' et FDV EFFECTIVE
        
        Input : Dataframe avec colonnes : 'Mois de la Vente', 'Journée de la Vente', 'Libellé AD', 'Code PDV',
            'Libellé PDV', 'Code Sous Famille', 'Libellé Sous Famille',
            'Code Produit (EAN)', 'Libellé Produit', 'Nombre de produits vendus',
            'CA net TTC', 'CA TTC avant remise', 'Prix de revient du produit'
            (CSV clean_orange de Boris)
            
        Requirements : - CSV 'EANs par famille 20190415.xlsx' tel quel
            
        Output : Dataframe sans les EAN alienation et fdv effective , avec colonne weekday et week_number et mois de la vente
    """
    df2 = df2.drop(columns=['Unnamed: 0'])
    ean_fam_file_path = './CSV/EANs par famille 20190415.xlsx'
    
    df_ean = pd.read_excel(ean_fam_file_path)
    ean_to_drop = df_ean[df_ean['Statut du cycle de vie'] != 'ALIENATION']
    ean_to_drop = ean_to_drop[ean_to_drop['Statut du cycle de vie'] != 'FDV EFFECTIVE']
    ean_to_keep = ean_to_drop[ean_to_drop['Statut du cycle de vie'] != 'ANNULEE'] # LOL
    
    df2 = df2.drop(columns=['CA TTC avant remise','CA net TTC','Prix de revient du produit'])
    df2 = df2.drop(columns=['Libellé PDV'])
    df2['Journée de la Vente'] = pd.to_datetime(df2['Journée de la Vente'])
    df2['Mois de la Vente'] = df2['Journée de la Vente'].dt.month
    
    ean_to_keep = list(ean_to_keep['EAN'])

    def isinlist(row):
        if row in ean_to_keep:
            return True
        else:
            return False
    df2 = df2[df2['Code Produit (EAN)'].apply(isinlist)]
    #df2 = df2.drop(columns=['Code Produit (EAN)'])
    df2 = df2.drop(columns=['Libellé Sous Famille'])
    df2['Weekday'] = df2['Journée de la Vente'].apply(lambda x: x.weekday())
    df2['Week_number'] = df2['Journée de la Vente'].apply(lambda x: x.strftime("%V"))
    df2['Week_number'] = df2['Week_number'].astype('int')
    df2['Day_number'] = df2['Journée de la Vente'].apply(lambda x: x.day())
    df2['Day_number'] = df2['Day_number'].astype('int')
    
    return df2

df = pre_process(df)
#%%
df['Journée de la Vente'] = pd.to_datetime(df['Journée de la Vente'])
df['Week_number'] = df['Journée de la Vente'].apply(lambda x: x.strftime("%V"))
df['Week_number'] = df['Week_number'].astype('int')
dfs = pd.DataFrame(df.groupby(['Code Produit (EAN)','Week_number'])['Nombre de produits vendus'].sum()).reset_index(level=[0,1])
dfp = pd.read_csv('CSV/produits_code.csv')
dfp = dfp.drop(columns=['Unnamed: 0'])
def find_libelle(code):
    libelle = dfp[dfp['code'] == code]['libelle'].values[0]
    libelle = libelle.replace( "[" ,"")
    libelle = libelle.replace( "]" ,"")
    libelle = libelle.replace( "'" ,"")
    return libelle

libelle = find_libelle(8562014527)
dfs['Libelle'] = dfs['Code Produit (EAN)'].apply(lambda x: find_libelle(x))
############ TEST  ###########################

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(dfs['Libelle'])


vect = pd.DataFrame(data = X.toarray(), columns = vectorizer.get_feature_names())
df_final = pd.concat([dfs, vect], axis=1)
df_final = df_final.drop(columns=['Code Produit (EAN)','Libelle'])

list_x = list(df_final.columns)
list_x.remove('Nombre de produits vendus')

X = df_final[list_x]
y = df_final['Nombre de produits vendus']  
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

from sklearn.ensemble import RandomForestRegressor
regr = RandomForestRegressor(max_depth=2, n_estimators=200)
regr.fit(X_train,y_train)

regr.score(X_test,y_test)
#df['Journée de la Vente'].dtype
#%%

df2 = df  ####### COPY DF W/O DUMMIES TO DF2 TO TEST RANDOM FOREST
def vectorize_libelle(dataframe):
    """
        Fonction de vectorisation du nom de produit
        
        Input : Dataframe avec une colonne 'Libellé Produit'
        
        Output : Dataframe avec colonne 'Libellé Produit' remplacée par la vectorisation 
    """
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus)
    
    return dataframe
#%%
categoricals = ['Code Sous Famille','Code PDV','Libellé AD','Weekday']
for col in categoricals:
  dum = pd.get_dummies(df2.loc[:,col])
  dum = dum.drop(dum.columns[0],axis=1)
  df2 = df2.drop(col,axis=1)
  df2 = pd.concat([df2,dum],axis=1)
  
list_x = list(df.columns)
list_x.remove('Nombre de produits vendus')
list_x.remove('Journée de la Vente')
########### DF NON DUMMIES POUR RANDOM FOREST REGRESSOR

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
df['Libellé AD'] = le.fit_transform(df['Libellé AD'])

X = df[list_x]
y = df['Nombre de produits vendus']  
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
  
from sklearn.ensemble import RandomForestRegressor
regr = RandomForestRegressor(max_depth=2, n_estimators=200)
regr.fit(X_train,y_train)


############ DF2 DUMMIES POUR AUTRES REGRESSIONS


#%%
df.dtypes.unique()

df = df.fillna(method='ffill')

colonnes = list(df.columns)
print(colonnes)
print(recap[recap['feature_name'] == 'Libellé AD'])

df = df.drop(columns=['CA TTC avant remise','CA net TTC','Prix de revient du produit'])

list_sf = list(df['Libellé Sous Famille'].unique())
sous_familles = pd.DataFrame(columns=['code','libelle'])
for sf in list_sf:
    res = df[df['Libellé Sous Famille'] == sf]
    code = res['Code Sous Famille'].unique()[0]
    line = {'code':code,'libelle':sf}
    sous_familles = sous_familles.append(line,ignore_index=True)

df = df.drop(columns=['Libellé Sous Famille'])

list_produits = list(df['Libellé Produit'].unique())
codes_produits = pd.DataFrame(columns=['code','libelle'])
for p in list_produits:
    res = df[df['Libellé Produit'] == p]
    code = list(res['Code Produit (EAN)'].unique())
    line = {'code':code,'libelle':p}
    codes_produits = codes_produits.append(line,ignore_index=True)

#%%
import matplotlib.pyplot as plt
pd.plotting.scatter_matrix(df, figsize=(20,20))
plt.show()

import multiprocessing as mp

# Check the number of cores and memory usage
num_cores = mp.cpu_count()




