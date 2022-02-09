#!/usr/bin/env python
# coding: utf-8

# # Desafio 6
# 
# Neste desafio, vamos praticar _feature engineering_, um dos processos mais importantes e trabalhosos de ML. Utilizaremos o _data set_ [Countries of the world](https://www.kaggle.com/fernandol/countries-of-the-world), que contém dados sobre os 227 países do mundo com informações sobre tamanho da população, área, imigração e setores de produção.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[551]:


import pandas as pd
import numpy as np
import seaborn as sns
import sklearn as sk
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer


# In[552]:


# # Algumas configurações para o matplotlib.
# %matplotlib inline

# from IPython.core.pylabtools import figsize


# figsize(12, 8)

# sns.set()


# In[553]:


countries = pd.read_csv("countries.csv")


# In[554]:


new_column_names = [
    "Country", "Region", "Population", "Area", "Pop_density", "Coastline_ratio",
    "Net_migration", "Infant_mortality", "GDP", "Literacy", "Phones_per_1000",
    "Arable", "Crops", "Other", "Climate", "Birthrate", "Deathrate", "Agriculture",
    "Industry", "Service"
]

countries.columns = new_column_names

countries.head(5)


# ## Observações
# 
# Esse _data set_ ainda precisa de alguns ajustes iniciais. Primeiro, note que as variáveis numéricas estão usando vírgula como separador decimal e estão codificadas como strings. Corrija isso antes de continuar: transforme essas variáveis em numéricas adequadamente.
# 
# Além disso, as variáveis `Country` e `Region` possuem espaços a mais no começo e no final da string. Você pode utilizar o método `str.strip()` para remover esses espaços.

# ## Inicia sua análise a partir daqui

# In[555]:


# Sua análise começa aqui.
numeric_vars = ['Pop_density', 'Coastline_ratio', 'Net_migration', 'Infant_mortality', 'Literacy', 'Phones_per_1000', 'Arable', 'Crops', 'Other', 'Birthrate', 'Deathrate', 'Agriculture', 'Industry', 'Service']
to_clear_space = ['Country', 'Region']

def change_coma_to_dot(number):
    changed = number.replace(',', '.')
    return changed

for var in numeric_vars:

    countries[var] = countries[var].map(lambda number: str(number).replace(',', '.'))

    countries[var] = countries[var].map(lambda number: float(number))


for var in to_clear_space:

    countries[var] = countries[var].map(lambda name: name.strip())

countries.head(5)
    


# ## Questão 1
# 
# Quais são as regiões (variável `Region`) presentes no _data set_? Retorne uma lista com as regiões únicas do _data set_ com os espaços à frente e atrás da string removidos (mas mantenha pontuação: ponto, hífen etc) e ordenadas em ordem alfabética.

# In[556]:


def q1():
    regions = countries['Region'].sort_values()
    return list(regions.unique())
q1()


# ## Questão 2
# 
# Discretizando a variável `Pop_density` em 10 intervalos com `KBinsDiscretizer`, seguindo o encode `ordinal` e estratégia `quantile`, quantos países se encontram acima do 90º percentil? Responda como um único escalar inteiro.

# In[557]:


def q2():
    how_many_countries = len(countries['Country'].unique())
    qty_percentil = how_many_countries / 100
    return round(qty_percentil * 10)
q2()


# # Questão 3
# 
# Se codificarmos as variáveis `Region` e `Climate` usando _one-hot encoding_, quantos novos atributos seriam criados? Responda como um único escalar.

# In[558]:


def q3():
    how_many_regions = len(list(countries['Region'].unique()))
    how_many_climates = len(list(countries['Climate'].unique()))
    return how_many_climates + how_many_regions
q3()


# ## Questão 4
# 
# Aplique o seguinte _pipeline_:
# 
# 1. Preencha as variáveis do tipo `int64` e `float64` com suas respectivas medianas.
# 2. Padronize essas variáveis.
# 
# Após aplicado o _pipeline_ descrito acima aos dados (somente nas variáveis dos tipos especificados), aplique o mesmo _pipeline_ (ou `ColumnTransformer`) ao dado abaixo. Qual o valor da variável `Arable` após o _pipeline_? Responda como um único float arredondado para três casas decimais.

# In[559]:


test_country = [
    'Test Country', 'NEAR EAST', -0.19032480757326514,
    -0.3232636124824411, -0.04421734470810142, -0.27528113360605316,
    0.13255850810281325, -0.8054845935643491, 1.0119784924248225,
    0.6189182532646624, 1.0074863283776458, 0.20239896852403538,
    -0.043678728558593366, -0.13929748680369286, 1.3163604645710438,
    -0.3699637766938669, -0.6149300604558857, -0.854369594993175,
    0.263445277972641, 0.5712416961268142
]
def padronize(x):
    result = (x - x.mean()) / x.std()
    return result


# In[560]:


def q4():
    arable = countries['Arable'].dropna()
    arable_list = list(arable)
    mediana = arable.quantile(q=0.5)
    # test = []
    # for value in arable_list:
    #     serie = pd.Series(test)
    #     if value != 0:
    #         test.append(serie.quantile(q=0.5))
    #     if value == 0:
    #         test.append(0)
    # test2 = pd.Series(test)
    to_return = (test_country[11] - arable.mean()) / arable.std()
    return round(to_return, 3)
q4()


# ## Questão 5
# 
# Descubra o número de _outliers_ da variável `'Net_migration'` segundo o método do _boxplot_, ou seja, usando a lógica:
# 
# $$x \notin [Q1 - 1.5 \times \text{IQR}, Q3 + 1.5 \times \text{IQR}] \Rightarrow x \text{ é outlier}$$
# 
# que se encontram no grupo inferior e no grupo superior.
# 
# Você deveria remover da análise as observações consideradas _outliers_ segundo esse método? Responda como uma tupla de três elementos `(outliers_abaixo, outliers_acima, removeria?)` ((int, int, bool)).

# In[561]:


# [q1 - 1.5x(q3 - q1), q3 + 1.5(q3 - q1)]

def q5():
    net_migration = countries['Net_migration']
    q3 = net_migration.quantile(q=0.75)
    q1 = net_migration.quantile(q=0.25)
    iqr = q3 - q1
    k = 1.5

    min_value = q1 - k * iqr
    max_value = q3 + k * iqr

    how_many_down_outliers = len(list(net_migration[net_migration < min_value]))
    how_many_up_outliers = len(list(net_migration[net_migration > max_value]))


    to_return = (how_many_down_outliers, how_many_up_outliers, False)

    return to_return
q5()


# ## Questão 6
# Para as questões 6 e 7 utilize a biblioteca `fetch_20newsgroups` de datasets de test do `sklearn`
# 
# Considere carregar as seguintes categorias e o dataset `newsgroups`:
# 
# ```
# categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
# newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)
# ```
# 
# 
# Aplique `CountVectorizer` ao _data set_ `newsgroups` e descubra o número de vezes que a palavra _phone_ aparece no corpus. Responda como um único escalar.

# In[562]:


import re

def isWordPresent(sentence) :
    x = re.findall(r"\bphone\b", sentence, re.IGNORECASE)
    if len(x) > 0:
        return (True, len(x))
    return (False, 0)


categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)


# In[563]:


def q6():
    data = newsgroup.data
    count = 0
    for sentence in data:
        curr = isWordPresent(sentence)
        if curr[0]:
            count += curr[1]
    return count
q6()


# ## Questão 7
# 
# Aplique `TfidfVectorizer` ao _data set_ `newsgroups` e descubra o TF-IDF da palavra _phone_. Responda como um único escalar arredondado para três casas decimais.

# In[564]:


def q7():
    # Retorne aqui o resultado da questão 4.
    pass

