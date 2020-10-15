#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 14:22:20 2020

@author: bruno
"""

# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # SHM modelo predição de dano explicado pelo SHAP

# %%
import pandas as pd
import numpy as np
import shap
import os
import matplotlib.pyplot as plt

# ## Leitura da base de dados
# %%
diretorio = 'Pasta de Dados/Validos/'
quantidadeamostras = 20
quantidadeniveis = 3
quantidadetemperaturas = 7
quantidadepzt = 3

direc = os.listdir(diretorio)
# %%
temperaturas = {1:-10, 2:0, 3:10, 4:20, 5:30, 6:40, 7:50}
values = pd.qcut(np.arange(2000),20, labels=np.arange(20).tolist())
mapping = dict(list(enumerate(values)))
k=0
for i in direc:
    for niv in range(quantidadeniveis):
        plt.figure(niv+k)
    for j in range(1,quantidadetemperaturas+1):
        #print(diretorio+'/'+i+'/'+i+'T'+str(j)+'.csv')
        locals()[i+'_T'+str(j)] = pd.read_csv(diretorio+'/'+i+'/'+i+'T'+str(j)+'.csv',header=None,delimiter=' ').T
        aux_std = eval(i+'_T'+str(j)).groupby(mapping, axis = 1).std()
        aux_std.columns = aux_std.columns+20
        locals()[i+'_T'+str(j)] = pd.concat([eval(i+'_T'+str(j)).groupby(mapping, axis = 1).mean(), aux_std], axis=1)
        print(i+'_T'+str(j))
        for niv in range(quantidadeniveis):
            plt.figure(niv+k)
            plt.plot(eval(i+'_T'+str(j)).loc[0,:],eval(i+'_T'+str(j)).loc[1+quantidadeamostras*niv:quantidadeamostras+1+quantidadeamostras*niv,:].mean())
            plt.title(str(niv)+' '+i)
        #eval(i+'_T'+str(j)).loc[:,'temp'] = temperaturas[j]
    k=k+quantidadeniveis

# %% [markdown]
# ### Construindo os dados de treino e validação
# %%
for i in ['PZT3_T1','PZT3_T2','PZT3_T3','PZT6_T1','PZT6_T2','PZT6_T3','PZT7_T1','PZT7_T2','PZT7_T3']:
    #eval(i).loc[0, 'target'] = 11111
    for niv in range(quantidadeniveis):
        eval(i).loc[1+quantidadeamostras*niv:quantidadeamostras+1+quantidadeamostras*niv, 'target'] = int(niv)
    eval(i).dropna(inplace=True)
eval(i).describe()

# %% [markdown]
# ### separando os dados de treino e validação
# sinais = pd.concat([PZT3_T1,PZT3_T2, PZT3_T3])
# sinais =sinais.sample(frac=1).reset_index(drop=True)
# features = sinais.drop(columns=['target'])
# y = sinais['target']
# y.value_counts()

# from sklearn.model_selection import train_test_split
# Xtrain, Xval, ytrain, yval = train_test_split(features, y, train_size=0.5)

# Xtrain.shape, Xval.shape, ytrain.shape, yval.shape

# yval.value_counts()

# Xtrain.columns

# %%
import seaborn as sns
corr = sinais.corr()
# plot the heatmap
fig_dims = (25, 20)
fig, ax = plt.subplots(figsize=fig_dims)
sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns,
        annot=True, fmt=".2f", ax=ax, cmap='RdYlBu')
# %%
from pycaret.classification import *
exp_clf102 = setup(data = sinais, target = "target", session_id=123)

# %% [markdown]
# ### Para acessar os dados de treino e teste deve acessar da seguinte forma 
# #### Xteste = exp_clf102 \[3\]
# #### yteste = exp_clf102 \[5\]

# %%
best_model = compare_models()
print(best_model)

# %% [markdown]
# ### Precision:
#     Todo mundo que era da classe 1 quantos % foi previstos sendo da classe 1
# ### recall:
#     Todos os exemplos da classe 1 quantos % foram detectados
# ### f1-score
#     Média armonica entre precision e recall
#     

# %%
knn = create_model('knn')
tuned_knn = tune_model(knn)
print(tuned_knn)

plot_model(tuned_knn, plot = 'auc')

plot_model(tuned_knn, plot = 'pr')

plot_model(tuned_knn, plot = 'confusion_matrix')


# %%
import shap

# load JS visualization code to notebook
shap.initjs()

# %% [markdown]
# ### Criando o explainer que é um objeto que representa o modelo
# ### E o shap_values que é uma lista com 2 arrays para cada exemplo treinado, e apresenta a probabilidade de ser da classe 1 para o primeiro array e probabilidade de ser da classe 2 para o array 2 

# %%
explainer = shap.KernelExplainer(tuned_knn.predict_proba, exp_clf102[2])
shap_values = explainer.shap_values(Xtrain)
shap_values[1].shape

# %% [markdown]
# ### Expected_values são as previsoes médias feitas pelo explainer para cada dado treinado

# %%
print('Dados de treino: ' )
# Xtrain
#print(exp_clf102[2])
# Ytrain
print(exp_clf102[4].values)


# %%
print('Dados de treino: ' )
# Xtrain
#print(exp_clf102[3])
# Ytrain
print(exp_clf102[5].values)


# %%
print('Previsoes do modelo: ')
print(tuned_knn.predict(exp_clf102[3]))


# %%
explainer.expected_value

# %% [markdown]
# ### Classe 2 - dano 2 para a instancia 0
#     Probabilidade de pertencer a classe de dano 2

# %%
shap.force_plot(explainer.expected_value[2], shap_values[2][0,:], Xtrain.iloc[0,:])

# %% [markdown]
# ### Classe 1 - dano 1 para a instancia 0
#     Probabilidade de pertencer a classe de dano 1

# %%
shap.force_plot(explainer.expected_value[1], shap_values[1][0,:], Xtrain.iloc[0,:])

# %% [markdown]
# ### Classe 0 - baseline para a instancia 0
#     Probabilidade de pertencer a classe de baseline

# %%
shap.force_plot(explainer.expected_value[0], shap_values[0][0,:], Xtrain.iloc[0,:])


# %%
shap.force_plot(explainer.expected_value[1], shap_values[1], Xtrain)

# %% [markdown]
# ## Features mais importantes do modelo
# #### Cor: vermelho: maior que amédia
# #### Cor: azul: menor que a média
# Eixo x é o impacto na previsão (empurrando pra cima + ou para baixo -)

# %%
shap.summary_plot(shap_values[1], Xtrain)


# %%
shap.dependence_plot(15, shap_values[0], Xtrain, interaction_index=3)


# %%



