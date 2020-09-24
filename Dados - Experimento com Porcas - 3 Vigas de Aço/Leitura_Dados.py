# IMPORTA AS BIBLIOTECAS NECESSARIAS PARA A EXECUCAO DO PROGRAMA -------------------------------------------------------
from tkinter import *  # NECESSARIA PARA A MANIPULACAO DE COMPONENTES DA INTERFACE
from tkinter import filedialog  # NECESSARIA PARA A CRIACAO DA INTERFACE MODAL (SELECIONAR O DIRETORIO RAIZ)
from tkinter import messagebox  # NECESSARIA PARA A CRIACAO DA INTERFACE MODAL (ALERTA DE ERROR)
import os as so  # NECESSARIA PARA A MANIPULACAO DAS FUNCOES DO SISTEMA OPERACIONAL
from os.path import isfile, join  # NECESSARIA PARA A VERIFICACAO DOS SUBDIRETORIOS
import numpy as np  # NECESSARIA PARA A MANIPULACAO VETORIAL (ARRAYS)
import csv  # NECESSARIA PARA A LEITURA DOS ARQUIVOS .CSV
import tensorflow as tf  # NECESSARIA PARA A EXECUCAO/TREINAMENTO E SIMULACAO DO MODELO CNN
import matplotlib.pyplot as plt  # NECESSARIA PARA A PLOTAGEM DAS AMOSTRAS E DEMAIS GRAFICOS
from sklearn.preprocessing import normalize
from tensorflow import keras
import os
# ----------------------------------------------------------------------------------------------------------------------

# IMPORTA O MODELO (ARQUITETURA) DEFINIDA EM OUTRO .py -----------------------------------------------------------------
import Codigos_2020.Qualificacao.Vigas_UFU_CNN.Arquitetura_CNN_3 as Rede

# ----------------------------------------------------------------------------------------------------------------------

''' REALIZA A LEITURA DO DIRETORIO E QUANTIDADES DE CADA COMPONENTE ================================================ '''

''' INTERFACE DE ENTRADA (INICIO) ---------------------------------------------------------------------------------- '''

global diretorio, quantidadeamostras, quantidadeniveis, quantidadetemperaturas, quantidadepzt
diretorio = 'C:/Users/Stanl/OneDrive/Área de Trabalho/Experimento - Porcas - UFU - 3 Vigas/Dados - Experimento com Porcas - 3 Vigas de Aço/Pasta de Dados/Validos/'
quantidadeamostras = 20
quantidadeniveis = 5
quantidadetemperaturas = 3
quantidadepzt = 1

''' REALIZA A LEITURA DE CADA AMOSTRA (INICIO) --------------------------------------------------------------------- '''

# ENCONTRA OS SUB-DIRETORIOS CONTIDOS DENTRO DO DIRETORIO PRINCIPAL ----------------------------------------------------
direc = so.listdir(diretorio)
onlyfiles = [f for f in so.listdir(diretorio) if isfile(join(diretorio, f))]
for i in range(np.shape(onlyfiles)[0]):
    direc.remove(onlyfiles[i])
diretorios = []
for j in range(quantidadepzt):
    diretorios.append(diretorio + direc[j])

# REALIZA A LEITURA DOS ARQUIVOS .csv POR PZT E TEMPERATURA ------------------------------------------------------------
real = []
freq = []
for j in range(quantidadepzt):
    aux1 = []
    aux2 = []
    for i in range(quantidadetemperaturas):
        arquivo = diretorios[j] + '/' + direc[j] + 'T' + str(i + 2) + '.csv'
        print(arquivo)
        # arquivo = diretorios[j] + '/' + direc[j] + 'T' + str(i + 1) + '.csv'
        with open(arquivo, 'r') as ficheiro:
            reader = csv.reader(ficheiro, delimiter=' ')
            aux3 = np.zeros((2000, quantidadeamostras * quantidadeniveis), 'float')
            aux4 = np.zeros((2000, 1), 'float')
            k = 0
            for linha in reader:
                aux3[k][:] = linha[1:(quantidadeamostras * quantidadeniveis + 1)]
                aux4[k] = linha[0]
                k = k + 1
        aux1.append(aux3)
        aux2.append(aux4)
    real.append(aux1)
    freq.append(aux2)
# ----------------------------------------------------------------------------------------------------------------------

# TRANSFORMA A LISTA EM UM VETOR (ARRAY) (PARA ACESSAR OS INDICES UTILIZAR DA SEGUINTE FORMA: amostras[0,0] ------------
amostras = np.asarray(real)
freq_plot = np.asarray(freq)

plt.rcParams["font.family"] = "Times New Roman"
plt.plot(freq_plot[0, 0, 668:1556, 0], np.mean(amostras[0, 0, 668:1556, 0:(quantidadeamostras-1)], axis=1),
         'k', label="Baseline - $0^\circ C$")
plt.plot(freq_plot[0, 0, 668:1556, 0], np.mean(amostras[0, 0, 668:1556, quantidadeamostras:99], axis=1),
         'r', label="Damage - $0^\circ C$")
plt.plot(freq_plot[0, 0, 668:1556, 0], np.mean(amostras[0, 1, 668:1556, 0:(quantidadeamostras-1)], axis=1),
         'b', label="Baseline - $10^\circ C$")
plt.plot(freq_plot[0, 0, 668:1556, 0], np.mean(amostras[0, 1, 668:1556, quantidadeamostras:99], axis=1),
         'm', label="Damage - $10^\circ C$")
plt.plot(freq_plot[0, 0, 668:1556, 0], np.mean(amostras[0, 2, 668:1556, 0:(quantidadeamostras-1)], axis=1),
         'g', label="Baseline - $20^\circ C$")
plt.plot(freq_plot[0, 0, 668:1556, 0], np.mean(amostras[0, 2, 668:1556, quantidadeamostras:99], axis=1),
         'c', label="Damage - $20^\circ C$")
plt.legend()
plt.grid()
plt.ylabel('Impedance - Real Part($\Omega$)')
plt.xlabel('Frequency ($Hz$)')
plt.title('Mean of the Impedance Signatures of Each Rated Group')
# ----------------------------------------------------------------------------------------------------------------------

''' REALIZA A LEITURA DE CADA AMOSTRA (TERMINO) -------------------------------------------------------------------- '''

''' PONTO ATUAL: EXECUTOU-SE APENAS A LEITURA DOS ARQUIVOS .csv, ARMAZENANDO OS DADOS DO EXPERIMENTO EM UM ARRAY ----'''

''' DEFINE AS CONFIGURACOES DOS DADOS PARA TREINAMENTO/SIMULACAO (INICIO) ------------------------------------------ '''

# DEFINE AS TAXAS (QUANTIDADES) DE DADOS A SEREM TREINADOS OU SIMULADOS ------------------------------------------------
taxa_treinamento = 0.9
taxa_simulacao = 1.0 - taxa_treinamento
# ----------------------------------------------------------------------------------------------------------------------

# DEFINE ALEATORIAMENTE QUAIS SAO OS INDICES DAS AMOSTRAS DE VALIDACAO PARA O MODELO DE REDE A SER DESENVOLVIDO --------
quantidade_simulacao = np.ceil(quantidadeamostras * taxa_simulacao)
indices = np.random.choice(quantidadeamostras - 1, int(quantidade_simulacao), replace=False)
indices_simulacao = indices
for i in range(quantidadeniveis - 1):
    indices_simulacao = np.concatenate((indices_simulacao, indices + quantidadeamostras), axis=0)
    indices = indices + quantidadeamostras
# ----------------------------------------------------------------------------------------------------------------------

# DEFINE QUAIS SAO OS INDICES DAS AMOSTRAS DE TREINAMENTO PARA O MODELO DE REDE A SER DESENVOLVIDO ---------------------
indices_treinamento = np.asarray(np.arange(quantidadeamostras * quantidadeniveis))
indices_treinamento = np.delete(indices_treinamento, indices_simulacao, axis=0)
# ----------------------------------------------------------------------------------------------------------------------

# ARMAZENA AS AMOSTRAS DE TREINAMENTO E DE SIMULACAO EM VARIAVEIS SEPARADAS --------------------------------------------
amostras_simulacao = amostras[:, :, 668:1556, indices_simulacao]
amostras_simulacao = amostras_simulacao[0]
amostras_treinamento = amostras[:, :, 668:1556, indices_treinamento]
amostras_treinamento = amostras_treinamento[0]
# ----------------------------------------------------------------------------------------------------------------------

# AJUSTANDO A DISPOSICAO DOS DADOS EM RELACAO AS TEMPERATURAS (COLOCA EM UMA MATRIZ BIDIMENSIONAL) ---------------------
amostras_train = np.zeros((np.shape(amostras_simulacao)[1],
                           quantidadetemperaturas * np.shape(amostras_treinamento)[2]))
amostras_simu = np.zeros((np.shape(amostras_simulacao)[1],
                          quantidadetemperaturas * np.shape(amostras_simulacao)[2]))
lit = 0
lut = np.shape(amostras_treinamento)[2]
lis = 0
lus = np.shape(amostras_simulacao)[2]
for i in range(quantidadetemperaturas):
    am_t = amostras_treinamento[i]
    am_s = amostras_simulacao[i]
    amostras_train[0:np.shape(amostras_treinamento)[1], lit:lut] = am_t
    amostras_simu[0:np.shape(amostras_simulacao)[1], lis:lus] = am_s
    lit = lit + np.shape(amostras_treinamento)[2]
    lut = lut + np.shape(amostras_treinamento)[2]
    lis = lis + np.shape(amostras_simulacao)[2]
    lus = lus + np.shape(amostras_simulacao)[2]
amostras_treinamento = amostras_train
amostras_simulacao = amostras_simu
# ----------------------------------------------------------------------------------------------------------------------

# NORMALIZA OS SINAIS ENTRE 0 e 1 --------------------------------------------------------------------------------------
amostras_treinamento = normalize(amostras_treinamento, axis=0)
amostras_simulacao = normalize(amostras_simulacao, axis=0)
# ----------------------------------------------------------------------------------------------------------------------

# DEFINE OS TARGETS DAS AMOSTRAS DE TREINAMENTO E DE SIMULACAO ---------------------------------------------------------
todas_classes = np.arange(quantidadetemperaturas * quantidadeniveis)
target_treinamento = np.repeat(todas_classes, int(quantidadeamostras - quantidade_simulacao))
target_simulacao = np.repeat(todas_classes, int(quantidade_simulacao))

quant = quantidadeamostras - quantidade_simulacao
aux = np.concatenate((np.zeros(int(quant)), np.ones(int((quantidadeniveis - 1) * quant))), axis=0)
target_treinamento = []
for i in range(quantidadetemperaturas):
    target_treinamento = np.concatenate((target_treinamento, aux), axis=0)
# ----------------------------------------------------------------------------------------------------------------------

# DEFINE OS TARGETS COMO VETOR DE CATEGORIAS ([1,0,0,0]) ---------------------------------------------------------------
target_treinamento = tf.keras.utils.to_categorical(target_treinamento,
                                                   num_classes=2)
target_simulacao = tf.keras.utils.to_categorical(target_simulacao,
                                                 num_classes=quantidadetemperaturas * quantidadeniveis)
# ----------------------------------------------------------------------------------------------------------------------

# PLOTANDO AS AMOSTRAS UTILIZADAS PARA SIMULACAO (APENAS VERIFICACAO) --------------------------------------------------
plt.plot(amostras_simulacao)
plt.show()
# ----------------------------------------------------------------------------------------------------------------------

# AJUSTA AS DIMENSOES DO VETOR (ARRAY) DE DADOS PARA A SUA APLICACAO NO MODELO CNN -------------------------------------
amostras_treinamento = np.transpose(amostras_treinamento)
amostras_simulacao = np.transpose(amostras_simulacao)
amostras_treinamento = tf.keras.backend.expand_dims(amostras_treinamento, axis=2)
amostras_simulacao = tf.keras.backend.expand_dims(amostras_simulacao, axis=2)
# ----------------------------------------------------------------------------------------------------------------------

# ENCONTRA AS DIMENSOES DE CADA SINAL ----------------------------------------------------------------------------------
formato_sinal = np.shape(amostras_treinamento)[1:3]
# ----------------------------------------------------------------------------------------------------------------------

# IMPRIMINDO AS INFORMACOES DOS DADOS ----------------------------------------------------------------------------------
print('')
print('---------------------------------------------------------------------------------')
print('Quantidade de indices para simulação por estado: ', int(quantidade_simulacao))
print('---------------------------------------------------------------------------------')
print('')
print('---------------------------------------------------------------------------------')
print('Indices utilizados na simulação: ', indices_simulacao)
print('---------------------------------------------------------------------------------')
print('')
print('---------------------------------------------------------------------------------')
print('Dimensões dos dados de treinamento: ', np.shape(amostras_treinamento))
print('---------------------------------------------------------------------------------')
print('')
print('---------------------------------------------------------------------------------')
print('Dimensões dos dados de simulação: ', np.shape(amostras_simulacao))
print('---------------------------------------------------------------------------------')
print('')
print('---------------------------------------------------------------------------------')
print('Dimensões dos targets de treinamento: ', np.shape(target_treinamento))
print('---------------------------------------------------------------------------------')
print('')
print('---------------------------------------------------------------------------------')
print('Dimensões dos targets de simulação: ', np.shape(target_simulacao))
print('---------------------------------------------------------------------------------')
print('')
# ----------------------------------------------------------------------------------------------------------------------

''' DEFINE AS CONFIGURACOES DOS DADOS PARA TREINAMENTO/SIMULACAO (TERMINO) ----------------------------------------- '''

''' PONTO ATUAL: EXECUTOU-SE A LEITURA DOS ARQUIVOS .csv E DEFINIU OS CONJUNTOS DE TREINAMENTO E DE SIMULACAO ------ '''