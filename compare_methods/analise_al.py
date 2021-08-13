# -*- coding: utf-8 -*-
"""
Criação: Abril/2021
    
Modificação: 

@author: Edvonaldo (edvonaldohoracio@gmail.com)
"""
import os
import csv
import math
import random
import pylab
import matplotlib
import numpy as np
import pandas as pd
import datetime as dt
import seaborn as sns
from scipy.stats import norm
import matplotlib.pyplot as plt

# Funções úteis
# Tempo de execução
def seconds_transform(seconds_time):
  hours = int(seconds_time/3600)
  rest_1 = seconds_time%3600
  minutes = int(rest_1/60)
  seconds = rest_1 - 60*minutes
  #print(seconds)
  print(" ", (hours), "h ", (minutes), "min ", round(seconds,2), " s")
  return hours, minutes, round(seconds,2)

# Controle de build
# Controle de build
def version_file(name_file, fields, rows_version):
    rows_aux = []
    
    if os.path.isfile(name_file):
        file_version_py = name_file      
        df = pd.read_csv(name_file)
        teste = df['Version'].iloc[-1]
        value = int(teste)
        value += 1
        rows_version['Version'] = value
        rows_aux = [rows_version]
        with open(file_version_py, 'a') as csvfile:
            # creating a csv writer object  
            csvwriter = csv.DictWriter(csvfile, fieldnames = fields) 
            # writing the data rows  
            csvwriter.writerows(rows_aux) 
    else:
        file_version_py = name_file
        rows_aux = [rows_version]
        with open(file_version_py, 'a') as csvfile:
            # creating a csv writer object  
            csvwriter = csv.DictWriter(csvfile, fieldnames = fields) 
            # writing the fields
            csvwriter.writeheader()
            # writing the data rows 
            csvwriter.writerows(rows_aux)
            #print ("File not exist")

#%% PREPARANDO OS DADOS

data_al2014 = pd.read_csv(r'tcc_data/AL_2014.csv')
data_al2015 = pd.read_csv(r'tcc_data/AL_2015.csv')
data_al2016 = pd.read_csv(r'tcc_data/AL_2016.csv')
data_al2017 = pd.read_csv(r'tcc_data/AL_2017.csv')
data_al2018 = pd.read_csv(r'tcc_data/AL_2018.csv')

labels_al = [] # Labels
features_al = [] # Features
features_al_list = [] # Guardando as variáveis das features
features_al_list_oh = [] # Variáveis das features com one-hot

#%% Percentual de valores NaN

nan_2014 = float(data_al2014['NT_GER'].isnull().sum());
nan_2015 = float(data_al2015['NT_GER'].isnull().sum());
nan_2016 = float(data_al2016['NT_GER'].isnull().sum());
nan_2017 = float(data_al2017['NT_GER'].isnull().sum());
nan_2018 = float(data_al2018['NT_GER'].isnull().sum());

column_2014 = float(data_al2014.shape[0]);
column_2015 = float(data_al2015.shape[0]);
column_2016 = float(data_al2016.shape[0]);
column_2017 = float(data_al2017.shape[0]);
column_2018 = float(data_al2018.shape[0]);

per_2014 = nan_2014/column_2014;
per_2015 = nan_2015/column_2015;
per_2016 = nan_2016/column_2016;
per_2017 = nan_2017/column_2017;
per_2018 = nan_2018/column_2018;


print("Qtde. % NaN values in NT_GER 2014:", 100*round(per_2014, 4));
print("Qtde. % NaN values in NT_GER 2015:", 100*round(per_2015, 4));
print("Qtde. % NaN values in NT_GER 2016:", 100*round(per_2016, 4));
print("Qtde. % NaN values in NT_GER 2017:", 100*round(per_2017, 4));
print("Qtde. % NaN values in NT_GER 2018:", 100*round(per_2018, 4));

#%% Pré-processamento e enriquecimento

def processing_set_al(data_al2014, data_al2015, data_al2016, data_al2017, data_al2018):
    #% 2.1 - Limpeza
    del data_al2014['Unnamed: 0']
    del data_al2015['Unnamed: 0']
    del data_al2016['Unnamed: 0']
    del data_al2017['Unnamed: 0']
    del data_al2018['Unnamed: 0']

    # Escolhendo apenas as colunas de interesse
    data_al2014 = data_al2014.loc[:,'NT_GER':'QE_I26']
    data_al2015 = data_al2015.loc[:,'NT_GER':'QE_I26']
    data_al2016 = data_al2016.loc[:,'NT_GER':'QE_I26']
    data_al2017 = data_al2017.loc[:,'NT_GER':'QE_I26']
    data_al2018 = data_al2018.loc[:,'NT_GER':'QE_I26']

    data_al2014 = data_al2014.drop(data_al2014.loc[:, 'CO_RS_I1':'CO_RS_I9'].columns, axis=1)
    data_al2015 = data_al2015.drop(data_al2015.loc[:, 'CO_RS_I1':'CO_RS_I9'].columns, axis=1)
    data_al2016 = data_al2016.drop(data_al2016.loc[:, 'CO_RS_I1':'CO_RS_I9'].columns, axis=1)
    data_al2017 = data_al2017.drop(data_al2017.loc[:, 'CO_RS_I1':'CO_RS_I9'].columns, axis=1)
    data_al2018 = data_al2018.drop(data_al2018.loc[:, 'CO_RS_I1':'CO_RS_I9'].columns, axis=1)

    data_al2014 = data_al2014.drop(data_al2014.loc[:, 'NT_FG':'NT_CE_D3'].columns, axis=1)
    data_al2015 = data_al2015.drop(data_al2015.loc[:, 'NT_FG':'NT_CE_D3'].columns, axis=1)
    data_al2016 = data_al2016.drop(data_al2016.loc[:, 'NT_FG':'NT_CE_D3'].columns, axis=1)
    data_al2017 = data_al2017.drop(data_al2017.loc[:, 'NT_FG':'NT_CE_D3'].columns, axis=1)
    data_al2018 = data_al2018.drop(data_al2018.loc[:, 'NT_FG':'NT_CE_D3'].columns, axis=1)
    
    # MERGE NOS DADOS: data al
    frames = [data_al2014, data_al2015, data_al2016, data_al2017, data_al2018];
    data_al = pd.concat(frames);

    # Enriquecimento
    data_al['NT_GER'] = data_al['NT_GER'].str.replace(',','.')
    data_al['NT_GER'] = data_al['NT_GER'].astype(float)

    data_al_media = round(data_al['NT_GER'].mean(),2)
    
    data_al['NT_GER'] = data_al['NT_GER'].fillna(data_al_media)
    
    dataset_al = data_al
    
    describe_al = data_al.describe()
    
    # 3 - Transformação
    labels_al = np.array(data_al['NT_GER'])

    # Removendo as features de notas
    data_al = data_al.drop(['NT_GER'], axis = 1)
    
    features_al_list = list(data_al.columns)


    # One hot encoding - QE_I01 a QE_I26
    features_al = pd.get_dummies(data=data_al, columns=['QE_I01','QE_I02','QE_I03','QE_I04',
                                                        'QE_I05','QE_I06','QE_I07','QE_I08',
                                                        'QE_I09','QE_I10','QE_I11','QE_I12',
                                                        'QE_I13','QE_I14','QE_I15','QE_I16',
                                                        'QE_I17','QE_I18','QE_I19','QE_I20',
                                                        'QE_I21','QE_I22','QE_I23','QE_I24',
                                                        'QE_I25','QE_I26'])
    # Salvando os nomes das colunas (features) com os dados para uso posterior
    # depois de codificar
    features_al_list_oh = list(features_al.columns)
    #
    # Convertendo para numpy
    features_al = np.array(features_al)
    
    return features_al, labels_al, features_al_list_oh, dataset_al

#%% Aplicando o pré-processamento

features_al, labels_al, features_al_list_oh, dataset_al = processing_set_al(data_al2014, data_al2015, data_al2016, data_al2017, data_al2018)

#%% Percentual de valores NaN

nan_al = float(nan_2014+nan_2015+nan_2016+nan_2017+nan_2018);

column_al = float(labels_al.shape[0]);

per_al = nan_al/column_al;

print("Qtde. % NaN values in NT_GER AL:", 100*round(per_al, 4));
#%% Dados estatísticos gerais
# < 20 --> Amostra
# >= 20 --> população
import statistics as stats

min_al = min(labels_al)
max_al = max(labels_al)
mean_al = stats.mean(labels_al)
median_al = stats.median(labels_al)
variance_al = stats.variance(labels_al)
std_dev_al = stats.stdev(labels_al)

print("Min:", round(min_al, 4))
print("Max:", round(max_al, 4))
print("Media:", round(mean_al, 4))
print("Mediana:", round(median_al, 4))
print("Variancia:", round(variance_al, 4))
print("Desvio padrao: ", round(std_dev_al, 4))

#%% Escrevendo em arquivo

fields_stats_al = ['Version',
                   'Media',
                   'Mediana',
                   'Variancia',
                   'Desvio padrao',
                   'Max val',
                   'Min val',
                   '% Nan val']

rows_stats_al = {'Version':0,
                 'Media':mean_al,
                 'Mediana':median_al,
                 'Variancia':variance_al,
                 'Desvio padrao':std_dev_al,
                 'Max val':max_al,
                 'Min val':min_al,
                 '% Nan val': 100*per_al}

file_stats_al = "../tcc_codes/analise_stats/AL/Stats_AL.csv"

version_file(file_stats_al, fields_stats_al, rows_stats_al)
#%% Dataframe com Questionário do estudante
# Ref: https://pandas.pydata.org/docs/user_guide/advanced.html


df_qe_aux = pd.DataFrame([["QE_I01", "Estado civil","A:Solteiro(a)"],
          ["QE_I01", "Estado civil","B:Casado(a)"],
          ["QE_I01", "Estado civil","C:Separado(a)"],
          ["QE_I01", "Estado civil","D:Viúvo(a)"],
          ["QE_I01", "Estado civil","E:Outro"],
          
          ["QE_I02", "Cor ou raça", "A:Branca"],
          ["QE_I02", "Cor ou raça", "B:Preta"],
          ["QE_I02", "Cor ou raça", "C:Amarela"],
          ["QE_I02", "Cor ou raça", "D:Parda"],
          ["QE_I02", "Cor ou raça", "E:Indígena"],
          ["QE_I02", "Cor ou raça", "F:Não quero declarar"],
          
          ["QE_I03",  "Nacionalidade?","A:Brasileira"],
          ["QE_I03",  "Nacionalidade?","A:Brasileira naturalizada"],
          ["QE_I03",  "Nacionalidade?","A:Estrageira"],
          
          ["QE_I04", "Escolarização máxima do pai","A:Nenhuma"],
          ["QE_I04", "Escolarização máxima do pai","B:Fundamental: 1º ao 5º ano"],
          ["QE_I04", "Escolarização máxima do pai","C:Fundamental: 6º ao 9º ano"],
          ["QE_I04", "Escolarização máxima do pai","D:Ensino Médio"],
          ["QE_I04", "Escolarização máxima do pai","E:Ensino superior"],
          ["QE_I04", "Escolarização máxima do pai","F:Pós-graduação"],
          
          ["QE_I04", "Escolarização máxima do mãe","A:Nenhuma"],
          ["QE_I04", "Escolarização máxima do mãe","B:Fundamental: 1º ao 5º ano"],
          ["QE_I04", "Escolarização máxima do mãe","C:Fundamental: 6º ao 9º ano"],
          ["QE_I04", "Escolarização máxima do mãe","D:Ensino Médio"],
          ["QE_I04", "Escolarização máxima do mãe","E:Ensino superior"],
          ["QE_I04", "Escolarização máxima do mãe","F:Pós-graduação"],
          
          ["QE_I06", "Onde e com quem o estudante mora","A:Casa/apartamento,sozinho"],
          ["QE_I06", "Onde e com quem o estudante mora","B:Casa/apartamento,pais/parentes"],
          ["QE_I06", "Onde e com quem o estudante mora","C:Casa/apartamento,cônjugue/filhos"],
          ["QE_I06", "Onde e com quem o estudante mora","D:Casa/apartamento/república,com outras pessoas"],
          ["QE_I06", "Onde e com quem o estudante mora","E:Alojamento na IES"],
          ["QE_I06", "Onde e com quem o estudante mora","F:Outros (hotel,hospedaria,etc)"],
          
          ["QE_I07", "Pessoas da família que moram com o estudante","A:Nenhuma"],
          ["QE_I07", "Pessoas da família que moram com o estudante","B:Uma"],
          ["QE_I07", "Pessoas da família que moram com o estudante","C:Duas"],
          ["QE_I07", "Pessoas da família que moram com o estudante","D:Três"],
          ["QE_I07", "Pessoas da família que moram com o estudante","E:Quatro"],
          ["QE_I07", "Pessoas da família que moram com o estudante","F:Cinco"],
          ["QE_I07", "Pessoas da família que moram com o estudante","G:Seis"],
          ["QE_I07", "Pessoas da família que moram com o estudante","H:Sete ou mais"],
          
          ["QE_I08", "Renda total da família (incluindo rendimento do estudante)",
           "A:Até R$1.431,00"],
          ["QE_I08", "Renda total da família (incluindo rendimento do estudante)",
           "B:De R$1.431,01 a R$2.862,00"],
          ["QE_I08", "Renda total da família (incluindo rendimento do estudante)",
           "C:De R$2.862,01 a R$4.293,00"],
          ["QE_I08", "Renda total da família (incluindo rendimento do estudante)",
           "D:De R$4.293,01 a R$5.724,00"],
          ["QE_I08", "Renda total da família (incluindo rendimento do estudante)",
           "E:De R$5.274,01 a R$9.540,00"],
          ["QE_I08", "Renda total da família (incluindo rendimento do estudante)",
           "F:De R$9.540,01 a R$28.620,00"],
          ["QE_I08", "Renda total da família (incluindo rendimento do estudante)",
           "G:Mais de R$28.620,00"],
          
          ["QE_I09", "Situação financeira atual","A:Sem renda;gastos por programas do governo"],
          ["QE_I09", "Situação financeira atual","B:Sem renda;gastos pela família ou outras pessoas"],
          ["QE_I09", "Situação financeira atual","C:Com renda;recebo auxílio de outras pessoas"],
          ["QE_I09", "Situação financeira atual","D:Com renda;não preciso de ajuda financeira"],
          ["QE_I09", "Situação financeira atual","E:Com renda;ajudo a família"],
          ["QE_I09", "Situação financeira atual","F:Responsável pelo sustento da família"],
          
          ["QE_I10", "Situação atual de trabalho","A:Não trabalho"],
          ["QE_I10", "Situação atual de trabalho","B:Trabalho eventualmente"],
          ["QE_I10", "Situação atual de trabalho","C:Até 20h/semana"],
          ["QE_I10", "Situação atual de trabalho","D:De 21h/semana a 39h/semana"],
          ["QE_I10", "Situação atual de trabalho","E:Pelo menos 40h/semana"],
          
          ["QE_I11", "Tipo de financiamento recebido p/ custeio das mensalidades",
           "A:Nenhum;curso gratuito"],
          ["QE_I11", "Tipo de financiamento recebido p/ custeio das mensalidades",
           "B:Nenhum;curso pago"],
          ["QE_I11", "Tipo de financiamento recebido p/ custeio das mensalidades",
           "C:ProUni integral"],
          ["QE_I11", "Tipo de financiamento recebido p/ custeio das mensalidades",
           "D:ProUni parcial,apenas"],
          ["QE_I11", "Tipo de financiamento recebido p/ custeio das mensalidades",
           "E:FIES,apenas"],
          ["QE_I11", "Tipo de financiamento recebido p/ custeio das mensalidades",
           "F:ProUni parcial e FIES"],
          ["QE_I11", "Tipo de financiamento recebido p/ custeio das mensalidades",
           "G:Bolsa pelo estado,governo ou município"],
          ["QE_I11", "Tipo de financiamento recebido p/ custeio das mensalidades",
           "H:Bolsa pela IES"],
          ["QE_I11", "Tipo de financiamento recebido p/ custeio das mensalidades",
           "I:Bolsa por outra entidade"],
          ["QE_I11", "Tipo de financiamento recebido p/ custeio das mensalidades",
           "J:Financiamento pela IES"],
          ["QE_I11", "Tipo de financiamento recebido p/ custeio das mensalidades",
           "K:Financiamento bancário"],
          
          ["QE_I12", "Tipo de bolsa de permanência durante da graduação","A:Nenhum"],
          ["QE_I12", "Tipo de bolsa de permanência durante da graduação","B:Aux.moradia"],
          ["QE_I12", "Tipo de bolsa de permanência durante da graduação","C:Aux.alimentação"],
          ["QE_I12", "Tipo de bolsa de permanência durante da graduação","D:Aux.moradia e alimentaçãos"],
          ["QE_I12", "Tipo de bolsa de permanência durante da graduação","E:Aux.permanência"],
          ["QE_I12", "Tipo de bolsa de permanência durante da graduação","F:Outro"],
          
          ["QE_I13", "Tipo de bolsa acadêmica durante a graduação","A:Nenhum"],
          ["QE_I13", "Tipo de bolsa acadêmica durante a graduação","B:PIBIC"],
          ["QE_I13", "Tipo de bolsa acadêmica durante a graduação","C:Extensão"],
          ["QE_I13", "Tipo de bolsa acadêmica durante a graduação","D:Monitoria/tutoria"],
          ["QE_I13", "Tipo de bolsa acadêmica durante a graduação","E:PET"],
          ["QE_I13", "Tipo de bolsa acadêmica durante a graduação","F:Outro"],
          
          ["QE_I14", "Participação de programas e/ou atividades curriculares no exterior",
           "A:Não participei"],
          ["QE_I14", "Participação de programas e/ou atividades curriculares no exterior",
           "B:Sim;Ciência sem Fronteiras"],
          ["QE_I14", "Participação de programas e/ou atividades curriculares no exterior",
           "C:Sim;Finan.:Gov.Federal(Marca,PLI,outro)"],
          ["QE_I14", "Participação de programas e/ou atividades curriculares no exterior",
           "D:Sim;Finan.:Gov.Estadual"],
          ["QE_I14", "Participação de programas e/ou atividades curriculares no exterior",
           "E:Sim;Finan.:pela IES"],
          ["QE_I14", "Participação de programas e/ou atividades curriculares no exterior",
           "F:Sim;Outro não Institucional"],
          
          ["QE_I15", "Ingresso por ação afrimativa e critério","A:Não"],
          ["QE_I15", "Ingresso por ação afrimativa","B:Sim;étnico-racial"],
          ["QE_I15", "Ingresso por ação afrimativa","C:Sim;renda"],
          ["QE_I15", "Ingresso por ação afrimativa","D:Sim;esc.pública/bolsa esc. privada"],
          ["QE_I15", "Ingresso por ação afrimativa","E:Sim;2 ou mais critérios anteriores"],
          ["QE_I15", "Ingresso por ação afrimativa","F:Sim;outro critério"],
          
          ["QE_I16", "Unidade da Federação que concluiu o ensino médio","11:Rondônia"],
          ["QE_I16", "Unidade da Federação que concluiu o ensino médio","12:Acre"],
          ["QE_I16", "Unidade da Federação que concluiu o ensino médio","13:Amazonas"],
          ["QE_I16", "Unidade da Federação que concluiu o ensino médio","14:Roraima"],
          ["QE_I16", "Unidade da Federação que concluiu o ensino médio","15:Pará"],
          ["QE_I16", "Unidade da Federação que concluiu o ensino médio","16:Amapá"],
          ["QE_I16", "Unidade da Federação que concluiu o ensino médio","17:Tocantins"],
          ["QE_I16", "Unidade da Federação que concluiu o ensino médio","21:Maranhão"],
          ["QE_I16", "Unidade da Federação que concluiu o ensino médio","22:Piauí"],
          ["QE_I16", "Unidade da Federação que concluiu o ensino médio","23:Ceará"],
          ["QE_I16", "Unidade da Federação que concluiu o ensino médio","24:Rio Grande do Norte"],
          ["QE_I16", "Unidade da Federação que concluiu o ensino médio","25:Paraíba"],
          ["QE_I16", "Unidade da Federação que concluiu o ensino médio","26:Pernambuco"],
          ["QE_I16", "Unidade da Federação que concluiu o ensino médio","27:Alagoas"],
          ["QE_I16", "Unidade da Federação que concluiu o ensino médio","28:Sergipe"],
          ["QE_I16", "Unidade da Federação que concluiu o ensino médio","29:Bahia"],
          ["QE_I16", "Unidade da Federação que concluiu o ensino médio","31:Minas Gerais"],
          ["QE_I16", "Unidade da Federação que concluiu o ensino médio","33:Espírito Santo"],
          ["QE_I16", "Unidade da Federação que concluiu o ensino médio","35:São Paulo"],
          ["QE_I16", "Unidade da Federação que concluiu o ensino médio","41:Paraná"],
          ["QE_I16", "Unidade da Federação que concluiu o ensino médio","42:Santa Catarina"],
          ["QE_I16", "Unidade da Federação que concluiu o ensino médio","43:Rio Grande do Sul"],
          ["QE_I16", "Unidade da Federação que concluiu o ensino médio","50:Mato Grosso do Sul"],
          ["QE_I16", "Unidade da Federação que concluiu o ensino médio","51:Mato Grosso"],
          ["QE_I16", "Unidade da Federação que concluiu o ensino médio","52:Goiás"],
          ["QE_I16", "Unidade da Federação que concluiu o ensino médio","53:Distrito Federal"],
          ["QE_I16", "Unidade da Federação que concluiu o ensino médio","99:Não se aplica"],
          
          
          ["QE_I17", "Tipo de escola que cursou o ensino médio","A:Todo em pública"],
          ["QE_I17", "Tipo de escola que cursou o ensino médio","B:Todo em particular"],
          ["QE_I17", "Tipo de escola que cursou o ensino médio","C:Todo no exterior"],
          ["QE_I17", "Tipo de escola que cursou o ensino médio","D:Maior parte em pública"],
          ["QE_I17", "Tipo de escola que cursou o ensino médio","E:Maior parte em particular"],
          ["QE_I17", "Tipo de escola que cursou o ensino médio","F:Brasil e exterior"],
          
          
          ["QE_I18", "Modalidade de ensino médio","A:Tradicional"], 
          ["QE_I18", "Modalidade de ensino médio","B:Profissionalizante técnico"], 
          ["QE_I18", "Modalidade de ensino médio","C:Profissionalizante magistério"], 
          ["QE_I18", "Modalidade de ensino médio","D:EJA e/ou Supletivo"], 
          ["QE_I18", "Modalidade de ensino médio","E:Outro"], 
          
          ["QE_I19", "Pessoa que mais incentivou a cursar a graduação","A:Ninguém"], 
          ["QE_I19", "Pessoa que mais incentivou a cursar a graduação","B:Pais"], 
          ["QE_I19", "Pessoa que mais incentivou a cursar a graduação","C:Outros membros da família"], 
          ["QE_I19", "Pessoa que mais incentivou a cursar a graduação","D:Professores"], 
          ["QE_I19", "Pessoa que mais incentivou a cursar a graduação","E:Líder religioso"], 
          ["QE_I19", "Pessoa que mais incentivou a cursar a graduação","F:Colegas/amigos"], 
          ["QE_I19", "Pessoa que mais incentivou a cursar a graduação","G:Outras pessoas"],
          
          ["QE_I20", "Grupo(s) decisivo que o ajudou a enfrentar dificuldades durante o curso",
           "A:Sem dificuldades"],
          ["QE_I20", "Grupo(s) decisivo que o ajudou a enfrentar dificuldades durante o curso",
           "B:Não recebi apoio"],
          ["QE_I20", "Grupo(s) decisivo que o ajudou a enfrentar dificuldades durante o curso",
           "C:Pais"],
          ["QE_I20", "Grupo(s) decisivo que o ajudou a enfrentar dificuldades durante o curso",
           "D:Avós"],
          ["QE_I20", "Grupo(s) decisivo que o ajudou a enfrentar dificuldades durante o curso",
           "E:Irmãos/primos/tios"],
          ["QE_I20", "Grupo(s) decisivo que o ajudou a enfrentar dificuldades durante o curso",
           "F:Líder religioso"],
          ["QE_I20", "Grupo(s) decisivo que o ajudou a enfrentar dificuldades durante o curso",
           "G:Colegas de curso/amigos"],
          ["QE_I20", "Grupo(s) decisivo que o ajudou a enfrentar dificuldades durante o curso",
           "I:Profissionais de apoio da IES"],
          ["QE_I20", "Grupo(s) decisivo que o ajudou a enfrentar dificuldades durante o curso",
           "J:Colegas de trabalho"],
          ["QE_I20", "Grupo(s) decisivo que o ajudou a enfrentar dificuldades durante o curso",
           "K:Outro"],
          
          ["QE_I21", "Alguém em sua família concluiu um curso superior","A:Sim"],
          ["QE_I21", "Alguém em sua família concluiu um curso superior","B:Não"],
          
          ["QE_I22", "Livros lido no ano do Enade (excluindo a bibliografia do curso)",
           "A:Nenhum"],
          ["QE_I22", "Livros lido no ano do Enade (excluindo a bibliografia do curso)",
           "B:Um/Dois"],
          ["QE_I22", "Livros lido no ano do Enade (excluindo a bibliografia do curso)",
           "C:Três a cinco"],
          ["QE_I22", "Livros lido no ano do Enade (excluindo a bibliografia do curso)",
           "D:Oito a doze"],
          ["QE_I22", "Livros lido no ano do Enade (excluindo a bibliografia do curso)",
           "E:Mais de doze"],
          
          ["QE_I23", "Horas de estudo por semana (excluindo aulas)","A:Nenhuma"],
          ["QE_I23", "Horas de estudo por semana (excluindo aulas)","B:Uma a três"],
          ["QE_I23", "Horas de estudo por semana (excluindo aulas)","C:Quatro a sete"],
          ["QE_I23", "Horas de estudo por semana (excluindo aulas)","D:Oito a doze"],
          ["QE_I23", "Horas de estudo por semana (excluindo aulas)","E:Mais de doze"],
          
          ["QE_I24", "Oportunidade de aprendizado de idioma estrangeiro na IES",
           "A:Sim;apenas presencial"],
          ["QE_I24", "Oportunidade de aprendizado de idioma estrangeiro na IES",
           "B:Sim;apenas semipresencial"],
          ["QE_I24", "Oportunidade de aprendizado de idioma estrangeiro na IES",
           "C:Sim;presencial e semipresencial"],
          ["QE_I24", "Oportunidade de aprendizado de idioma estrangeiro na IES",
           "D:Sim;EAD"],
          ["QE_I24", "Oportunidade de aprendizado de idioma estrangeiro na IES",
           "E:Não"],
          
          ["QE_I25", "Principal motivo de escolha do curso",
           "A:Inserção no mercado de trabalho"],
          ["QE_I25", "Principal motivo de escolha do curso",
           "B:Influência familiar"],
          ["QE_I25", "Principal motivo de escolha do curso",
           "C:Valorização profissional"],
          ["QE_I25", "Principal motivo de escolha do curso",
           "D:Prestígio social"],
          ["QE_I25", "Principal motivo de escolha do curso",
           "E:Vocação"],
          ["QE_I25", "Principal motivo de escolha do curso",
           "F:Porque é EAD"],
          ["QE_I25", "Principal motivo de escolha do curso",
           "G:Baixa concorrência no ingresso"],
          ["QE_I25", "Principal motivo de escolha do curso",
           "H:Outro"],
          
          
          ["QE_I26", "Principal motivo de escolha da IES","A:Gratuidade"],
          ["QE_I26", "Principal motivo de escolha da IES","B:Preço da mensalidade"],
          ["QE_I26", "Principal motivo de escolha da IES","C:Prox. da residência"],
          ["QE_I26", "Principal motivo de escolha da IES","D:Prox. do trabalho"],
          ["QE_I26", "Principal motivo de escolha da IES","E:Facilidade de acesso"],
          ["QE_I26", "Principal motivo de escolha da IES","F:Qualidade/reputação"],
          ["QE_I26", "Principal motivo de escolha da IES","G:Foi a única onde ingressei"],
          ["QE_I26", "Principal motivo de escolha da IES","H:Possibilidade de bolsa"],
          ["QE_I26", "Principal motivo de escolha da IES","I:Outro"]],
                         columns=["cod_qe","description","alternatives"])
              
#tuples = list(zip(cod_qe, description))

df_qe = pd.MultiIndex.from_frame(df_qe_aux)
#%% Gaussiana com matplotlib da distribuição anterior
print("Matplotlib version: ",matplotlib.__version__)
print("Pandas version", pd.__version__)
font = {'family' : 'sans-serif',
        'weight' : 'normal',
        'size'   : 10}

matplotlib.rc('font', **font)

mu_al, std_al = norm.fit(labels_al) # Média e desvio padrão dos dados

# Histograma
plt.hist(labels_al, bins=150, density=True, alpha=0.0)
  
# Limites
min_ylim, max_ylim = plt.ylim(0,0.06)
xmin, xmax = plt.xlim(0,100)
x_al = np.linspace(xmin, xmax, 100)

# Normalizando (Gaussiana)
p_al = norm.pdf(x_al, mu_al, std_al)

# Plot Gaussiana
plt.plot(x_al, p_al, 'k', linewidth=1.5)# Ref: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html#examples-using-matplotlib-pyplot-plot
plt.fill_between(x_al, p_al, color='brown')# Ref: https://moonbooks.org/Articles/How-to-fill-an-area-in-matplotlib-/
plt.axvline(labels_al.mean(), color='k', linestyle='dashed', linewidth=1.5)
plt.text(labels_al.mean()*1.1, max_ylim*0.9, 'Média: {:.2f}'
         .format(labels_al.mean()))
plt.text(labels_al.mean()*1.1, max_ylim*0.83, 'Desvio padrão: {:.2f}'
         .format(labels_al.std()))
plt.title("Distribuição de notas do Enade em Alagoas: 2014 a 2018")
plt.xlabel('Notas do Enade');
plt.ylabel('Distribuição');
plt.savefig('../tcc_codes/analise_stats/AL/imagens/DIST_NOTA_AL.png', 
            dpi=150, bbox_inches='tight', pad_inches=0.015);
#%%
# Ref: https://dev.to/thalesbruno/subplotting-with-matplotlib-and-seaborn-5ei8
#fig, axes = plt.subplots(2, 3, figsize=(10,10))

#fig.suptitle('Distribuição de notas do Enade de Alagoas: QE_I02')

qe_i02 = dataset_al[["QE_I02", "NT_GER"]]

desc_qe_i02 = []

#plt.set_title(r'Distribuição de notas do Enade de 2014 a 2018: Alagoas - Categoria QE_I02')
# Dica: você deve estar na pasta tcc_codes (Variable explorer)
#plt.savefig('../tcc_codes/analise_stats/AL/imagens/QE_I02_AL.png', dpi=150, bbox_inches='tight', pad_inches=0.015);

#%% Subplots - Maior impacto
# Ref: https://tryolabs.com/blog/2017/03/16/pandas-seaborn-a-guide-to-handle-visualize-data-elegantly/
# Ref: https://www.python-graph-gallery.com/25-histogram-with-several-variables-seaborn
# QE_02
import matplotlib.colors as mcolors

df_qe_i02 = dataset_al[['QE_I02', 'NT_GER']]

df_qe_i02.hist(by="QE_I02", figsize=(7,7), layout=(3,2), bins=30,
               alpha=0.9, density=True, color="seagreen")

plt.suptitle('Distribuição de notas do Enade de Alagoas: QE_I02')
#plt.set_title(r'Distribuição de notas do Enade de 2014 a 2018: Alagoas - Categoria QE_I02')
# Dica: você deve estar na pasta tcc_codes (Variable explorer)
plt.savefig('../tcc_codes/analise_stats/AL/imagens/QE_I02_BOX_AL.png', dpi=150, bbox_inches='tight', pad_inches=0.015);

#%%
# Ref: https://tryolabs.com/blog/2017/03/16/pandas-seaborn-a-guide-to-handle-visualize-data-elegantly/
# Ref: https://www.python-graph-gallery.com/25-histogram-with-several-variables-seaborn
# QE_02

g_qei02 = sns.FacetGrid(dataset_al, col="QE_I02", height=2, col_wrap=3, ylim=(0,0.7))
g_qei02.map(sns.kdeplot, "NT_GER")

#sns.displot(data=dataset_al, x="NT_GER", hue="QE_I02", kind="kde")
#plt.show()
#plt.title("QE_I02: ...");
#plt.xlabel('Notas do Enade');
#plt.ylabel('Distribuição');
#plt.legend();
# Dica: você deve estar na pasta tcc_codes (Variable explorer)
plt.savefig('../tcc_codes/analise_stats/AL/imagens/QE_I02_KDE2_AL.png', dpi=150, bbox_inches='tight', pad_inches=0.015);



#%%
# QE_08

# QE_11

# QE_13

# QE_17

# QE_18

# QE_23

#%% Subplots - Menor impacto
# QE_01

# QE_03

# QE_12

# QE_15

# QE_16

# QE_18

# QE_19

# QE_21