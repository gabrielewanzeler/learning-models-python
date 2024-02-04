
#importa as biliotecas necessárias
import pandas as pd
import numpy as np
import seaborn as srn
import statistics as sts
import csv

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
import graphviz
from sklearn.ensemble import RandomForestClassifier
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from sklearn.preprocessing import MinMaxScaler

# Tratamento do dataset
data = pd.read_csv("/content/diabetes_prediction_dataset.csv",sep=",")

data.head()

data.info()

index = data.columns
data.shape

#imprime dados duplicados
duplicados = data.duplicated(
    subset = index,
    keep=False)
data[duplicados]

#exclui dados duplicados
data.drop_duplicates(
    subset = index,
    keep='first', inplace=True)

data.shape

#verifica se há idades maiores que 110
age = data['age']
list_idades = []
for idade in age:
    if idade>110:
        list_idades.append(idade)

len(list_idades)

#verifica os valores 'No_info' na coluna 'smoking_history'
col = data['smoking_history']
list_no_info = []
for valor in col:
    if valor=='No Info':
        list_no_info.append(valor)

len(list_no_info)

# encontra as linhas que contêm o valor 'No_info'
filtro = data.isin(['No Info'])

# Seleciona as linhas e colunas que contêm o valor 'No_info'
linhas_noInfo= data.loc[filtro.any(axis=1), filtro.any(axis=0)]

# Exibe as linhas e colunas que contêm o valor 'No_info'
linhas_noInfo

# Exclui as linhas que contêm o valor 'No_info'
data = data[~filtro.any(axis=1)]

#sequencia para verificar se há dados diferentes do esperado em cada célula
print(f"valores na coluna gender: {set(data['gender'])}")
print(f"valores na coluna hypertension: {set(data['hypertension'])}")
print(f"valores na coluna heart_disease: {set(data['heart_disease'])}")
print(f"valores na coluna smoking_history: {set(data['smoking_history'])}")
print(f"valores na coluna diabetes: {set(data['diabetes'])}")

#verifica os valores nas colunas 'age', 'blood_glucose_level', 'HbA1c_level', 'bmi' que não podem ser inteiros
texto = pd.to_numeric(data["age"], errors='coerce').isna()
print(f"valores na coluna age não numéricos: {data[texto].values}")

texto = pd.to_numeric(data["blood_glucose_level"], errors='coerce').isna()
print(f"valores na coluna 'blood_glucose_level' não numéricos: {data[texto].values}")

texto = pd.to_numeric(data["HbA1c_level"], errors='coerce').isna()
print(f"valores na coluna 'HbA1c_level' não numéricos: {data[texto].values}")

texto = pd.to_numeric(data["bmi"], errors='coerce').isna()
print(f"valores na coluna 'bmi' não numéricos: {data[texto].values}")

data = data.reset_index(drop=True)
data


labelencoder = LabelEncoder()

data['gender'] = labelencoder.fit_transform(data['gender'])
label_gender = labelencoder.classes_
data['smoking_history'] = labelencoder.fit_transform(data['smoking_history'])
label_smoking_history = labelencoder.classes_

data

# Graficos do dataset

#Gráfico de pessoas por gênero


# Conta o número de ocorrências de cada valor na coluna 'gender'
ocorrencias = data['gender'].value_counts()

total = ocorrencias.sum()
fem_pct = ocorrencias[0] / total * 100
male_pct = ocorrencias[1] / total * 100
other_pct = ocorrencias[2] / total * 100

porcentagens = [fem_pct, male_pct, other_pct]


# Cria uma lista com as cores para cada fatia
cores = ['tab:orange', 'tab:blue', 'gray']

# Cria o gráfico de pizza
plt.pie(porcentagens, labels=['Feminino', 'Masculino', 'Outro'], colors=cores, autopct='%1.1f%%')

# Adiciona um título ao gráfico
plt.title('Porcentagem de pessoas por gênero')

plt.show()

#Gráfico de pessoas com e sem diabetes

# Conta o número de ocorrências de cada valor na coluna 'diabetes'
ocorrencias = data['diabetes'].value_counts()

total = ocorrencias.sum()
sem_diabetes_pct = ocorrencias[0] / total * 100
com_diabetes_pct = ocorrencias[1] / total * 100

# Cria uma lista com as porcentagens
porcentagens = [sem_diabetes_pct, com_diabetes_pct]

# Cria uma lista com as cores para cada fatia
cores = ['Turquoise', 'Orange']


plt.pie(porcentagens, labels=['Sem Diabetes', 'Com Diabetes'], colors=cores, autopct='%1.1f%%')

plt.title('Porcentagem de pessoas com e sem diabetes')

plt.show()

#Gráfico de pessoas com diabetes por gênero

gender_diabeticos = data.loc[data['diabetes'] == 1, 'gender']


gender_diabeticos = gender_diabeticos.value_counts()

gender_diabeticos = gender_diabeticos.append(pd.Series([2]), ignore_index=True)

total = gender_diabeticos.sum()
diabeticos_fem = gender_diabeticos[0] / total * 100
diabeticos_male = gender_diabeticos[1] / total * 100
diabeticos_other = gender_diabeticos[2] / total * 100

porcentagens = [diabeticos_fem, diabeticos_male, diabeticos_other]

# Cria uma lista com as cores para cada fatia
cores = ['tab:purple', 'y']


plt.pie(porcentagens, labels=['Feminino', 'Maculino', 'Outro'], colors=cores, autopct='%1.1f%%')

# Adiciona um título ao gráfico
plt.title('Porcentagem de pessoas com diabetes por gênero')

plt.show()

"""Gráfico de pessoas com diabetes por faixa etária"""

#encontra as pessoas com diabetes
diabeticos = data.loc[data['diabetes'] == 1, 'age'].tolist()

# cria intervalos de idade
intervalos = pd.cut(diabeticos, bins=range(0, 81, 10))

# conta quantos valores estão em cada intervalo
contagem = intervalos.value_counts()

# plotar gráfico de barras
plt.bar(['0-10','10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80'], contagem.values.tolist())

# definir título e rótulos dos eixos
plt.title('Número de pessoas diabéticas por faixa etária')
plt.xlabel('Faixa etária')
plt.ylabel('Número de pessoas')
plt.xticks(rotation=45)
plt.show()

# Normaliza o Dataset e separa os previsores

# seleciona as colunas que serão normalizadas
col_p_normalizar = data[['age', 'bmi', 'HbA1c_level','blood_glucose_level']]
demais_col = pd.DataFrame(data[['gender', 'hypertension', 'smoking_history','heart_disease','diabetes']])

# normaliza as colunas selecionadas numericas
scaler = MinMaxScaler()
col_normalizada = scaler.fit_transform(col_p_normalizar)
col_normalizada = pd.DataFrame(col_normalizada)
col_normalizada.columns = ['age', 'bmi', 'HbA1c_level','blood_glucose_level']

#seleciona as colunas categoricas
demais_col = demais_col.reset_index(drop=True)
col_normalizada = col_normalizada.reset_index(drop=True)

#concatenas todas as colunas
data_normal = pd.concat([col_normalizada, demais_col], axis=1)
data_normal.head()

data_normal[data_normal.isna().any(axis=1)]

# Dados de treinamento e teste

#separa os previsores e classe

previsores = data_normal.iloc[:,0:8].values
classe = data_normal.iloc[:,8].values
print(previsores.shape)
print(classe.shape)

#separa as variaveis para teste e treinamento
X_treinamento,X_teste,y_treinamento,y_teste=train_test_split(previsores,classe,test_size=0.3,random_state=0)

# KNN

#cria o modelo knn e o treina
knn= KNeighborsClassifier(n_neighbors=4)
knn.fit(X_treinamento,y_treinamento)

#faz as previsoes com os dados de teste
previsao = knn.predict(X_teste)
y_scores = knn.predict_proba(X_treinamento)[:,1]
previsao

#cria a matriz de confusão
confusao=confusion_matrix(y_teste,previsao)
np.set_printoptions(precision=0, suppress=True)
labels = ['sem diabetes', 'com diabetes']
srn.heatmap(confusao, annot=True, fmt='g', cmap="Blues", xticklabels=labels, yticklabels=labels)

# Plota a matriz de confusão
plt.title("Matriz de confusão Modelo KNN")
plt.xlabel("Verdadeiras Classes")
plt.ylabel("Classe Previstas")
plt.show()
confusao

#obtem a acuracia das previsoes
acuracia_knn = accuracy_score(y_teste,previsao)
acuracia_knn

# Cria a curva ROC
fpr, tpr, thresholds = roc_curve(y_treinamento, y_scores)
roc_auc = auc(fpr, tpr)

#Plota a curva ROC
plt.plot(fpr, tpr, lw=1, alpha=0.8, label='ROC (AUC = %0.2f)' % (roc_auc))
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=0.8)
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Curva ROC')
plt.legend(loc="lower right")


# Arvore de decisão
#cria o modelo e o treina

arvore = DecisionTreeClassifier()
arvore.fit(X_treinamento,y_treinamento)

#faz as previsoes com os dados de teste
previsao = arvore.predict(X_teste)
previsao

#cria a matriz de confusão
confusao=confusion_matrix(y_teste,previsao)
confusao

#obtem a acuracia das previsoes
acuracia_arvore = accuracy_score(y_teste,previsao)
acuracia_arvore

# Plota a árvore de decisão
plt.figure(figsize=(10,10))
plot_tree(arvore, filled=True, feature_names=data.columns)
plt.show()

np.set_printoptions(precision=0, suppress=True)
srn.heatmap(confusao, annot=True, fmt='g', cmap="Blues")

# Plota a matriz de confusão
plt.title("Matriz de confusão Modelo Árvore de Decisão")
plt.xlabel("Verdadeiras Classes")
plt.ylabel("Classe Previstas")
plt.show()
confusao

# RandomForestClassifier

#cria o modelo e o treina
fl = RandomForestClassifier(n_estimators=500)
fl.fit(X_treinamento,y_treinamento)

#faz as previsoes com os dados de teste e a acuracia
previsores=fl.predict(X_teste)
acuracia_floresta =accuracy_score(y_teste,previsores)
acuracia_floresta

#cria a matriz de confusão
confusao=confusion_matrix(y_teste,previsores)
confusao

np.set_printoptions(precision=0, suppress=True)
srn.heatmap(confusao, annot=True, fmt='g', cmap="Blues")
# Plota a matriz de confusão
plt.title("Matriz de confusão Modelo Random Forest Classifier")
plt.xlabel("Verdadeiras Classes")
plt.ylabel("Classe Previstas")
plt.show()

colunas = data_normal.columns.tolist()
colunas.remove('diabetes')


importancia = fl.feature_importances_

# Criar um gráfico de barras mostrando os atributos de maior importancia
plt.figure(figsize=(15,6))
plt.bar(range(len(importancia)), importancia)
plt.xticks(range(len(importancia)), colunas)
plt.title('Gráfico do Grau de importância para cada variável')
plt.xlabel('Variáveis')
plt.ylabel('Importância')
plt.show()
importancia

# Rede Neural

#cria o modelo
modelo = Sequential()
modelo.add(Dense(units=8, activation='sigmoid', input_dim=8))
modelo.add(Dense(units=8, activation='sigmoid',))
modelo.add(Dense(units=1, activation='sigmoid'))

modelo.summary()

X_treinamento,X_teste,y_treinamento,y_teste=train_test_split(previsores,classe,test_size=0.3,random_state=0)

#treina o modelo
modelo.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

historico = modelo.fit(X_treinamento, y_treinamento, validation_data=(X_teste, y_teste), epochs=50, batch_size=15)


#faz as previsoes com os dados de teste
previsao = modelo.predict(X_teste)
previsao = (previsao >0.5)
previsao

#cria a matriz de confusão
confusao=confusion_matrix(y_teste,previsao)
confusao

np.set_printoptions(precision=0, suppress=True)
srn.heatmap(confusao, annot=True, fmt='g', cmap="Blues", xticklabels=labels, yticklabels=labels)
# Plota a matriz de confusão
plt.title("Matriz de confusão Modelo Redes Neurais")
plt.xlabel("Verdadeiras Classes")
plt.ylabel("Classe Previstas")
plt.show()

#verifica a acuracia das previsoes
acuracia_RN=accuracy_score(y_teste, previsao)
acuracia_RN

#plota o grafico do historico de treinamento
plt.plot(historico.history['accuracy'])
plt.plot(historico.history['val_accuracy'])
plt.title('Histórico de Treinamento')
plt.ylabel('Precisão')
plt.xlabel('Epoch')
plt.legend(['Treinamento', 'Teste'], loc='upper left')
plt.show()

# Resultados

precisoes = [acuracia_knn*100, acuracia_arvore*100,acuracia_floresta*100, acuracia_RN*100]
modelos = ['KNN', 'Árvore de Decisão', 'Foreist Random', 'Redes Neurais']
# Criar um gráfico de barras para a acuracia de cada modelo
plt.figure(figsize=(8,5))
plt.bar(range(len(precisoes)), precisoes)
plt.xticks(range(len(precisoes)), modelos)
plt.title('Gráfico de acurácia dos modelos')
plt.xlabel('Modelos')
plt.ylabel('Acurácia(%)')
plt.ylim([85, 100])
for i, v in enumerate(precisoes):
    plt.text(i, v + 1, str(round(v,2)), ha='center')

plt.show()