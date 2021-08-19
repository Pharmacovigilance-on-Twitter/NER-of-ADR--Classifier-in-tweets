# -*- coding: utf-8 -*-
#NER OF ADR- Classifier in tweets.ipynb


#Original file is located at https://colab.research.google.com/drive/1nXOJ6U4UCJWjvsH6q-XffCubGpQpouDC


#Instalar e importar bibliotecas**
!pip install sklearn_crfsuite

!pip3 install dnspython
import dns

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from itertools import chain

import nltk
import sklearn
#crf - algoritmos de aprendizados de maquina 
import scipy.stats
#modulos do sklearn
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score, RandomizedSearchCV, train_test_split

import sklearn_crfsuite
from sklearn_crfsuite import scorers, metrics

from joblib import dump, load
#criar o arquivo do modelo

from pymongo import MongoClient #access MongoDB

import pandas as pd

"""**Conectar ao MongoDB**"""

uri = "mongodb+srv://you-link-conection"
db = MongoClient(uri, connectTimeoutMS=300000).get_database('your-Bio-tokens-database')
collection = db.get_collection('your-bio-tokens-collection')

#**Selecionar campos da coleção**"""

df_bio_schema = pd.DataFrame(list(collection.find({},{'_id':0}).sort("_id")))

df_bio_schema.head()


#**Modificar a estrutura dos dados da coleção**

#função para criar uma array onde tem a word + tag
def f(x):
  return [x[1], x[2]]

df_bio_schema['word+tag'] = df_bio_schema.apply(f, axis=1)

df_bio_schema.head()


#**Agrupar por sentença**
df_bio_schema_concat_sentence = df_bio_schema.groupby('sentence')['word+tag'].apply(lambda x: list(x))

df_bio_schema_concat_sentence.head()



#**Algoritmo CRF**

#Definir características utilizadas no algoritmo***


def word2features(sent, i):    
    word = sent[i][0]    
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),        
    }
    if i > 0:
        word1 = sent[i-1][0]        
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),            
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1][0]                
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),            
        })
    else:
        features['EOS'] = True

    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, label in sent]

def sent2tokens(sent):
    return [token for token, label in sent]

sent2features(list(df_bio_schema_concat_sentence)[0])[3]



#Separar documentos: treinamento e teste

# Commented out IPython magic to ensure Python compatibility.
# %%time
# train_sents, test_sents = train_test_split(list(df_bio_schema_concat_sentence), test_size=0.2)


train_sents[0]


#Treinar algoritmo

# Commented out IPython magic to ensure Python compatibility.
# %%time
# crf = sklearn_crfsuite.CRF(
#     algorithm='lbfgs',
#     c1=0.1,
#     c2=0.1,
#     max_iterations=100,
#     all_possible_transitions=True
# )
# crf.fit(X_train, y_train)



#Verficar desempenho
labels = list(crf.classes_)
labels.remove('O', 'I-Drug')
labels


y_pred = crf.predict(X_test)

metrics.flat_f1_score(y_test, y_pred, average='weighted', labels=labels)


# group B and I results
sorted_labels = sorted(labels, key=lambda name: (name[1:], name[0]))
print(metrics.flat_classification_report(y_test, y_pred, labels=sorted_labels, digits=3))


#Salvar algoritmo no Google Drive

from google.colab import drive
drive.mount('drive')
dump(crf, '/content/drive/MyDrive/MyFolder/' + 'arq_name.joblib')
