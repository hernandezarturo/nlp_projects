#!/usr/bin/env python
# coding: utf-8

# # Taller 5 TFIDF Solucion
# 

# # Punto 1: Pre-Procesamiento
# 
# Leer el archivo `Princesas.csv` usando `pandas` y crear una nueva columna con el texto en minúscula, sin caracteres especiales ni números, sin palabras vacias y hacer stemming de las palabras

# In[336]:


import re
import numpy as np
import pandas as pd

from nltk.corpus import stopwords
stopwords_sp = stopwords.words('spanish')

from nltk.stem.snowball import SnowballStemmer
spanishStemmer=SnowballStemmer("spanish")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# In[337]:


data = pd.read_csv('C:/Python/NLP/nlp_projects/Clase_cinco/Princesas.csv', encoding = 'UTF-8')
data 


# In[338]:


data['procesado']= data['Personalidad'].apply(lambda fila: fila.lower())
data['procesado']= data['procesado'].apply(lambda fila: re.sub(r"[\W\d_]+"," ",fila))
data['procesado']= data['procesado'].apply(lambda fila: fila.strip())
data['procesado']= data['procesado'].apply(lambda fila: fila.split())
data['procesado']= data['procesado'].apply(lambda fila: [palabra for palabra in fila if palabra not in stopwords_sp])
data['procesado']= data['procesado'].apply(lambda fila: [spanishStemmer.stem(palabra) for palabra in fila])
data['procesado']= data['procesado'].apply(lambda fila:" ".join(fila))
data


# # Punto 2: TF-IDF
# 
# Crear la matriz TF-IDF

# In[339]:


from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vect = TfidfVectorizer()
tfidf = tfidf_vect.fit_transform(data.procesado.values)

tfidf_matrix = pd.DataFrame(data=tfidf.toarray(), columns=tfidf_vect.get_feature_names())

tfidf_matrix = tfidf_matrix.T.round(3)
tfidf_matrix.columns = data['Princesa']

tfidf_matrix


# # Punto 3: Distancia del coseno
# 
# - Calcular la distancia del coseno entre cada una de las princesas
# - ¿Cuáles son las princesas más parecidas?
# - ¿Cuáles son las princesas más diferentes?

# In[356]:


from sklearn.metrics.pairwise import cosine_distances

dist_cos = cosine_distances(tfidf_matrix.T.values)
dist_cos = pd.DataFrame(dist_cos, columns = tfidf_matrix.columns, index = tfidf_matrix.columns)
dist_cos


# In[480]:


dist_cos.max()


# In[519]:


for col in dist_cos:
    print(col)
    print(max(dist_cos[col]))
    print(" ")


# **Las princesas mas parecidas son:**
# 
# - Blancanieves y Merida
# - Cenienta y Moana
# - Aurora y Moana
# - Bella y Tiana
# - Jasmin y Merida
# - Pocahontas y Merida
# - Mulan y Tiana
# - Tiana y Mulan
# - Merida y Mulan
# - Moana y Cenicienta
# 
# **Las princesas menos parecidas son:**
# 
# - Blancanieves y Aurora
# - Cenicienta y Blancanieves
# - Aurora y Bella
# - Bella y Aurora
# - Jasmín y Bella
# - Pocahontas y Aurora
# - Mulan y Bella
# - Tiana y Cenicienta
# - Mérida y Aurora
# - Moana y Bella
# 

# In[ ]:




