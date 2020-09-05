#!/usr/bin/env python
# coding: utf-8

# # Procesamiento de Lenguage Natural
# 
# ## Taller #3: Pre-Procesamiento de Textos
# 

# In[1]:


import os


# In[2]:


os.getcwd()


# In[3]:


os.chdir("C:/Python/NLP/nlp_projects/Clase_tres/")


# In[4]:


os.getcwd()


# # Punto 1:
# 
# - Leer el archivo UNA_SOLITARIA_VOZ_HUMANA.txt con with open
# - Convertir a minúsculas
# - Conservar sólo palabras
# - Tokenizar

# In[5]:


##CARGANDO EL ARCHIVO USANDO OPEN
text = open('UNA_SOLITARIA_VOZ_HUMANA.txt', encoding = 'UTF-8', mode = 'r')


# In[6]:


##IMPRIMIENDO EL TEXTO

for line in text:
    print(line)


# In[90]:


##IMPRIMIR EN MINUSCULA

text = open('UNA_SOLITARIA_VOZ_HUMANA.txt', encoding = 'UTF-8', mode = 'r')
for line in text:
    print(line.lower())


# In[7]:


##IMPRIMIR SOLO LAS PALABRAS NO SIMBOLOS

import re # primero se importa la libreria para trabajar con Regex


# In[199]:


text = open('UNA_SOLITARIA_VOZ_HUMANA.txt', encoding = 'UTF-8', mode = 'r')
for line in text:
    print(re.sub(r"[^\w\d_]+"," ",line))


# In[127]:


## TOKENIZAR

text = open('UNA_SOLITARIA_VOZ_HUMANA.txt', encoding = 'UTF-8', mode = 'r')

data = text.read().lower()

for line in data:
    a = re.sub(r"[^\w\d_]+"," ",data)
    break
a.split()
    


# # Punto 2
# 
# - Quitar palabras vacias
# - ¿Cuáles son las 10 palabras no vacias más usadas?

# In[9]:


import nltk


# In[10]:


from nltk.corpus import stopwords
stopwords_sp = stopwords.words('spanish')


# In[126]:


text = open('UNA_SOLITARIA_VOZ_HUMANA.txt', encoding = 'UTF-8', mode = 'r')

data = text.read().lower()

for line in data:
    a = re.sub(r"[^\w\d_]+"," ",data).split()
    texto = [palabra for palabra in a if palabra not in stopwords_sp]
    break
texto
    


# In[125]:


from collections import Counter
comun = Counter(texto).most_common()
comun [0:10]


# # Punto 3:
# 
# - Stemming del documento
# - ¿Cuáles son las 10 raíces más usadas?

# In[46]:


from nltk.stem.snowball import SnowballStemmer
spanishStemmer = SnowballStemmer("spanish")


# In[124]:


raiz = []

for palabra in texto:
    steem = spanishStemmer.stem(palabra)
    raiz.append(steem)
raiz


# In[129]:


common_steem = Counter(raiz).most_common()
common_steem [0:10]


# # BONUS Punto 4:
# 
# - Contar cuántas ocurrencias hay por cada parte de la oración

# In[96]:


import es_core_news_sm
spacy_es = es_core_news_sm.load()


# In[128]:


componente = []

a = " ".join(texto)
b = spacy_es(a)

for palabra in b:
    componente.append(palabra.pos_)
Counter(componente).most_common()

