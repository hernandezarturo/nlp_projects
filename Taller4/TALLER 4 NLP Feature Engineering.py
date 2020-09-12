#!/usr/bin/env python
# coding: utf-8

# In[2]:


import re
import pandas as pd

from nltk.corpus import stopwords
stopwords_sp = stopwords.words('spanish')

from sklearn.feature_extraction.text import CountVectorizer


# ## Punto 1: Pre-Procesamiento
# 
# - Leer el archivo `dialogos.csv` usando `pandas`
# - Crear una nueva columna con el texto en minúscula, sin caracteres especiales ni números y sin palabras vacias

# In[27]:


data = pd.read_csv('C:/Python/NLP/nlp_projects/Clase_cuatro/dialogos.csv', encoding = 'UTF-8')
data  


# Crear una nueva columna con el texto en minúscula, sin caracteres especiales ni números y sin palabras vacias

# In[51]:


#" ".join(data['Respuesta'].values)

#re.sub(r"[^\w\d_]+"," ",data).

data['procesado']= data['Locución'].apply(lambda fila: fila.lower())
data['procesado']= data['procesado'].apply(lambda fila: re.sub(r"[\W\d_]+"," ",fila))
data['procesado']= data['procesado'].apply(lambda fila: fila.strip())
data['procesado']= data['procesado'].apply(lambda fila: fila.split())
data['procesado']= data['procesado'].apply(lambda fila: [palabra for palabra in fila if palabra not in stopwords_sp])
data['procesado']= data['procesado'].apply(lambda fila:" ".join(fila))
data


# ## Punto 2: Representación vectorial
# 
# - Crear una bolsa de palabras (BoW) del corpus usando la columna pre-procesada
# - ¿Cuántas palabras hay en el vocabulario? (Usando la función de sklearn)

# In[52]:


count_vect = CountVectorizer(binary=True)
bow_rep = count_vect.fit_transform(data.procesado.values)
len(count_vect.vocabulary_)


# ## Punto 3: 🤔
# 
# - ¿En qué casos es buena idea tomar en la cuenta la frecuencia de las palabras para la bolsa de palabras?
# 
#   
# - **Respuesta** La frecuencia es importante cuando dentro del problema analítico, se requiere conocer la cantidad de veces que se repite una palabra. Por ejemplo, cuando nos interese conocer la frecuencia o repetición de una palabra en la radicación de peticiones o quejas. Con las cantidades reales del (total de palabras) podemos realizar estadisticas descriptivas o visualización como histogramas. Mientras que cuando no utilizamos la frecuencia de las palabras podemos obtener un contexto general del texto, pero si buscamos un patrón o tendencia no es posible realizarlo. 
# 
# 
# - ¿Cuándo es una mejor idea usar una bolsa de n-gramas en vez de una bolsa de palabras?
# 
# 
# - **Respuesta** Los n-gramas permiten construir contexto con las palabras. Por lo que cuando se requiere dar una respuesta dentro de un contexto, los n-gramas dentro de un modelo de aprendizaje son muy utiles en la construcción por ejemplo de bots de respuestas escritas para servicio al cliente o para generar textos predictivos. La bolsa de palabras es menos eficaz cuando se trata de respuestas predictivas. 
#         

# In[ ]:




