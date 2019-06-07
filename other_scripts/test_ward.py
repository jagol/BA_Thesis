
# coding: utf-8

# In[3]:


from scipy.cluster.hierarchy import dendrogram, linkage  
from matplotlib import pyplot as plt
import numpy as np


# In[10]:


import gensim


# In[11]:


mpath = '/home/pogo/Dropbox/UZH/BA_Thesis/code/pipeline/wiki.simple.bin'


# In[12]:


model = gensim.models.FastText.load_fasttext_format(mpath)


# In[14]:


words = ['profession', 'teacher', 'nurse', 'policeman', 'officer', 'researcher', 'worder', 'financier', 'trader', 'clerk']


# In[16]:


X = [model.wv[w] for w in words]


# In[4]:


X = np.array([[5,3],  
    [10,15],
    [15,12],
    [24,10],
    [30,30],
    [85,70],
    [71,80],
    [60,78],
    [70,55],
    [80,91],])


# In[24]:


linked = linkage(X, 'ward')


# In[25]:


labelList = words


# In[32]:


plt.figure(figsize=(100, 60))  


# In[31]:


dendrogram(linked,  
            orientation='top',
            labels=labelList,
            distance_sort='descending',
            show_leaf_counts=True)


# In[33]:


plt.show()

