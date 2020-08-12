#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16


# In[2]:


module = VGG16(include_top=False, weights='imagenet')
module.summary()


# In[ ]:




