#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sagemaker import get_execution_role

role = get_execution_role()
bucket='sagemaker-jupiter-telecom-bucket'


# In[13]:


url = 'https://#############/jupiter-telecom-input-data.csv'

import urllib.request
res = urllib.request.urlopen(url)
data = [r.split(',') for r in res.read().decode('utf-8').split()[1:]]


# In[14]:


get_ipython().system('cat "iris_dnn_classifier.py"')


# In[ ]:
