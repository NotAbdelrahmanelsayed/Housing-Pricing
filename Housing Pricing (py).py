#!/usr/bin/env python
# coding: utf-8

# In[12]:


import os
import urllib
import tarfile


# **create function to fetch data download and load it**

# **defining the data's place**

# In[13]:



DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


# In[14]:


#creating directory and download the data file 
fetch_housing_data()


# In[15]:


import pandas as pd
#reading data
def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path,"housing.csv")
    return pd.read_csv(csv_path)


# ### Exploratory the  data 

# In[16]:


# read our dataframe as housing
housing = load_housing_data()
# take a quick view at the data
housing.head()


# In[17]:


# quick descreption of the data 
housing.info()


# > looks like we have 207 missing value in bedrooms feature
# 
# > just ocean_proximity not float

# In[18]:


# about ocean_proximity's values
housing['ocean_proximity'].value_counts()


# In[19]:


# summarty of numerical attributes
housing.describe()


# In[20]:


# quick exploratory of the data
from jupyterthemes import jtplot
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
housing.hist(bins = 50, figsize = (20,15));
jtplot.style()


# In[21]:


# creating a test set 
from sklearn.model_selection import train_test_split
                                    # The data , test size , random_seed(42) for fixed test set
train_set, test_set = train_test_split(housing, test_size = 0.2, random_state=42)


# **Creating median income categories**

# In[22]:


import numpy as np 
housing['income_cat'] = pd.cut(housing['median_income'],
                              bins=[0.,1.5,3.0,4.5,6.,np.inf],
                              labels=[1, 2, 3, 4, 5])


# In[23]:


plt.hist(housing['income_cat'])
plt.title('income categories')


# **now we can do stratified sampling**

# In[24]:


from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state = 42)
for train_index, test_index in split.split(housing, housing['income_cat']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]


# In[25]:


# proportion of each category in incomes
strat_test_set['income_cat'].value_counts() / len(strat_test_set)


# In[26]:


# remove categories 
for set_ in (strat_train_set, strat_test_set):
    set_.drop('income_cat', axis=1, inplace=True)


# ### Explonatory the data 

# **Discvoer and visualize The data**
# 

# In[27]:


#but test set aside 
housing = strat_train_set.copy()


# In[28]:


housing.plot(kind = 'scatter', x = 'longitude', y='latitude', alpha=0.1);


# **add housing price and the population features to above scatter will give us more useful insights**

# In[29]:


housing.plot(kind='scatter', x='longitude', y='latitude', alpha =0.4,
            s=housing['population']/100,
             label='population', figsize=(10,7),
            c='median_house_value',cmap=plt.get_cmap('jet'), colorbar=True)
plt.legend();


# **The colors above telss us red = expinsive hous, blue = cheap | large_indicates = Large population**
# 
# **we can see corrolation here between houses close to ocean and high price**

#   > **looking for correlations** 

# In[30]:


corr_matrix = housing.corr()


# In[32]:


corr_matrix['median_house_value'].sort_values(ascending=True)


# **here we can see positive correlation with median_house_value and median_income**  
# **in other words: when income incrase the house values incrase**
# 

# In[38]:


housing.plot(kind='scatter', x='median_house_value', y='median_income', alpha=0.8, figsize=(10,7));


# In[46]:


from pandas.plotting import scatter_matrix
# check for correlation bettwen these specific attributes against each of them
attributes = ['median_house_value', 'median_income', 'total_rooms', 'housing_median_age']
scatter_matrix(housing[attributes], figsize=(12,8));


# <br/>**now we see the most promising attribute to predict house price is "median_income"**  
# <br/> 

# > **Will Try some attribute combinations**

# In[54]:


# number of rooms per house hold
housing['rooms_per_household']= housing['total_rooms']/housing['households']
# compare num of bedrooms per total rooms
housing['bedrooms_per_room'] = housing['total_bedrooms']/housing['total_rooms']
# population per household
housing['population_per_household'] = housing['population']/housing['households']


# **lets look at the correlations again** 
# 

# In[56]:


corr_matrix = housing.corr()
corr_matrix['median_house_value'].sort_values(ascending=True)


# **there is negative correlation between bedrooms_per_room and median_house_value**  
# **rooms_per_household more informative than total_rooms**

# <br/>**sepearte predictors and lables**

# In[60]:


housing = strat_train_set.drop(['median_house_value'],axis=1)
housing_lables = strat_train_set['median_house_value'].copy()


# ### Data Cleaning

# **fill the missing data in total_bedrooms with the median**

# In[62]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')


# **exclude categorical attributes to compute the median** 

# In[63]:


housing_num = housing.drop('ocean_proximity', axis=1)


# In[64]:


imputer.fit(housing_num)


# In[68]:


# median of all features
imputer.statistics_


# to continiou

# In[ ]:




