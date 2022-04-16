#!/usr/bin/env python
# coding: utf-8

# In[108]:


import os
import urllib
import tarfile


# **create function to fetch data download and load it**

# **defining the data's place**

# In[109]:



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


# In[110]:


#creating directory and download the data file 
fetch_housing_data()


# In[111]:


import pandas as pd
#reading data
def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path,"housing.csv")
    return pd.read_csv(csv_path)


# ### Exploratory the  data 

# In[112]:


# read our dataframe as housing
housing = load_housing_data()
# take a quick view at the data
housing.head()


# In[113]:


# quick descreption of the data 
housing.info()


# > looks like we have 207 missing value in bedrooms feature
# 
# > just ocean_proximity not float

# In[114]:


# about ocean_proximity's values
housing['ocean_proximity'].value_counts()


# In[115]:


# summarty of numerical attributes
housing.describe()


# In[116]:


# quick exploratory of the data
from jupyterthemes import jtplot
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
housing.hist(bins = 50, figsize = (20,15));
jtplot.style()


# In[117]:


# creating a test set 
from sklearn.model_selection import train_test_split
                                    # The data , test size , random_seed(42) for fixed test set
train_set, test_set = train_test_split(housing, test_size = 0.2, random_state=42)


# **Creating median income categories**

# In[118]:


import numpy as np 
housing['income_cat'] = pd.cut(housing['median_income'],
                              bins=[0.,1.5,3.0,4.5,6.,np.inf],
                              labels=[1, 2, 3, 4, 5])


# In[119]:


plt.hist(housing['income_cat'])
plt.title('income categories')


# **now we can do stratified sampling**

# In[120]:


from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state = 42)
for train_index, test_index in split.split(housing, housing['income_cat']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]


# In[121]:


# proportion of each category in incomes
strat_test_set['income_cat'].value_counts() / len(strat_test_set)


# In[122]:


# remove categories 
for set_ in (strat_train_set, strat_test_set):
    set_.drop('income_cat', axis=1, inplace=True)


# ### Explonatory the data 

# **Discvoer and visualize The data**
# 

# In[123]:


#but test set aside 
housing = strat_train_set.copy()


# In[124]:


housing.plot(kind = 'scatter', x = 'longitude', y='latitude', alpha=0.1);


# **add housing price and the population features to above scatter will give us more useful insights**

# In[125]:


housing.plot(kind='scatter', x='longitude', y='latitude', alpha =0.4,
            s=housing['population']/100,
             label='population', figsize=(10,7),
            c='median_house_value',cmap=plt.get_cmap('jet'), colorbar=True)
plt.legend();


# **The colors above telss us red = expinsive hous, blue = cheap | large_indicates = Large population**
# 
# **we can see corrolation here between houses close to ocean and high price**

#   > **looking for correlations** 

# In[126]:


corr_matrix = housing.corr()


# In[127]:


corr_matrix['median_house_value'].sort_values(ascending=True)


# **here we can see positive correlation with median_house_value and median_income**  
# **in other words: when income incrase the house values incrase**
# 

# In[128]:


housing.plot(kind='scatter', x='median_house_value', y='median_income', alpha=0.8, figsize=(10,7));


# In[129]:


from pandas.plotting import scatter_matrix
# check for correlation bettwen these specific attributes against each of them
attributes = ['median_house_value', 'median_income', 'total_rooms', 'housing_median_age']
scatter_matrix(housing[attributes], figsize=(12,8));


# <br/>**now we see the most promising attribute to predict house price is "median_income"**  
# <br/> 

# > **Will Try some attribute combinations**

# In[130]:


# number of rooms per house hold
housing['rooms_per_household']= housing['total_rooms']/housing['households']
# compare num of bedrooms per total rooms
housing['bedrooms_per_room'] = housing['total_bedrooms']/housing['total_rooms']
# population per household
housing['population_per_household'] = housing['population']/housing['households']


# **lets look at the correlations again** 
# 

# In[131]:


corr_matrix = housing.corr()
corr_matrix['median_house_value'].sort_values(ascending=True)


# **there is negative correlation between bedrooms_per_room and median_house_value**  
# **rooms_per_household more informative than total_rooms**

# <br/>**sepearte predictors and lables**

# In[132]:


housing = strat_train_set.drop(['median_house_value'],axis=1)
housing_lables = strat_train_set['median_house_value'].copy()


# ### Data Cleaning

# **fill the missing data in total_bedrooms with the median**

# In[133]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')


# **exclude categorical attributes to compute the median** 

# In[134]:


housing_num = housing.drop('ocean_proximity', axis=1)


# In[135]:


imputer.fit(housing_num)


# In[136]:


# median of all features
imputer.statistics_


# **daeling with categorical feature**

# In[137]:


housing_cat = housing[['ocean_proximity']]
housing_cat.value_counts()


# **convert text to numbers**

# In[138]:


from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder()
housing_cat_encoded = cat_encoder.fit_transform(housing_cat)
housing_cat_encoded.toarray()


# In[139]:


cat_encoder.categories_


# ### Feature scalling 

# >**this below code copied from hands on machine learning [book by ageron]**

# **custom Transformer to add extra attributes**

# In[140]:


from sklearn.base import BaseEstimator, TransformerMixin

# column index
col_names = "total_rooms", "total_bedrooms", "population", "households"
rooms_ix, bedrooms_ix, population_ix, households_ix = [
    housing.columns.get_loc(c) for c in col_names] # get the column indices

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)


# **recover The data frame**

# In[141]:


housing_extra_attribs = pd.DataFrame(
    housing_extra_attribs,
    columns=list(housing.columns)+["rooms_per_household", "population_per_household"],
    index=housing.index)
housing_extra_attribs.head()


# **Transformation Pipelines**

# In[142]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])

housing_num_tr = num_pipeline.fit_transform(housing_num)


# In[143]:


housing_num_tr


# **apply our transformations to the data**

# In[144]:


from sklearn.compose import ColumnTransformer
num_attribs = list(housing_num)
cat_attribs = ['ocean_proximity']


# In[145]:



full_pipline = ColumnTransformer([
    ('num', num_pipeline, num_attribs),
    ('cat', OneHotEncodere(), cat_attribst)
])


# In[ ]:





# In[ ]:




