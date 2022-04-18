#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Exploratory-the--data" data-toc-modified-id="Exploratory-the--data-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Exploratory the  data</a></span></li><li><span><a href="#Explonatory-the-data" data-toc-modified-id="Explonatory-the-data-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Explonatory the data</a></span></li><li><span><a href="#Data-Cleaning" data-toc-modified-id="Data-Cleaning-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Data Cleaning</a></span></li><li><span><a href="#Feature-scalling" data-toc-modified-id="Feature-scalling-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Feature scalling</a></span></li><li><span><a href="#Select-and-Train-The-model" data-toc-modified-id="Select-and-Train-The-model-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Select and Train The model</a></span></li><li><span><a href="#Fine-Tune-our-model" data-toc-modified-id="Fine-Tune-our-model-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Fine-Tune our model</a></span></li><li><span><a href="#Analyze-the-Best-Models-and-Their-Errors" data-toc-modified-id="Analyze-the-Best-Models-and-Their-Errors-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>Analyze the Best Models and Their Errors</a></span></li><li><span><a href="#Evaluate-the-final-model" data-toc-modified-id="Evaluate-the-final-model-8"><span class="toc-item-num">8&nbsp;&nbsp;</span>Evaluate the final model</a></span></li></ul></div>

# In[1]:


import os
import urllib
import tarfile


# **create function to fetch data download and load it**

# **defining the data's place**

# In[2]:



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


# In[3]:


#creating directory and download the data file 
fetch_housing_data()


# In[4]:


import pandas as pd
#reading data
def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path,"housing.csv")
    return pd.read_csv(csv_path)


# ### Exploratory the  data 

# In[5]:


# read our dataframe as housing
housing = load_housing_data()
# take a quick view at the data
housing.head()


# In[6]:


# quick descreption of the data 
housing.info()


# > looks like we have 207 missing value in bedrooms feature
# 
# > just ocean_proximity not float

# In[7]:


# about ocean_proximity's values
housing['ocean_proximity'].value_counts()


# In[8]:


# summarty of numerical attributes
housing.describe()


# In[9]:


# quick exploratory of the data
from jupyterthemes import jtplot
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
housing.hist(bins = 50, figsize = (20,15));
jtplot.style()


# In[10]:


# creating a test set 
from sklearn.model_selection import train_test_split
                                    # The data , test size , random_seed(42) for fixed test set
train_set, test_set = train_test_split(housing, test_size = 0.2, random_state=42)


# **Creating median income categories**

# In[11]:


import numpy as np 
housing['income_cat'] = pd.cut(housing['median_income'],
                              bins=[0.,1.5,3.0,4.5,6.,np.inf],
                              labels=[1, 2, 3, 4, 5])


# In[12]:


plt.hist(housing['income_cat'])
plt.title('income categories')


# **now we can do stratified sampling**

# In[13]:


from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state = 42)
for train_index, test_index in split.split(housing, housing['income_cat']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]


# In[14]:


# proportion of each category in incomes
strat_test_set['income_cat'].value_counts() / len(strat_test_set)


# In[15]:


# remove categories 
for set_ in (strat_train_set, strat_test_set):
    set_.drop('income_cat', axis=1, inplace=True)


# ### Explonatory the data 

# **Discvoer and visualize The data**
# 

# In[16]:


#but test set aside 
housing = strat_train_set.copy()


# In[17]:


housing.plot(kind = 'scatter', x = 'longitude', y='latitude', alpha=0.1);


# **add housing price and the population features to above scatter will give us more useful insights**

# In[18]:


housing.plot(kind='scatter', x='longitude', y='latitude', alpha =0.4,
            s=housing['population']/100,
             label='population', figsize=(10,7),
            c='median_house_value',cmap=plt.get_cmap('jet'), colorbar=True)
plt.legend();


# **The colors above telss us red = expinsive hous, blue = cheap | large_indicates = Large population**
# 
# **we can see corrolation here between houses close to ocean and high price**

#   > **looking for correlations** 

# In[19]:


corr_matrix = housing.corr()


# In[20]:


corr_matrix['median_house_value'].sort_values(ascending=True)


# **here we can see positive correlation with median_house_value and median_income**  
# **in other words: when income incrase the house values incrase**
# 

# In[21]:


housing.plot(kind='scatter', x='median_house_value', y='median_income', alpha=0.8, figsize=(10,7));


# In[22]:


from pandas.plotting import scatter_matrix
# check for correlation bettwen these specific attributes against each of them
attributes = ['median_house_value', 'median_income', 'total_rooms', 'housing_median_age']
scatter_matrix(housing[attributes], figsize=(12,8));


# <br/>**now we see the most promising attribute to predict house price is "median_income"**  
# <br/> 

# > **Will Try some attribute combinations**

# In[23]:


# number of rooms per house hold
housing['rooms_per_household']= housing['total_rooms']/housing['households']
# compare num of bedrooms per total rooms
housing['bedrooms_per_room'] = housing['total_bedrooms']/housing['total_rooms']
# population per household
housing['population_per_household'] = housing['population']/housing['households']


# **lets look at the correlations again** 
# 

# In[24]:


corr_matrix = housing.corr()
corr_matrix['median_house_value'].sort_values(ascending=True)


# **there is negative correlation between bedrooms_per_room and median_house_value**  
# **rooms_per_household more informative than total_rooms**

# <br/>**sepearte predictors and lables**

# In[25]:


housing = strat_train_set.drop(['median_house_value'],axis=1)
housing_lables = strat_train_set['median_house_value'].copy()


# ### Data Cleaning

# **fill the missing data in total_bedrooms with the median**

# In[26]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')


# **exclude categorical attributes to compute the median** 

# In[27]:


housing_num = housing.drop('ocean_proximity', axis=1)


# In[28]:


imputer.fit(housing_num)


# In[29]:


# median of all features
imputer.statistics_


# **daeling with categorical feature**

# In[30]:


housing_cat = housing[['ocean_proximity']]
housing_cat.value_counts()


# **convert text to numbers**

# In[31]:


from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder()
housing_cat_encoded = cat_encoder.fit_transform(housing_cat)
housing_cat_encoded.toarray()


# In[32]:


cat_encoder.categories_


# ### Feature scalling 

# >**this below code copied from hands on machine learning [book by ageron]**

# **custom Transformer to add extra attributes**

# In[33]:


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

# In[34]:


housing_extra_attribs = pd.DataFrame(
    housing_extra_attribs,
    columns=list(housing.columns)+["rooms_per_household", "population_per_household"],
    index=housing.index)
housing_extra_attribs.head()


# **Transformation Pipelines**

# In[35]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])

housing_num_tr = num_pipeline.fit_transform(housing_num)


# In[36]:


housing_num_tr


# **apply our transformations to the data**

# In[37]:


from sklearn.compose import ColumnTransformer

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])

housing_prepared = full_pipeline.fit_transform(housing)


# ### Select and Train The model 

# **First Test linear regression**

# In[38]:


from sklearn.linear_model import LinearRegression
lr_mod = LinearRegression()
lr_mod.fit(housing_prepared, housing_lables)


# **try the model in very small sample**

# In[39]:


some_data = housing.iloc[:5]
some_labels = housing_lables.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)
print('predictions : ', lr_mod.predict(some_data_prepared))


# In[40]:


print('the labels predicted', list(some_labels))


# **mesure model error using rmse on the hole data**

# In[41]:


from sklearn.metrics import mean_squared_error
housing_prediection = lr_mod.predict(housing_prepared)
lr_rmse = mean_squared_error(housing_lables, housing_prediection)
calc_rmse = np.sqrt(lr_rmse)
calc_rmse


# **Trying Desicion Tree Model**

# In[42]:


from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_lables)
housing_prediection = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_lables, housing_prediection)
tree_rmse = np.sqrt(tree_mse)
tree_rmse


# **there is 0.0 error can be perfect model ! let's try it on the test set**

# In[43]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(tree_reg, housing_prepared, housing_lables, 
                        scoring ='neg_mean_squared_error', cv = 10)
tree_rmse_scores = np.sqrt(-scores)


# **let's look at the result**

# In[44]:


def display_scores(scores) : 
    print('scores', scores)
    print('mean',scores.mean())
    print('std', scores.std())


# In[45]:


display_scores(tree_rmse_scores)


# **compute the same score for the linear regression model**

# In[46]:


linear_cross = cross_val_score(lr_mod, housing_prepared, housing_lables,
                              scoring='neg_mean_squared_error', cv = 10)
lr_rmse_scores = np.sqrt(-linear_cross)
display_scores(lr_rmse_scores)


# **obiviously linear regression model doing better than Desicion Tree model**

# > **let's try  RandomForestRegressor**

# In[47]:


from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared, housing_lables)


# In[48]:


housing_prediction = forest_reg.predict(housing_prepared)
forest_mse = mean_squared_error(housing_lables, housing_prediction)
forest_rmse = np.sqrt(forest_mse)
forest_rmse


# In[49]:


from sklearn.model_selection import cross_val_score

forest_scores = cross_val_score(forest_reg, housing_prepared, housing_lables,
                                scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)


# In[50]:


display_scores(forest_rmse_scores)


# **looks like rondom forest regressor is promising than to other model**
# 

# ### Fine-Tune our model 

# **Search for the best combination of hyperparamaeter values for the RandomForestRegression model**

# In[59]:


from sklearn.model_selection import GridSearchCV

param_grid = [
    # try 12 (3×4) combinations of hyperparameters
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    # then try 6 (2×3) combinations with bootstrap set as False
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
  ]

forest_reg = RandomForestRegressor(random_state=42)
# train across 5 folds, that's a total of (12+6)*5=90 rounds of training 
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error',
                           return_train_score=True)
grid_search.fit(housing_prepared, housing_lables
               )


# In[60]:


print('best parameters for the model is \n', grid_search.best_params_)


# In[61]:


grid_search.best_estimator_


# **evaluation score (best rmse score)**

# In[62]:


curves = grid_search.cv_results_
for mean_score, params in zip(curves['mean_test_score'], curves['params']) : 
    print(np.sqrt(-mean_score), params)


# ### Analyze the Best Models and Their Errors

# In[66]:


feature_importances = grid_search.best_estimator_.feature_importances_
feature_importances


# In[67]:


extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
#cat_encoder = cat_pipeline.named_steps["cat_encoder"] # old solution
cat_encoder = full_pipeline.named_transformers_["cat"]
cat_one_hot_attribs = list(cat_encoder.categories_[0])
attributes = num_attribs + extra_attribs + cat_one_hot_attribs
sorted(zip(feature_importances, attributes), reverse=True)


# ### Evaluate the final model

# In[70]:


final_model = grid_search.best_estimator_

X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

X_test_prepared = full_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)


# In[71]:


final_rmse


# In[ ]:





# 
