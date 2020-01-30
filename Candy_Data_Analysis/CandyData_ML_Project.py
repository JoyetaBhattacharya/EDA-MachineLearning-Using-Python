#!/usr/bin/env python
# coding: utf-8

# # Machine Lerning Study On Candy Data

# ## Table of Content
# 
# 1. [Problem Statement](#section1)<br>
# 2. [Data Loading and Description](#section2)<br>
# 3. [Exploratory Data Analysis](#section3)<br>
#    - 3.1 [Data Profiling](#section301)<br> 
#        - 3.1.1 [Understanding Dataset](#section3011)<br>
#        - 3.1.2 [PreProfiling](#section3012)<br>
#        - 3.1.3 [Data Analysis](#section3013)<br>
# 4. [Linear Regression](#section4)<br>
#     - 4.1 [Preparation of X and Y data (80:20 Training and Testing)](#section401)<br>
#        - 4.1.1 [Model Evaluation](#section4011)<br>
#     - 4.2 [Preparation of X and Y data (70:30 Training and Testing)](#section402)<br>
#        - 4.2.1 [Model Evaluation](#section4021)<br>
# 5. [Conclusion](#section5)<br>

# <a id=section1></a>
# ### 1. Problem Statement

# The purpose of this below evaluation is to deriving the most important attributes of a candy.

# <a id=section2></a>
# ### 2. Data Loading and Description

# The Candy data has 12 attributes, total 83 rows and 13 columns 
# 
# Below are the description of 12 attributes of the candy data, 
# 
# 1.  chocolate: Does it contain chocolate?
# 2.  fruity: Is it fruit flavored?
# 3.  caramel: Is there caramel in the candy?
# 4.  peanutalmondy: Does it contain peanuts, peanut butter or almonds?
# 5.  nougat: Does it contain nougat?
# 6.  crispedricewafer: Does it contain crisped rice, wafers, or a cookie component?
# 7.  hard: Is it a hard candy?
# 8.  bar: Is it a candy bar?
# 9.  pluribus: Is it one of many candies in a bag or box?
# 10. sugarpercent: The percentile of sugar it falls under within the data set.
# 11. pricepercent: The unit price percentile compared to the rest of the set.
# 12. winpercent: The overall win percentage according to 269,000 matchups.

# In[2]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


from sklearn import metrics

import numpy as np

# allow plots to appear directly in the notebook
get_ipython().run_line_magic('matplotlib', 'inline')


# Importing the Dataset

# In[3]:


Candy_data = pd.read_csv('https://raw.githubusercontent.com/insaid2018/Term-2/master/Projects/candy-data.csv')
Candy_data.head()


# What are the Features ?

# chocolate
# fruity
# caramel
# peanutyalmondy
# nougat
# crispedricewafer
# hard
# bar
# pluribus
# sugarpercent
# pricepercent

# What is the response ?

# winpercent

# <a id=section3></a>
# ### 3. Exploratory Data Analysis

# <a id=section301></a>
# #### 3.1 Data Profiling

# <a id=section301></a>
# ##### 3.1.1 Understanding Dataset

# In[4]:


Candy_data.shape   # Find out number of rows and column in the data frame. 


# In[5]:


Candy_data.columns # Name of the columns in the data frame. 


# In[6]:


Candy_data.describe()


# In[8]:


Candy_data.info()


# In[9]:


Candy_data.isnull().sum()


# <a id=section3012></a>
# ##### 3.1.2 PreProfiling

# In[10]:


import pandas_profiling


# In[11]:


pre_profile = pandas_profiling.ProfileReport(Candy_data)
pre_profile.to_file(output_file="Candy_before_preprocessing.html")


# <a id=section3013></a>
# ##### 3.1.3 Data Analysis

# - Using count plot

# In[13]:


feature_cols1 = ['chocolate', 'fruity', 'caramel', 'peanutyalmondy', 'nougat']

Candy_Df1 = Candy_data[feature_cols1]
Candy_Df1.head() 


# In[14]:


sns.countplot(x="variable", hue="value" , data=pd.melt(Candy_Df1)).set_title('Count plot for different features of candies.')


# In[15]:


feature_cols2 = ['crispedricewafer', 'hard', 'bar', 'pluribus']

Candy_Df2 = Candy_data[feature_cols2]
Candy_Df2.head() 


# In[16]:


sns.countplot(x="variable", hue="value" , data=pd.melt(Candy_Df2)).set_title('Count plot for different features of candies.')


# #### From this graph we can see that 
#   1. Most of the cadies are pluribus, meaning they are one of the many candies in the bag or box.
#   2. Most of the candies either contains chocolate or has a fruity flavor.

# In[17]:


Column_Sort=['competitorname','winpercent','chocolate', 'fruity']
Candy_Sort = Candy_data[Column_Sort]
result = Candy_Sort.sort_values(['winpercent'], ascending=0)
result['Rank']=result['winpercent'].rank(ascending=0) 
result


# - As per the above ranking of the data, it seems that, "ReeseÕs Peanut Butter cup" is in the most demanding candy among top 10 candies. Also it has been noticed that top 20 candies are mostly chocolate flavoured and except Starburst and Skittles original are Fruity flavoured.   

# - Correlation Of Data

# In[18]:


Candy_data.corr()


# In[17]:


sns.heatmap( Candy_data.corr(), annot=True );


# - As per the above heat map, we can see that the highest correlation with Winpercent, is for chocolate, peanutyalmondy and bar.
# - The highest correlation for Pricepercent is with chocolate and bar.
# - Chocolate has higher correlation with Bar, peanutyalmondy. 

# - Visualising Pairwise correlation

# In[19]:


sns.pairplot(Candy_data, x_vars=['sugarpercent','pricepercent'], y_vars='winpercent', height=5, aspect=1, kind='reg')


# In[20]:


sns.pairplot(Candy_data, x_vars=['winpercent','pricepercent'], y_vars='sugarpercent', height=5, aspect=1, kind='reg')


# As per the above plots there is no strong linear relationship among winpercent, pricepercent and sugarpercent.

# <a id=section4></a>
# ### 4. Linear Regression

# <a id=section401></a>
# ### 4.1 Preparation of X and Y data (80:20 Training and Testing)

# In[21]:


feature_cols = ['chocolate', 'fruity', 'caramel', 'peanutyalmondy', 'nougat', 'crispedricewafer', 'hard', 'bar', 'pluribus']
x = Candy_data[feature_cols]
y= Candy_data.winpercent 


# Preparation of x_train, y_train, x_test, y_test

# In[22]:


from sklearn.model_selection import train_test_split

def split(x,y):
    return train_test_split(x, y, test_size=0.20, random_state=1)


# In[23]:


x_train, x_test, y_train, y_test = split(x,y)
print('Train cases as below')
print('X_train shape: ',x_train.shape)
print('y_train shape: ',y_train.shape)
print('\nTest cases as below')
print('X_test shape: ',x_test.shape)
print('y_test shape: ',y_test.shape)


# In[24]:


def linear_reg( x, y, gridsearch = False):
    
    x_train, x_test, y_train, y_test = split(x,y)
    
    from sklearn.linear_model import LinearRegression
    linreg = LinearRegression()
    
    if not(gridsearch):
        linreg.fit(x_train, y_train) 

    else:
        from sklearn.model_selection import GridSearchCV
        parameters = {'normalize':[True,False], 'copy_X':[True, False]}
        linreg = GridSearchCV(linreg,parameters, cv = 10,refit = True)
        linreg.fit(x_train, y_train)                                                           # fit the model to the training data (learn the coefficients)
        print('Train cases as below')
        print('X_train shape: ',x_train.shape)
        print('y_train shape: ',y_train.shape)
        print('\nTest cases as below')
        print('X_test shape: ',x_test.shape)
        print('y_test shape: ',y_test.shape)
        print("Mean cross-validated score of the best_estimator : ", linreg.best_score_)  
        
        y_pred_test = linreg.predict(x_test)                                                   # make predictions on the testing set

        RMSE_test = np.sqrt(metrics.mean_squared_error(y_test, y_pred_test))                          # compute the RMSE of our predictions
        print('RMSE for the test set is {}'.format(RMSE_test))

    return linreg


# In[25]:


linreg = linear_reg(x_train,y_train)


# In[26]:


print('Intercept:',linreg.intercept_)          # print the intercept 
print('Coefficients:',linreg.coef_)


# In[27]:


#feature_cols.insert(0,'Intercept') # this was executed once to inser 'Intercept', more than execution of this code, insert 'Intercept that many times so commented out'
feature_cols


# In[28]:


coef = linreg.coef_.tolist() # storing coefficient as list in coef
coef


# In[29]:


#coef.insert(0, linreg.intercept_) # inserting the intercept value in coef list at 0 position
coef


# In[31]:


eq1 = zip(feature_cols, coef)

for c1,c2 in eq1:
    print(c1,c2)


# Linear Equation with 80:20 (Train and Test) data

# ***y = 37.94 + 17.95 * chocolate + 10.28 * fruity + 0.65 * caramel + 10.1 * peanutyalmondy + 13.18 * nougat + 
#       14.27 * crispedricewafer - 7.41 * hard - 5.19 * bar - 2.39 * pluribus***

# ### Using the Model for Prediction

# In[32]:


y_pred_train = linreg.predict(x_train) 
y_pred_test = linreg.predict(x_test)


# <a id=section4011></a>
# ### 4.1.1 Model Evaluation using metrics

# #### 1. Computing the MAE for our winprice predictions

# In[34]:


MAE_train = metrics.mean_absolute_error(y_train, y_pred_train)
MAE_test = metrics.mean_absolute_error(y_test, y_pred_test)


# In[35]:


print('MAE for training set is {}'.format(MAE_train))
print('MAE for test set is {}'.format(MAE_test))


# #### 2. Computing the MSE for our winprice predictions

# In[36]:


MSE_train = metrics.mean_squared_error(y_train, y_pred_train)
MSE_test = metrics.mean_squared_error(y_test, y_pred_test)


# In[37]:


print('MSE for training set is {}'.format(MSE_train))
print('MSE for test set is {}'.format(MSE_test))


# #### 3. Computing the RMSE  for our winprice predictions

# In[39]:


RMSE_train = np.sqrt( metrics.mean_squared_error(y_train, y_pred_train))
RMSE_test = np.sqrt(metrics.mean_squared_error(y_test, y_pred_test))


# In[40]:


print('RMSE for training set is {}'.format(RMSE_train))
print('RMSE for test set is {}'.format(RMSE_test))


# #### 4. Model Evaluation using Rsquared value

# In[41]:


yhat = linreg.predict(x_train)
SS_Residual = sum((y_train-yhat)**2)
SS_Total = sum((y_train-np.mean(y_train))**2)
r_squared = 1 - (float(SS_Residual))/SS_Total
adjusted_r_squared = 1 - (1-r_squared)*(len(y_test)-1)/(len(y_test)-x_test.shape[1]-1)
print(r_squared, adjusted_r_squared)


# In[43]:


VIF= 1/(1-r_squared)
VIF


# **Observations for (80:20)training and testing data**
# 1. We could see that the candies which contains chocolate is 17.95 percentage points higher in terms of winpercent compared to candies with no chocolate. 
# 2. We also observed that fruity taste has a relatively high positive coefficient which contradicts our correlation heatmap results wherein it is inversely proportional with winning percentage. This might be caused by multicollinearity since we also found out from the correlation heatmap that chocolate and fruity has a strong negative correlation.
# 3. We got the RMSE for test data 11.9 and RSquared value is 0.5268, so we can say that  52.68% of the variance of winpercent can be explained by the factors we have used. 
# 4. Also we got VIF 2.11 we can say that there is no multicolinearity present in the model.

# <a id=section402></a>
# ### 4.2 Preparation of X and Y data (70:30 Training and Testing)

# In[44]:


feature_cols2 = ['chocolate', 'fruity', 'caramel', 'peanutyalmondy', 'nougat', 'crispedricewafer', 'hard', 'bar', 'pluribus']
x = Candy_data[feature_cols2]
y= Candy_data.winpercent 


# In[45]:


from sklearn.model_selection import train_test_split

def split(x,y):
    return train_test_split(x, y, test_size=0.30, random_state=1)


# In[46]:


x_train, x_test, y_train, y_test = split(x,y)
print('Train cases as below')
print('X_train shape: ',x_train.shape)
print('y_train shape: ',y_train.shape)
print('\nTest cases as below')
print('X_test shape: ',x_test.shape)
print('y_test shape: ',y_test.shape)


# In[47]:


def linear_reg( x, y, gridsearch = False):
    
    x_train, x_test, y_train, y_test = split(x,y)
    
    from sklearn.linear_model import LinearRegression
    linreg = LinearRegression()
    
    if not(gridsearch):
        linreg.fit(x_train, y_train) 

    else:
        from sklearn.model_selection import GridSearchCV
        parameters = {'normalize':[True,False], 'copy_X':[True, False]}
        linreg = GridSearchCV(linreg,parameters, cv = 10,refit = True)
        linreg.fit(x_train, y_train)                                                           # fit the model to the training data (learn the coefficients)
        print('Train cases as below')
        print('X_train shape: ',x_train.shape)
        print('y_train shape: ',y_train.shape)
        print('\nTest cases as below')
        print('X_test shape: ',x_test.shape)
        print('y_test shape: ',y_test.shape)
        print("Mean cross-validated score of the best_estimator : ", linreg.best_score_)  
        
        y_pred_test = linreg.predict(x_test)                                                   # make predictions on the testing set

        RMSE_test = np.sqrt(metrics.mean_squared_error(y_test, y_pred_test))                          # compute the RMSE of our predictions
        print('RMSE for the test set is {}'.format(RMSE_test))

    return linreg


# In[48]:


linreg = linear_reg(x_train,y_train)


# In[49]:


print('Intercept:',linreg.intercept_)          # print the intercept 
print('Coefficients:',linreg.coef_)


# In[50]:


#feature_cols2.insert(0,'Intercept') # this was executed once to inser 'Intercept', more than execution of this code, insert 'Intercept that many times so commented out'
feature_cols2


# In[52]:


coef2 = linreg.coef_.tolist() # storing coefficient as list in coef
coef2


# In[53]:


coef2.insert(0, linreg.intercept_) # inserting the intercept value in coef list at 0 position
coef2


# In[54]:


eq2 = zip(feature_cols2, coef2)

for c1,c2 in eq2:
    print(c1,c2)


# Linear Equation with 70:30 (Train and Test) data
# 
# ***y = 40.74 + 13.41 * chocolate + 4.28 * fruity -5.59 * caramel + 10.2 * peanutyalmondy + 10.40 * nougat + 18.10 * crispedricewafer -1.751 * hard -5.63 * bar -3.10 * pluribus***

# ### Using the Model for Prediction
# 

# In[55]:


y_pred_train = linreg.predict(x_train) 
y_pred_test = linreg.predict(x_test)


# <a id=section4021></a>
# ### 4.2.1 Model Evaluation using metrics

# #### 1. Computing the MSE for our winprice predictions

# In[56]:


MAE_train = metrics.mean_absolute_error(y_train, y_pred_train)
MAE_test = metrics.mean_absolute_error(y_test, y_pred_test)


# In[58]:


print('MAE for training set is {}'.format(MAE_train))
print('MAE for test set is {}'.format(MAE_test))


# #### 2. Computing the MSE for our winprice predictions

# In[60]:


MSE_train = metrics.mean_squared_error(y_train, y_pred_train)
MSE_test = metrics.mean_squared_error(y_test, y_pred_test)


# In[61]:


print('MSE for training set is {}'.format(MSE_train))
print('MSE for test set is {}'.format(MSE_test))


# #### 3.Computing the RMSE for our winprice predictions

# In[62]:


RMSE_train = np.sqrt( metrics.mean_squared_error(y_train, y_pred_train))
RMSE_test = np.sqrt(metrics.mean_squared_error(y_test, y_pred_test))


# In[63]:


print('RMSE for training set is {}'.format(RMSE_train))
print('RMSE for test set is {}'.format(RMSE_test))


# #### 4.Model Evaluation using Rsquared value

# In[64]:


yhat = linreg.predict(x_train)
SS_Residual = sum((y_train-yhat)**2)
SS_Total = sum((y_train-np.mean(y_train))**2)
r_squared = 1 - (float(SS_Residual))/SS_Total
adjusted_r_squared = 1 - (1-r_squared)*(len(y_test)-1)/(len(y_test)-x_test.shape[1]-1)
print(r_squared, adjusted_r_squared)


# In[66]:


VIF=1/(1-r_squared)
VIF


# **Observations for (70:30)training and testing data**
# 1. We could see that the candies which contains chocolate is 13.41 percentage points higher in terms of winpercent compared to candies with no chocolate. 
# 2. It is interesting to see that fruity still has a positive coefficient. Crispedricewafer has higher coefficient. One more interesting finding we can observe is that the attribute caramel from a positive coefficient in the first regression model has a negative coefficient which contradicts the results of our correlation heatmap.
# 3. We got the RMSE for test data 13.25 and RSquared value is 0.5078, so we can say that 50.78% of the variance of winpercent can be explained by the factors we have used. 
# 4. Also we got VIF 2.03 we can say that there is no multicolinearity present in the model.

# <a id=section5></a>
# ### 5. Conclusion
# As per our evaluation with (80:20) training and testing data chocholate is the most important attribute of a candy followed by crispedricewafer, peanutyalmondy and nougat. 
