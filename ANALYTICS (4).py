
# coding: utf-8

# In[5]:

get_ipython().magic('pylab inline')
import pandas as pd
import numpy as np
#comments

# In[6]:

train = pd.read_csv("B:/Case studies/Student Hunt/train.csv")
test = pd.read_csv("B:/Case studies/Student Hunt/test.csv")


# # VISUALIZATION

# In[7]:

# to check data
train.head()
test.head()


# In[8]:

# to understand the general statistics of data
train.describe()


# In[9]:

# data type and dimension of features
train.info()


# In[10]:

#  for checking null values
train.isnull().sum()


# In[11]:

train.boxplot(column='Count')


# In[12]:

train['Count'].hist(bins=10)


# In[13]:

Q1 = train['Count'].quantile(0.25)
Q3 = train['Count'].quantile(0.75)
IQR = Q3 - Q1


# In[14]:

mean_value = train.Count.mean()


lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

def imputer(value):
    if value < lower_limit or value > upper_limit:
        return mean_value
    else:
        return value

result = train.Count.apply(imputer)


# In[15]:

result.plot(kind='box')


# #The above result is almost free from outliers, but it might have lost the important information. let not use this result right now.

# In[16]:

train['Count'].describe()


# In[17]:

train.ID.dtype


# In[18]:

# Splitting ID in to number of features
train['ID1'] = train['ID'].astype(str)
train['Year'] = train['ID1'].str[0:4]
train['Month'] = train['ID1'].str[4:6]
train['Day'] = train['ID1'].str[6:8] 
train['Time'] = train['ID1'].str[8:10] 


# In[19]:

# splitting ID int to no of feature
test['ID1'] = test['ID'].astype(str)
test['Year'] = test['ID1'].str[0:4]
test['Month'] = test['ID1'].str[4:6]
test['Day'] = test['ID1'].str[6:8] 
test['Time'] = test['ID1'].str[8:10] 


# In[20]:

train.head()
test.head()


# In[21]:

train.info()


# In[22]:

# New feature DATE
train['date'] = train['Year']+'-'+ train['Month'] + '-' + train['Day']
test['date'] = test['Year']+'-'+ test['Month'] + '-' + test['Day']


# In[23]:

# converting date in to datetime type
train['date'] = pd.to_datetime(train['date'])
test['date'] = pd.to_datetime(test['date'])


# In[24]:

# To determine day of week and week number
train['day-of-week'] = train['date'].dt.weekday_name
test['day-of-week'] = test['date'].dt.weekday_name
train['weekno'] = train['date'].dt.week
test['weekno'] = test['date'].dt.week


# In[25]:

train.head(100)


# In[26]:

train.boxplot(column = 'Count',by = 'day-of-week')


# In[27]:

temp = train.pivot_table(values = 'Count',index = 'day-of-week',columns='Month',aggfunc='sum')
temp


# It can be seen From the above two outputs on saturday and sunday, The Number of count is much lower as compared to other week days.

# In[28]:

train.boxplot(column = 'Count',by = 'Year')


# In[111]:

train['day-of-week'] = train['day-of-week'].str.replace('Saturday','6')
train['day-of-week'] = train['day-of-week'].str.replace('Sunday','6')
train['day-of-week'] = train['day-of-week'].str.replace('Monday','1')
train['day-of-week'] = train['day-of-week'].str.replace('Tuesday','2')
train['day-of-week'] = train['day-of-week'].str.replace('Wednesday','3')
train['day-of-week'] = train['day-of-week'].str.replace('Thursday','4')
train['day-of-week'] = train['day-of-week'].str.replace('Friday','5')

test['day-of-week'] = test['day-of-week'].str.replace('Saturday','6')
test['day-of-week'] = test['day-of-week'].str.replace('Sunday','6')
test['day-of-week'] = test['day-of-week'].str.replace('Monday','1')
test['day-of-week'] = test['day-of-week'].str.replace('Tuesday','2')
test['day-of-week'] = test['day-of-week'].str.replace('Wednesday','3')
test['day-of-week'] = test['day-of-week'].str.replace('Thursday','4')
test['day-of-week'] = test['day-of-week'].str.replace('Friday','5')


# In[112]:

train['Year'] = train['Year'].astype('int')
test['Year'] = test['Year'].astype('int')


# In[113]:

train['old'] = 2014 - train['Year']
test['old'] = 2014 - test['Year']

# Cross validation Technique
# In[352]:

ctrain=train[:11400]
ctest =train[11400:]
X = ctrain[['old','Month','Time','day-of-week','weekno']]
Y = ctrain[['Count']]
Xtest = ctest[['old','Month','Time','day-of-week','weekno']]
# importing random forest
from sklearn.ensemble import RandomForestRegressor
Regressor=RandomForestRegressor(n_estimators= 300 ,max_features=4, min_samples_leaf=5,max_depth = 12, oob_score=True)
Regressor.fit(X,Y)
predicted= Regressor.predict(Xtest)
predicted=np.around(predicted)
from sklearn.metrics import mean_squared_error
from math import sqrt
rms = sqrt(mean_squared_error(ctest['Count'], predicted))


# In[353]:

print(Regressor.feature_importances_)


# In[354]:

rms


# In[355]:

# Final Modelling
X = train[['old','Month','Time','day-of-week','weekno']]
Y = train[['Count']]
Xtest = test[['old','Month','Time','day-of-week','weekno']]
from sklearn.ensemble import RandomForestRegressor
Regressor=RandomForestRegressor(n_estimators= 300 ,max_features=4, min_samples_leaf=5,max_depth = 12,oob_score=True)
# fitting training dataset
Regressor.fit(X,Y)
# prediction
predicted2= Regressor.predict(Xtest)
predicted2


# In[356]:

print(Regressor.feature_importances_)


# In[357]:

predicted2=np.around(predicted2)


# In[358]:

# Make submission file and submit
submission = pd.DataFrame({'Count':predicted2,'ID':test['ID']},columns=['ID','Count'])
submission.to_csv('av15.csv', index = False)


# In[ ]:



