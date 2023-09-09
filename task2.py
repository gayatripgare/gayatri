#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
#
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px


# In[4]:


data=pd.read_csv("D:\\titanic dataset.csv",sep=",",encoding="utf-8")
data.head()


# In[5]:


data.info()


# In[6]:


data.describe()


# In[7]:


data.describe(include="object").T


# In[8]:


data.isnull().sum()


# In[9]:


plt.figure(figsize=(7,4))
sns.heatmap(data.isnull())


# In[10]:


data[data["Age"]<=0].shape


# In[11]:


data["PassengerId"].duplicated().sum()


# In[12]:


numerical_data = []
object_data = []

for column in data.columns:
    if data.dtypes[column] != 'object':
        numerical_data.append(column)
    else:
        object_data.append(column)


# In[13]:


numerical_data


# In[14]:


from sklearn. impute import KNNImputer


# In[15]:


imputer = KNNImputer(n_neighbors=5)


# In[16]:


data[numerical_data] = imputer.fit_transform(data[numerical_data])


# In[17]:


data.isnull().sum()


# In[18]:


for column in data.columns:
    missing_indices = data[data[column].isnull()].index
    available_values = data[column].dropna()
    
    for index in missing_indices:
        random_choice = np.random.choice(available_values)
        data.at[index, column] = random_choice


# In[19]:


data.isnull().sum()


# In[20]:


data['Fare']=data['Fare'].round(2)


# In[21]:


plt.figure(figsize=(7,4))
sns.heatmap(data.isnull())


# In[22]:


plt.figure(figsize=(7,5))
px.box(data_frame=data, y="Fare")


# In[23]:


sns.distplot(data["Fare"])
plt.show()


# In[24]:


#find the limits
upper_limit=data["Fare"].mean() + 3*data["Fare"].std()
lower_limit=data["Fare"].mean() - 3*data["Fare"].std()
print("upper limit: ",upper_limit)
print("lower limit: ",lower_limit)


# In[25]:


#find the outliers
outliers_df=data.loc[(data["Fare"]> upper_limit) |(data["Fare"] < lower_limit)]
outliers_df.shape


# In[26]:


#remove outliers from the data
new_df =data.loc[(data["Fare"]< upper_limit) & (data["Fare"] > lower_limit)]
print("before removing the outliers: ",len(data))
print("after removing the outliers: ",len(new_df))
print("the outliers: ",len(data)-len(new_df))


# In[27]:


plt.figure(figsize=(7,5))
px.box(data_frame=new_df, y="Fare")


# In[28]:


sns.distplot(new_df["Fare"])
plt.show()


# In[29]:


file_path="new df.csv"


# In[30]:


num_columns=list(data.drop(['Pclass','Survived'],axis=1).select_dtypes(include=['float','int']).columns)
num=int(len(num_columns)/2) if int(len(num_columns)/2)>1 else 2
fig ,ax = plt.subplots(num,num,figsize=(12,10))
for j in range(num):
    for i in range(num
):
        try:
            sns.histplot(data=df,x=num_columns[0],kde=True,bins=20,ax=ax[j][i])
            num_columns.pop(0)
        except:
            fig.delaxes(ax=ax[j][i])
fig.suptitle('Histograms of numerical columns', fontsize=16)
plt.show()


# In[31]:


cat_columns=list(data.select_dtypes(include=['object']).columns)
num=int(len(cat_columns)/2) if int(len(cat_columns)/2)>1 else 2
fig ,ax = plt.subplots(num,num,figsize=(12,10))
for j in range(num):
    for i in range(num
):
        try:
            sns.countplot(data=df,x=cat_columns[0],ax=ax[j][i],hue='Survived')
            cat_columns.pop(0)
        except:
            fig.delaxes(ax=ax[j][i])
fig.suptitle('Histograms of numerical columns', fontsize=16)
plt.show()


# In[32]:


num_columns=list(data.drop(['Pclass','Survived'],axis=1).select_dtypes(include=['float','int']).columns)
num=int(len(num_columns)/2) if int(len(num_columns)/2)>1 else 2
fig ,ax = plt.subplots(num,num,figsize=(12,10))
for j in range(num):
    for i in range(num
):
        try:
            sns.histplot(data=df,x=num_columns[0],kde=True,bins=20,ax=ax[j][i])
            num_columns.pop(0)
        except:
            fig.delaxes(ax=ax[j][i])
fig.suptitle('Histograms of numerical columns', fontsize=16)
plt.show()


# In[34]:


num_columns=list(data.drop(['Pclass','Survived'],axis=1).select_dtypes(include=['float','int']).columns)
num=int(len(num_columns)/2) if int(len(num_columns)/2)>1 else 2
fig ,ax = plt.subplots(num,num,figsize=(12,10))
for j in range(num):
    for i in range(num
):
        try:
            sns.boxplot(data=df,x=num_columns[0],ax=ax[j][i])
            num_columns.pop(0)
        except:
            fig.delaxes(ax=ax[j][i])
fig.suptitle('Histograms of numerical columns', fontsize=16)
plt.show()


# In[38]:


import scipy.stats as stats


# In[39]:


from sklearn import metrics


# In[40]:


from sklearn.metrics import classification_report


# In[47]:


get_ipython().run_line_magic('tensorflow_version', '1.x')
import tensflow as tf


# In[50]:


target=data.Survived


# In[51]:


age_groups = pd.cut(data['Age'], bins=range(0, 100, 10), include_lowest=True)
pd.crosstab(age_groups,data.Survived).plot(kind='bar')
plt.ylabel('No.of Passengers')
plt.title('Age over Survival')
plt.grid(color="red", linestyle=":", alpha=0.5)


# In[53]:


pd.crosstab(data.Sex,data.Survived).plot(kind='bar')
plt.ylabel('No.of Passengers')
plt.title('Sex over Survival')
plt.grid(color="red", linestyle=":", alpha=0.5)


# In[54]:


fare_groups = pd.cut(data['Age'], bins=range(0, 550, 10), include_lowest=True)
pd.crosstab(fare_groups,data.Survived).plot(kind='bar')
plt.ylabel('No.of Passengers')
plt.title('Fare over Survival')
plt.grid(color="red", linestyle=":", alpha=0.5)


# In[56]:


from sklearn import tree
model = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3)


# In[57]:


from sklearn import tree
model = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3)


# In[59]:


inputs=data[['Pclass','Sex','Age','Fare']]
inputs


# In[60]:


from sklearn.preprocessing import LabelEncoder
obj=LabelEncoder()
inputs['Sex']=obj.fit_transform(inputs['Sex'])
inputs.head(4)


# In[61]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(inputs,target, test_size=0.3)


# In[62]:


model.fit(x_train,y_train)


# In[63]:


model.score(x_test,y_test)


# In[64]:


model.predict([[2,1,22,100]])


# In[65]:


model.predict_proba([[2,1,22,100]])


# In[ ]:




