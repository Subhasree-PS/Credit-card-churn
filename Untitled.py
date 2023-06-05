#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import math
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore',category=UserWarning)


# In[10]:


pd.set_option('display.max_rows',5000000)
pd.set_option('display.max_columns',5000000)


# # Importing data set

# In[11]:


df=pd.read_csv('BankChurners.csv')


# In[12]:


df.shape


# In[13]:


df.head()


# # Data cleaning

# In[14]:


df.isna().sum()


# # Drop columns

# In[17]:


df.drop([df.columns[21],df.columns[22]],axis=1,inplace=True)
df.head(1)


# # EDA

# # Customer age

# In[18]:


sns.distplot(df['Customer_Age'])
plt.title('Credit Card Customer Age Distribution')


# Most of the customer age lies between 40 and 60

# # Gender

# In[19]:


df['Gender'].value_counts()


# In[23]:


plt.pie(df['Gender'].value_counts(), labels = ['Female', 'Male'], autopct='%1.1f%%',shadow = True)
plt.title('Proportion of Gender', fontsize = 16)
plt.show()


# # Existing VS Attired customers

# In[28]:


plt.pie(df['Attrition_Flag'].value_counts(), labels = ['Existing Customer', 'Attrited Customer'], shadow=True,
        autopct='%1.1f%%')
plt.title('Existing and Attrited Customer count', fontsize = 16)
plt.show()


# # Education level

# In[29]:


edu = df['Education_Level'].value_counts().to_frame('Counts') 
plt.figure(figsize = (8,8))
plt.pie(edu['Counts'], labels = edu.index, autopct = '%1.1f%%', shadow= True)
plt.title('Education Levels', fontsize = 18)
plt.show()


# # Existing Vs Attired customer by gender

# In[30]:


plt.figure(figsize=(10,6))
sns.countplot(x='Gender', hue='Attrition_Flag', data=df)
plt.title('Existing and Attrted Customers by Gender', fontsize=20)


# In[31]:


fig,(ax1,ax2)=plt.subplots(1,2,figsize=(15,15))

attrited_gender = df.loc[df["Attrition_Flag"] == "Attrited Customer", ["Gender"]].value_counts().tolist()
ax1.pie(x=attrited_gender, labels=["Male", "Female"], autopct='%1.1f%%', startangle=90)
ax1.set_title('Attrited Customer vs Gender', fontsize=16)

existing_gender=df.loc[df["Attrition_Flag"] == "Existing Customer", ["Gender"]].value_counts().tolist()
ax2.pie(x=existing_gender,labels=["Male","Female"],autopct='%1.1f%%', startangle=90)
ax2.set_title('Existing Customer vs Gender', fontsize=16)


# # Education level by existing vs attired customers 

# In[32]:


fig,(ax1,ax2)=plt.subplots(1,2,figsize=(15,15))

attrited_edu = df.loc[df["Attrition_Flag"] == "Attrited Customer", ["Education_Level"]].value_counts().tolist()
ax1.pie(x=attrited_edu, labels=['Graduate', 'Post-Graduate', 'College', 'Unknown', 'Uneducated',
                                     'Doctorate', 'High School'], autopct='%1.1f%%', startangle=90)
ax1.set_title('Attrited Customer vs Education Level', fontsize=16)

existing_edu = df.loc[df["Attrition_Flag"] == "Existing Customer", ["Education_Level"]].value_counts().tolist()
ax2.pie(x=existing_edu, labels=['Graduate', 'Post-Graduate', 'College', 'Unknown', 'Uneducated',
                                     'Doctorate', 'High School'], autopct='%1.1f%%', startangle=90)
ax2.set_title('Existing Customer vs Education Level', fontsize=16)


# # Education by gender

# In[33]:


fig,(ax1,ax2)=plt.subplots(1,2,figsize=(15,15))

attrited_eduprop = df.loc[df["Gender"] == "F", ["Education_Level"]].value_counts().tolist()
ax1.pie(x=attrited_eduprop, labels=['Graduate', 'Post-Graduate', 'College', 'Unknown', 'Uneducated',
                                     'Doctorate', 'High School'], autopct='%1.1f%%', startangle=90)
ax1.set_title('Female vs Education Level', fontsize=16)

existing_eduprop = df.loc[df["Gender"] == "M", ["Education_Level"]].value_counts().tolist()
ax2.pie(x=existing_eduprop, labels=['Graduate', 'Post-Graduate', 'College', 'Unknown', 'Uneducated',
                                     'Doctorate', 'High School'], autopct='%1.1f%%', startangle=90)
ax2.set_title('Male vs Education Level', fontsize=16)


# # Marital status by Existing VS attired

# In[34]:


fig,(ax1,ax2)=plt.subplots(1,2,figsize=(15,15))

attrited_mar = df.loc[df["Attrition_Flag"] == "Attrited Customer", ["Marital_Status"]].value_counts().tolist()
ax1.pie(x=attrited_mar, labels=['Married', 'Single', 'Unknown', 'Divorced'], autopct='%1.1f%%', startangle=90)
ax1.set_title('Attrited Customer vs Marital_Status', fontsize=16)

existing_mar = df.loc[df["Attrition_Flag"] == "Existing Customer", ["Marital_Status"]].value_counts().tolist()
ax2.pie(x=existing_mar, labels=['Married', 'Single', 'Unknown', 'Divorced'], autopct='%1.1f%%', startangle=90)
ax2.set_title('Existing Customer vs Marital_Status', fontsize=16)


# # Income category

# In[38]:


from collections import Counter


# In[39]:


fig,(ax1,ax2)=plt.subplots(1,2,figsize=(15,15))
count = Counter(df['Income_Category'])

attrited_inc = df.loc[df["Attrition_Flag"] == "Attrited Customer", ["Income_Category"]].value_counts().tolist()
ax1.pie(x=attrited_inc, labels=count, autopct='%1.1f%%', startangle=90)
ax1.set_title('Attrited Customer vs Income_Category', fontsize=16)

existing_inc = df.loc[df["Attrition_Flag"] == "Existing Customer", ["Income_Category"]].value_counts().tolist()
ax2.pie(x=existing_inc, labels=count, autopct='%1.1f%%', startangle=90)
ax2.set_title('Existing Customer vs Income_Category', fontsize=16)


# # Correlation

# In[46]:


f, ax = plt.subplots(figsize=(12, 8)) 
sns.heatmap(df.corr(), annot=True, cmap="hot") 
plt.show()


# # Preprocessing the data

# In[47]:


from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, average_precision_score, recall_score
from sklearn.model_selection import RandomizedSearchCV


# In[48]:


df_cat = df[['Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category']]
df_cat.head()


# In[49]:


df_num = df[['Customer_Age', 'Months_on_book', 'Total_Relationship_Count', 'Months_Inactive_12_mon', 'Contacts_Count_12_mon', 'Credit_Limit',
                      'Total_Revolving_Bal', 'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt', 'Total_Trans_Ct',
                      'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio']]
df_num.head()


# In[51]:


enc = OneHotEncoder()
df_categorical_enc = pd.DataFrame(enc.fit_transform(df_cat).toarray())
df_categorical_enc.head()


# In[53]:


df_comb = pd.concat([df_categorical_enc, df_num], axis=1)
df_comb.head()


# In[54]:


X = df_comb


# In[55]:


y = df['Attrition_Flag']


# In[56]:


le = LabelEncoder()
y = le.fit_transform(y)


# # Train and Test

# In[57]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


# In[58]:


scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[59]:


target_names = ['Attrited Customer', 'Existing Customer']


# In[60]:


parameters_randomforest = {'n_estimators':range(10,400,5), 'max_depth':range(2,8,2)}


# # Random Forest

# In[61]:


randomforest = RandomForestClassifier(class_weight = 'balanced')
clf_randomforest = RandomizedSearchCV(randomforest, parameters_randomforest, random_state=0)
clf_randomforest.fit(X_train, y_train)


# In[62]:


y_pred_randomforest = clf_randomforest.predict(X_test)


# In[63]:


average_precision_score(y_test, y_pred_randomforest), roc_auc_score(y_test, y_pred_randomforest)


# In[64]:


print(classification_report(y_test, y_pred_randomforest, target_names=target_names))


# In[65]:


parameters_gb = {'learning_rate':(0.1,0.01), 'n_estimators':range(10,400,5),
                'max_depth':range(2,8,2)
              }


# In[66]:


gb = GradientBoostingClassifier()

clf_gb = RandomizedSearchCV(gb, parameters_gb, random_state=0)

clf_gb.fit(X_train, y_train)


# In[67]:


y_pred_gb = clf_gb.predict(X_test)


# In[68]:


average_precision_score(y_test, y_pred_gb), roc_auc_score(y_test, y_pred_gb)


# In[69]:


sns.heatmap(confusion_matrix(y_test, y_pred_gb), annot=True)


# In[70]:


print(classification_report(y_test, y_pred_gb, target_names=target_names))


# In[71]:


clf_lg = LogisticRegression(C=0.5, penalty='l2',n_jobs=6, random_state=0)
clf_lg.fit(X_train, y_train)


# In[72]:


y_pred_lg = clf_lg.predict(X_test)


# In[73]:


average_precision_score(y_test, y_pred_lg), roc_auc_score(y_test, y_pred_lg)


# In[74]:


sns.heatmap(confusion_matrix(y_test, y_pred_lg), annot=True)


# In[75]:


print(classification_report(y_test, y_pred_lg, target_names=target_names))


# In[76]:


y_pred_all = (0.5*y_pred_gb) + (y_pred_randomforest*0.3) + (y_pred_lg*0.2)


# In[77]:


average_precision_score(y_test, y_pred_all), roc_auc_score(y_test, y_pred_all)


# In[ ]:




