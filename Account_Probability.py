# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 14:02:40 2019
@author: chetanjawlae
"""
import pandas as pd
import datetime as dt
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import functools
from functools import reduce
from sqlalchemy import create_engine

# Training Data
# Account_data = Current Status of Account
# Signup_data = Sign_up form of Account
dataset1 = pd.read_csv('/Account_data.csv')
dataset2 = pd.read_csv('/Signup_data.csv', error_bad_lines=False)

#print(dataset1)
print(dataset2['Is account created'].value_counts())
dataset2 = dataset2[dataset2['Is account created'] == 'YES']
dataset1=dataset1.rename(columns = {'Account Name':'Company name'})

# Feature Engineering and Enhancing data
dfs = [dataset1, dataset2]
dataset = functools.reduce(lambda left,right: pd.merge(left,right,on='Company name'), dfs)
dataset['Website'] = (dataset.Website.notnull()).astype('category')    # 1 = NAN, # 0 = Website available

#%%
# Sorting Public Sector // Private Sector Companies
com_typ = dataset['Company name'].str.findall(r"\bPvt ltd\b|\bpvt limited\b|\bPVT.LTD\b|\bpvt. Ltd\b|\bPrivate limited\b|\bPvt Ltd\b|\bPrivate Limited\b|\bpvt. ltd.\b|\bPVT LTD\b|\bPRIVATE LIMITED\b|\bpvt ltd\b|\bPvt. Ltd\b|\bPvt. Ltd.\b|\bPvt Ltd.\b|\bPvt Limited\b|\bprivate limited\b")
com_typ = pd.DataFrame(com_typ.values)
com_typ[0] = com_typ[0].astype(str)
com_typ[0] = com_typ[0].str.replace(r"[",'')
com_typ[0] = com_typ[0].str.replace(r"]",'')

cm_t = com_typ[0].str.len()
cm_lt = []
for value in cm_t[:]:
    if value > 3:
        cm_lt.append('Pvt Ltd')
    else:
        cm_lt.append(None)

dataset['Company type'] = cm_lt
#%%
print(dataset['State'].value_counts().count())
dataset['State'].unique()
# Removing irrelevant or unnecessary and duplicate variables 
#%%
# Analyzing based on Outsourced Email Domain Provider // Or carries Personal Domain 
# EMail domain 
em = dataset['Email address'].str.findall(r"(?i)\@\w+")
em = pd.DataFrame(em.values.tolist())
em[0] = em[0].astype(str)
em[0] = em[0].str.replace(r"[",'')
em[0] = em[0].str.replace(r"]",'')
em[0]= em[0].str.lower()
em[0].value_counts()

domn = ['@gmail','@yahoo','@hotmail','@outlook','@rediffmail','@live']
    
em[1] = em[0].apply(lambda x : ['@gmail' in x,'@yahoo' in x,
       '@hotmail' in x,'@outlook' in x,'@rediffmail' in x,'@live' in x])

dataset['E-dom'] = em[1].apply(lambda x : False if True in x else True)    

dataset['E-dom'] = dataset['E-dom'].where((pd.notnull(dataset['E-dom'])), False)
#%%
# city
dataset['City'] = dataset['City'].str.lower()
#%%
dataset.drop(['Is account created','Step Reached','First name','Last name','Mobile number',
              'Zip code','Phone number','Terms & Conditions agreed','User limit',
              'Mail link expired on','Feature value','Mobile verified',
              'Email address verified'],axis=1,inplace=True)

dataset['Last Accessed Date'] = dataset['Last Accessed Date'].combine_first(dataset['Start Date'])

dataset['Last Accessed Date'] = pd.to_datetime(dataset['Last Accessed Date'])
dataset['Start Date'] = pd.to_datetime(dataset['Start Date'])

dataset['days'] = dataset['Last Accessed Date'] - dataset['Start Date']
dataset['days'] = dataset['days'].astype(str)
dataset['days'] = dataset['days'].apply(lambda x: x.rsplit(" ", -1)[0])
dataset['days'] = dataset['days'].astype(int)
dataset['days'][dataset['days'] < 0] = 0
#dataset['days'] = dataset['days'].astype(int)
dataset.dtypes

#dataset['days'].plot()
#dataset['days'].value_counts()[:10].plot(kind='bar')

# Binning Statistically in two categories based on data skewness
dataset['days'] = pd.cut(dataset.days,bins=[-1, 7, 1000],labels=[0,1])

################################
dataset['usersCount'] = pd.cut(dataset.usersCount,
                     bins=[-1, 2, 1000],
                     labels=[0,1])
dataset['usersCount'].value_counts()[:20].plot(kind='bar')

# Bucketing User Types :
		# NEVER LOGGED IN
		# Interested
		# PotentiAL PREMIUM
		# PREMIUM

dataset['activity Frequency'].describe()
dataset['activity Frequency'] = pd.cut(dataset['activity Frequency'],bins=[-1, 0, 1000],
       labels=[0, 1])

dataset.Status[dataset.Status == 'Active'] = 1
dataset.Status[dataset.Status == 'InActive'] = 0

#%%
#Categories
dataset['Designation'].value_counts()
dataset['Industry'].value_counts()

dataset.Designation[dataset.Designation == 'DIRECTOR'] = 'Director'
dataset.Designation[dataset.Designation == 'director'] = 'Director'
dataset.Designation[dataset.Designation == 'ceo'] = 'CEO'
dataset.Designation[dataset.Designation == 'Ceo'] = 'CEO'
dataset.Designation[dataset.Designation == 'Sr Manager'] = 'Senior Manager'
dataset.Designation[dataset.Designation == 'MD'] = 'Managing Director'
dataset.Designation[dataset.Designation == 'Proprietor'] = 'Owner'
dataset.Designation[dataset.Designation == 'proprietor'] = 'Owner'
dataset.Designation[dataset.Designation == 'Propriter'] = 'Owner'
dataset.Designation[dataset.Designation == 'system administrator'] = 'System Administrator'
dataset.Designation[dataset.Designation == 'manager'] = 'Manager'
dataset.Designation[dataset.Designation == 'MANAGER'] = 'Manager'
dataset.Designation[dataset.Designation == 'General Manager'] = 'Manager'
dataset.Designation[dataset.Designation == 'Manager IT'] = 'Manager'
dataset.Designation[dataset.Designation == 'Marketing Manager'] = 'Manager'
dataset.Designation[dataset.Designation == 'Operations Manager'] = 'Manager'
dataset.Designation[dataset.Designation == 'Hr Manager'] = 'Manager'
dataset.Designation[dataset.Designation == 'Business Manager'] = 'Manager'
dataset.Designation[dataset.Designation == 'Mangaer'] = 'Manager'
dataset.Designation[dataset.Designation == 'Investment Manager'] = 'Manager'
dataset.Designation[dataset.Designation == 'Associate Manager'] = 'Manager'
dataset.Designation[dataset.Designation == 'Assistant Manager'] = 'Manager'
dataset.Designation[dataset.Designation == 'Zonal Manager - Sales'] = 'Manager'
dataset.Designation[dataset.Designation == 'BDM'] = 'Manager'
dataset.Designation[dataset.Designation == 'GM'] = 'Manager'
dataset.Designation[dataset.Designation == 'DGM'] = 'Manager'
dataset.Designation[dataset.Designation == 'Operation Manager'] = 'Manager'
dataset.Designation[dataset.Designation == 'PM'] = 'Manager'
dataset.Designation[dataset.Designation == 'abm'] = 'Manager'
dataset.Designation[dataset.Designation == 'Senior Manager'] = 'Manager'
dataset.Designation[dataset.Designation == 'Sales Manager'] = 'Manager'
dataset.Designation[dataset.Designation == 'Branch Manager'] = 'Manager'
dataset.Designation[dataset.Designation == 'SENIOR MANAGER'] = 'Manager'
dataset.Designation[dataset.Designation == 'HR/Admin Manager'] = 'HR'
dataset.Designation[dataset.Designation == 'Hr'] = 'HR'
dataset.Designation[dataset.Designation == 'Asst.Manager - Human Resource'] = 'HR'
dataset.Designation[dataset.Designation == 'HR & IT'] = 'HR'
dataset.Designation[dataset.Designation == 'HR & Admin'] = 'HR'
dataset.Designation[dataset.Designation == 'Human Resource Manager'] = 'HR'
dataset.Designation[dataset.Designation == 'Human Resource'] = 'HR'
dataset.Designation[dataset.Designation == 'partner'] = 'Partner'
dataset.Designation[dataset.Designation == 'Co-Founder'] = 'Partner'
dataset.Designation[dataset.Designation == 'Director'] = 'Director / Managing Director / Proprietor'
dataset.Designation[dataset.Designation == 'Managing Director '] = 'Director / Managing Director / Proprietor'
dataset.Designation[dataset.Designation == 'Proprietor'] = 'Director / Managing Director / Proprietor'

#%%
# Rich Datapoints [ Explainatory Analysis ]

Percent_Count = (dataset['Designation'].value_counts() / 259 ) * 100
Percent_Count[:10].sum()  # [70%]
#%%
# Categories to other
otr = dataset['Industry'].value_counts()
dataset['Industry']=np.where(dataset['Industry'].isin(otr.index[otr <= 2]), 'OTHER', dataset['Industry'])

otr1 = dataset['Designation'].value_counts()
dataset['Designation']=np.where(dataset['Designation'].isin(otr1.index[otr1 <= 2]), 'OTHER', dataset['Designation'])

otr2 = dataset['City'].value_counts()
dataset['City']=np.where(dataset['City'].isin(otr2.index[otr2 <= 2]), 'OTHER', dataset['City'])

#%%
# Missing
dataset.isnull().sum()
dataset.usersCount = dataset['usersCount'].fillna(0)
dataset = dataset.where((pd.notnull(dataset)), 'Not Available')

#%%
# Adress
address = dataset['Address'].str.len()
dataset['Is_Address'] = pd.cut(address,
                     bins=[-1, 20, 50 ,1000],
                     labels=['False','TF','True'])

# Name
name = dataset['Full name'].str.len()
name.describe()
name_li = []
for value in name[:]:
    if value > 6:
        name_li.append(True)
    else:
        name_li.append(False)
print(name_li)

dataset['Is_Name'] = name_li

# Company Name
#cname = dataset['Company name'].str.len()

#cname_li = []
#for value in cname[:]:
#    if value > 6:
#        cname_li.append(1)
#    else:
#       cname_li.append(0)
#print(cname_li)

#dataset['Is_CName'] = cname_li

#%%
# test 1
# Convert possible features into categories to simpify analysis
#dataset['usersCount'] = d1['usersCount']                   # Continuous
#dataset['Last Accessed Date'] = d1['Last Accessed Date']   # Binary
#dataset['activity Frequency'] = d1['activity Frequency']   # Continuous
#dataset['Status'] = d1['Status']                           # Binary

for col in ['Gender','Designation','Country','Industry','Website','State','City','Source value','Device',
            'Newsletter opt in','Plan','Is_Address','Is_Name','usersCount',
            'activity Frequency','Status','days','Company type','E-dom']:
    dataset[col] = dataset[col].astype('category')

dataset.dtypes
# Arrange in order 'y = Status'
dataset = dataset[['Company name','Company type','Form filled on','Full name','Gender','Designation','Industry','Website',
                   'E-dom','Country','State','City','Address','Is_Address','Is_Name',
                   'Plan','Newsletter opt in','Source value','Device',
                   'usersCount','activity Frequency','Status','days']]
#%%
# Label Encoding
colnamele =['Company type','Gender','Designation','Country','Industry','Website','E-dom',
            'State','City','Source value','Device','Newsletter opt in','Plan','Is_Address',
            'Is_Name']
##########################################################################################################################################
#%%  DATA Processing Completed #%%


##########################################################################################################################################
# MODELLING

# Multiple Dependent features extracted or generated.
# Using Ensemble modelling approach. as some data points are imbalanced and to control overfitting.

from sklearn import preprocessing
le={}  
#new_df = pd.DataFrame()
for x in colnamele:
    le[x]=preprocessing.LabelEncoder()
for x in colnamele:
    le[x].fit(dataset[x].astype('str'))
    le_dict_loop = dict(zip(le[x].classes_, le[x].transform(le[x].classes_)))
    dataset[x] = (dataset[x].astype('str')).apply(lambda x: le_dict_loop.get(x, -1))
    le[x] = dict(zip(le[x].classes_, le[x].transform(le[x].classes_)))

print(dataset.head())
dataset.drop(['Address','Full name','Company name','Form filled on'],axis=1,inplace=True)
#%%
Correlation = dataset.apply(lambda x : pd.factorize(x)[0]).corr(method='pearson', min_periods=1)
#%%
X = dataset.values[:,:-4]          #independent vars
Y = dataset.values[:,-4:]           # dependent var
Y= Y.astype(int)
#%%
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)
print(X)
#%%
# Training the Model   (X-Train, Y_train)
from sklearn.model_selection import train_test_split
#Split the data into test and train
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.30,
                                                    random_state=3)
#%%
# Training 
from sklearn.svm import SVC
svclassifier = SVC(probability = True) 
from sklearn.ensemble import RandomForestClassifier
rfm= RandomForestClassifier(n_estimators=25, min_samples_split=25, max_depth=7, max_features=1)
from xgboost import XGBClassifier
xgb = XGBClassifier(n_estimators = 400, max_depth = 5)    # 100, 3
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier(n_estimators = 100)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5, weights='distance', algorithm='brute', leaf_size=100, p=3)
from sklearn.linear_model import LogisticRegression
logit = LogisticRegression()
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# for 1st
svclassifier.fit(X_train, Y_train[:,:-3])  
Y_pred_svc1 = svclassifier.predict(X_test)
svm_confusion= [confusion_matrix(Y_test[:,:-3], Y_pred_svc1)]
svm_accuracy = [accuracy_score(Y_test[:,:-3].tolist(),Y_pred_svc1)]

rfm.fit(X_train,Y_train[:,:-3])
Y_pred_rfm1 = rfm.predict(X_test)
rfm_confusion= [confusion_matrix(Y_test[:,:-3], Y_pred_rfm1)]
rfm_accuracy = [accuracy_score(Y_test[:,:-3].tolist(),Y_pred_rfm1)]

xgb.fit(X_train,Y_train[:,:-3])
Y_pred_xgb1 = xgb.predict(X_test)
xgb_confusion= [confusion_matrix(Y_test[:,:-3], Y_pred_xgb1)]
xgb_accuracy = [accuracy_score(Y_test[:,:-3].tolist(),Y_pred_xgb1)]

gbc.fit(X_train,Y_train[:,:-3])
Y_pred_gbc1 = gbc.predict(X_test)
gbc_confusion= [confusion_matrix(Y_test[:,:-3], Y_pred_gbc1)]
gbc_accuracy = [accuracy_score(Y_test[:,:-3].tolist(),Y_pred_gbc1)]

knn.fit(X_train,Y_train[:,:-3])
Y_pred_knn1 = knn.predict(X_test)
knn_confusion= [confusion_matrix(Y_test[:,:-3], Y_pred_knn1)]
knn_accuracy = [accuracy_score(Y_test[:,:-3].tolist(),Y_pred_knn1)]

logit.fit(X_train,Y_train[:,:-3])
Y_pred_log1 = logit.predict(X_test)
log_confusion= [confusion_matrix(Y_test[:,:-3], Y_pred_log1)]
log_accuracy = [accuracy_score(Y_test[:,:-3].tolist(),Y_pred_log1)]

# for 2nd
svclassifier.fit(X_train, Y_train[:,1:-2])  
Y_pred_svc2 = svclassifier.predict(X_test) 
svm_confusion.append(confusion_matrix(Y_test[:,1:-2], Y_pred_svc2))
svm_accuracy.append(accuracy_score(Y_test[:,1:-2].tolist(),Y_pred_svc2))
rfm.fit(X_train,Y_train[:,1:-2])
Y_pred_rfm2 = rfm.predict(X_test)
rfm_confusion.append(confusion_matrix(Y_test[:,1:-2], Y_pred_rfm2))
rfm_accuracy.append(accuracy_score(Y_test[:,1:-2].tolist(),Y_pred_rfm2))

xgb.fit(X_train,Y_train[:,1:-2])
Y_pred_xgb2 = xgb.predict(X_test)
xgb_confusion.append(confusion_matrix(Y_test[:,1:-2], Y_pred_xgb2))
xgb_accuracy.append(accuracy_score(Y_test[:,1:-2].tolist(),Y_pred_xgb2))

gbc.fit(X_train,Y_train[:,1:-2])
Y_pred_gbc2 = gbc.predict(X_test)
gbc_confusion.append(confusion_matrix(Y_test[:,1:-2], Y_pred_gbc2))
gbc_accuracy.append(accuracy_score(Y_test[:,1:-2].tolist(),Y_pred_gbc2))

knn.fit(X_train,Y_train[:,1:-2])
Y_pred_knn2 = knn.predict(X_test)
knn_confusion.append(confusion_matrix(Y_test[:,1:-2], Y_pred_knn2))
knn_accuracy.append(accuracy_score(Y_test[:,1:-2].tolist(),Y_pred_knn2))

logit.fit(X_train,Y_train[:,1:-2])
Y_pred_log2 = logit.predict(X_test)
log_confusion.append(confusion_matrix(Y_test[:,1:-2], Y_pred_log2))
log_accuracy.append(accuracy_score(Y_test[:,1:-2].tolist(),Y_pred_log2))


# for 3nd
svclassifier.fit(X_train, Y_train[:,2:-1])  
Y_pred_svc3 = svclassifier.predict(X_test) 
svm_confusion.append(confusion_matrix(Y_test[:,2:-1], Y_pred_svc3))
svm_accuracy.append(accuracy_score(Y_test[:,2:-1].tolist(),Y_pred_svc3))

rfm.fit(X_train,Y_train[:,2:-1])
Y_pred_rfm3 = rfm.predict(X_test)
rfm_confusion.append(confusion_matrix(Y_test[:,2:-1], Y_pred_rfm3))
rfm_accuracy.append(accuracy_score(Y_test[:,2:-1].tolist(),Y_pred_rfm3))

xgb.fit(X_train,Y_train[:,2:-1])
Y_pred_xgb3 = xgb.predict(X_test)
xgb_confusion.append(confusion_matrix(Y_test[:,2:-1], Y_pred_xgb3))
xgb_accuracy.append(accuracy_score(Y_test[:,2:-1].tolist(),Y_pred_xgb3))

gbc.fit(X_train,Y_train[:,2:-1])
Y_pred_gbc3 = gbc.predict(X_test)
gbc_confusion.append(confusion_matrix(Y_test[:,2:-1], Y_pred_gbc3))
gbc_accuracy.append(accuracy_score(Y_test[:,2:-1].tolist(),Y_pred_gbc3))

knn.fit(X_train,Y_train[:,2:-1])
Y_pred_knn3 = knn.predict(X_test)
knn_confusion.append(confusion_matrix(Y_test[:,2:-1], Y_pred_knn3))
knn_accuracy.append(accuracy_score(Y_test[:,2:-1].tolist(),Y_pred_knn3))

logit.fit(X_train,Y_train[:,2:-1])
Y_pred_log3 = logit.predict(X_test)
log_confusion.append(confusion_matrix(Y_test[:,2:-1], Y_pred_log3))
log_accuracy.append(accuracy_score(Y_test[:,2:-1].tolist(),Y_pred_log3))

# for 4th
svclassifier.fit(X_train, Y_train[:,3:])  
Y_pred_svc4 = svclassifier.predict(X_test) 
svm_confusion.append(confusion_matrix(Y_test[:,3:], Y_pred_svc4))
svm_accuracy.append(accuracy_score(Y_test[:,3:].tolist(),Y_pred_svc4))

rfm.fit(X_train,Y_train[:,3:])
Y_pred_rfm4 = rfm.predict(X_test)
rfm_confusion.append(confusion_matrix(Y_test[:,3:], Y_pred_rfm4))
rfm_accuracy.append(accuracy_score(Y_test[:,3:].tolist(),Y_pred_rfm4))

xgb.fit(X_train,Y_train[:,3:])
Y_pred_xgb4 = xgb.predict(X_test)
xgb_confusion.append(confusion_matrix(Y_test[:,3:], Y_pred_xgb4))
xgb_accuracy.append(accuracy_score(Y_test[:,3:].tolist(),Y_pred_xgb4))

gbc.fit(X_train,Y_train[:,3:])
Y_pred_gbc4 = gbc.predict(X_test)
gbc_confusion.append(confusion_matrix(Y_test[:,3:], Y_pred_gbc4))
gbc_accuracy.append(accuracy_score(Y_test[:,3:].tolist(),Y_pred_gbc4))

knn.fit(X_train,Y_train[:,3:])
Y_pred_knn4 = knn.predict(X_test)
knn_confusion.append(confusion_matrix(Y_test[:,3:], Y_pred_knn4))
knn_accuracy.append(accuracy_score(Y_test[:,3:].tolist(),Y_pred_knn4))

logit.fit(X_train,Y_train[:,3:])
Y_pred_log4 = logit.predict(X_test)
log_confusion.append(confusion_matrix(Y_test[:,3:], Y_pred_log4))
log_accuracy.append(accuracy_score(Y_test[:,3:].tolist(),Y_pred_log4))

#%%
# =============================================================================
# # Combine and study results
# test_result = pd.DataFrame()
# test_result['svm'] = svm_confusion
# test_result['svm_acc'] = svm_accuracy
# test_result['rfm'] = rfm_confusion
# test_result['rfm_acc'] = rfm_accuracy
# test_result['xgb'] = xgb_confusion
# test_result['xgb_acc'] = xgb_accuracy
# test_result['gbc'] = gbc_confusion
# test_result['gbc_acc'] = gbc_accuracy
# test_result['knn'] = knn_confusion
# test_result['knn_acc'] = knn_accuracy
# test_result['log'] = log_confusion
# test_result['log_acc'] = log_accuracy
# 
# test_result['avg'] = (test_result['svm_acc']+ test_result['rfm_acc']+
#            test_result['xgb_acc']+test_result['gbc_acc']+test_result['knn_acc']+
#            test_result['log_acc']) / 6
#            
# test_cor = test_result.corr()


# Average of all Accuracy meets = 78 %
		#		 Precision =  71 %
		#        Recall    =  89 %
# =============================================================================
#%%
# TRAINING ENDS
#################################################################################################
#%%
########## FUNCTN ##############

def pre_test(dataframe):
    DF1 = dataframe
    DF1 = DF1[DF1['Is account created'] == 'YES']  
    DF1['Website'] = (DF1.Website.notnull()).astype('category')  
    com_typx = pd.DataFrame(DF1['Company name'].str.findall(r"\bPvt ltd\b|\bpvt limited\b|\bPVT.LTD\b|\bpvt. Ltd\b|\bPrivate limited\b|\bPvt Ltd\b|\bPrivate Limited\b|\bpvt. ltd.\b|\bPVT LTD\b|\bPRIVATE LIMITED\b|\bpvt ltd\b|\bPvt. Ltd\b|\bPvt. Ltd.\b|\bPvt Ltd.\b|\bPvt Limited\b|\bprivate limited\b"))
    com_typx.iloc[:,0] = com_typx.iloc[:,0].astype(str)
    com_typx.iloc[:,0] = com_typx.iloc[:,0].str.replace(r"[",'')
    com_typx.iloc[:,0] = com_typx.iloc[:,0].str.replace(r"]",'')
    cm_tx = com_typx.iloc[:,0].str.len()
    cm_ltx = []
    for value in cm_tx[:]:
        if value > 3:
            cm_ltx.append('Pvt Ltd')
        else:
            cm_ltx.append(None)
            
    DF1['Company type'] = cm_ltx 
    em = DF1['Email address'].str.findall(r"(?i)\@\w+")
    em = pd.DataFrame(em.values.tolist())
    em[0] = em[0].astype(str)
    em[0] = em[0].str.replace(r"[",'')
    em[0] = em[0].str.replace(r"]",'')
    em[0]= em[0].str.lower()
    em[0].value_counts()
    domn = ['@gmail','@yahoo','@hotmail','@outlook','@rediffmail','@live']
    em[1] = em[0].apply(lambda x : ['@gmail' in x,'@yahoo' in x,
      '@hotmail' in x,'@outlook' in x,'@rediffmail' in x,'@live' in x])

    DF1['E-dom'] = em[1].apply(lambda x : False if True in x else True)  
    DF1['E-dom'] = DF1['E-dom'].where(pd.notnull(DF1['E-dom']), False)
    DF1['City'] = DF1['City'].str.lower()
    DF1.drop(['Is account created','Step Reached','First name','Last name','Email address',
               'Mobile number','Zip code','Phone number','Terms & Conditions agreed','User limit',
               'Mail link expired on','Feature value','Mobile verified',
               'Email address verified'],axis=1,inplace=True)
    
    DF1.Designation[DF1.Designation == 'DIRECTOR'] = 'Director'
    DF1.Designation[DF1.Designation == 'director'] = 'Director'
    DF1.Designation[DF1.Designation == 'ceo'] = 'CEO'
    DF1.Designation[DF1.Designation == 'Ceo'] = 'CEO'
    DF1.Designation[DF1.Designation == 'Sr Manager'] = 'Senior Manager'
    DF1.Designation[DF1.Designation == 'MD'] = 'Managing Director'
    DF1.Designation[DF1.Designation == 'Proprietor'] = 'Owner'
    DF1.Designation[DF1.Designation == 'proprietor'] = 'Owner'
    DF1.Designation[DF1.Designation == 'Propriter'] = 'Owner'
    DF1.Designation[DF1.Designation == 'system administrator'] = 'System Administrator'
    DF1.Designation[DF1.Designation == 'manager'] = 'Manager'
    DF1.Designation[DF1.Designation == 'MANAGER'] = 'Manager'
    DF1.Designation[DF1.Designation == 'General Manager'] = 'Manager'
    DF1.Designation[DF1.Designation == 'Manager IT'] = 'Manager'
    DF1.Designation[DF1.Designation == 'Marketing Manager'] = 'Manager'
    DF1.Designation[DF1.Designation == 'Operations Manager'] = 'Manager'
    DF1.Designation[DF1.Designation == 'Hr Manager'] = 'Manager'
    DF1.Designation[DF1.Designation == 'Business Manager'] = 'Manager'
    DF1.Designation[DF1.Designation == 'Mangaer'] = 'Manager'
    DF1.Designation[DF1.Designation == 'Investment Manager'] = 'Manager'
    DF1.Designation[DF1.Designation == 'Associate Manager'] = 'Manager'
    DF1.Designation[DF1.Designation == 'Assistant Manager'] = 'Manager'
    DF1.Designation[DF1.Designation == 'Zonal Manager - Sales'] = 'Manager'
    DF1.Designation[DF1.Designation == 'BDM'] = 'Manager'
    DF1.Designation[DF1.Designation == 'GM'] = 'Manager'
    DF1.Designation[DF1.Designation == 'DGM'] = 'Manager'
    DF1.Designation[DF1.Designation == 'Operation Manager'] = 'Manager'
    DF1.Designation[DF1.Designation == 'PM'] = 'Manager'
    DF1.Designation[DF1.Designation == 'abm'] = 'Manager'
    DF1.Designation[DF1.Designation == 'Senior Manager'] = 'Manager'
    DF1.Designation[DF1.Designation == 'Sales Manager'] = 'Manager'
    DF1.Designation[DF1.Designation == 'Branch Manager'] = 'Manager'
    DF1.Designation[DF1.Designation == 'SENIOR MANAGER'] = 'Manager'
    DF1.Designation[DF1.Designation == 'HR/Admin Manager'] = 'HR'
    DF1.Designation[DF1.Designation == 'Hr'] = 'HR'
    DF1.Designation[DF1.Designation == 'Asst.Manager - Human Resource'] = 'HR'
    DF1.Designation[DF1.Designation == 'HR & IT'] = 'HR'
    DF1.Designation[DF1.Designation == 'HR & Admin'] = 'HR'
    DF1.Designation[DF1.Designation == 'Human Resource Manager'] = 'HR'
    DF1.Designation[DF1.Designation == 'Human Resource'] = 'HR'
    DF1.Designation[DF1.Designation == 'partner'] = 'Partner'
    DF1.Designation[DF1.Designation == 'Co-Founder'] = 'Partner'

    DF1 = DF1.where(pd.notnull(DF1), 'Not Available')
    # Categories to other
    otrT = DF1['Industry'].value_counts()
    DF1['Industry']=np.where(DF1['Industry'].isin(otrT.index[otrT <= 2]), 'OTHER', DF1['Industry'])
    otr1T = DF1['Designation'].value_counts()
    DF1['Designation']=np.where(DF1['Designation'].isin(otr1T.index[otr1T <= 2]), 'OTHER', DF1['Designation'])
    otr2T = DF1['City'].value_counts()
    DF1['City']=np.where(DF1['City'].isin(otr2T.index[otr2T <= 2]), 'OTHER', DF1['City'])
    address = DF1['Address'].str.len()
    DF1['Is_Address'] = pd.cut(address,
                         bins=[-1, 20, 50 ,1000],
                         labels=['False','TF','True'])
    # Name
    name = DF1['Full name'].str.len()
    name_li = []
    for value in name[:]:
        if value > 6:
            name_li.append(True)
        else:
            name_li.append(False)
    #print(name_li)
    
    DF1['Is_Name'] = name_li
    for col in ['Company type','Gender','Designation','Country','Industry','Website','State','City','Source value','Device',
                'Newsletter opt in','Plan','Is_Address','Is_Name','E-dom']:
        DF1[col] = DF1[col].astype('category')
    
    # NAive Bayes
    bay_db =  DF1.copy()
    bay_db.drop(['Address','Full name'],axis=1,inplace=True)
    bay_db = bay_db.rename(columns={'Company name': 'Company Name'})
    # Arrange in order
    DF1 = DF1[['Company type','Gender','Designation','Industry','Website','E-dom',
               'Country','State','City','Is_Address','Is_Name','Plan','Newsletter opt in',
               'Source value','Device']]
    DF11 = pd.DataFrame(DF1)
    return(DF11,bay_db)
#%%
# Data SQL
#%%
# To Predict the data
import mysql.connector
from sshtunnel import SSHTunnelForwarder
import pandas as pd
#%%
REMOTE_SERVER_IP = ''
PRIVATE_SERVER_IP = '127.0.0.1'

engine = create_engine("")
conn = engine.connect()
diy = pd.read_sql_query('select * from ;', con=conn)
users = pd.read_sql_query('select * from ;', con=conn)
accounts = pd.read_sql_query('select * from ;', con=conn)
account_region = pd.read_sql_query('select * from ;', con=conn)
ac_fq = pd.read_sql_query('SELECT * FROM ;', con=conn)
conn.close()
#%%
# DIY Conversion 
    
diy.is_account_created[diy.is_account_created == 1] = 'YES'
diy.is_account_created[diy.is_account_created == 0] = 'NO'    

diy.gender[diy.gender == 1] = 'Male'
diy.gender[diy.gender == 0] = 'Female'
    
diy.is_newsletter_opt_in[diy.is_newsletter_opt_in == 1] = 'YES'
diy.is_newsletter_opt_in[diy.is_newsletter_opt_in == 0] = 'NO'

diy.is_terms_condition_agreed[diy.is_terms_condition_agreed == 1] = 'YES'
diy.is_terms_condition_agreed[diy.is_terms_condition_agreed == 0] = 'NO'  

diy.isvalid_email_address.value_counts()
dataset2['Email address verified'].value_counts()
#1 = yes , 3 = Unknown
diy.isvalid_email_address[diy.isvalid_email_address == 1] = 'YES'
diy.isvalid_email_address[diy.isvalid_email_address == 3] = 'UNKNOWN'  

diy['is_mobile_valid'].value_counts()
dataset2['Mobile verified'].value_counts()
# 1 = yes, 2 = Unknown
diy.is_mobile_valid[diy.is_mobile_valid == 1] = 'YES'
diy.is_mobile_valid[diy.is_mobile_valid == 2] = 'UNKNOWN'  

diy = diy[['created_on','is_account_created','step_number','first_name','last_name',
             'full_name','email_address','mobile_number','gender','designation','company_name','country',
             'state','city','address1','zip_code','company_phone_number1','industry',
             'company_website','plan','user_limit','mail_expired_on',
             'is_terms_condition_agreed','is_newsletter_opt_in','g_source','g_feature',
             'device','is_mobile_valid','isvalid_email_address']]
    
diy.columns = ['Form filled on','Is account created','Step Reached','First name','Last name','Full name',
                'Email address','Mobile number','Gender','Designation','Company name',
                'Country','State','City','Address','Zip code','Phone number','Industry',
                'Website','Plan','User limit','Mail link expired on','Terms & Conditions agreed',
                'Newsletter opt in','Source value','Feature value','Device','Mobile verified',
                'Email address verified']

TEST,BAY = pre_test(diy)
#%%  
# Scale data
for x in colnamele:
    le_dict_loop = le[x]
    TEST[x] = (TEST[x].astype('str')).apply(lambda x: le_dict_loop.get(x, -1))

#scaler = StandardScaler()
TEST = scaler.transform(TEST)
###############################################################################################################
#%%
# Predictions
Prediction = pd.DataFrame()
Prediction['company_name'] = BAY['Company Name']
Prediction['Date'] = BAY['Form filled on']

Prediction['SVM 0'] = 0
Prediction['SVM 1'] = 0
Prediction['RFM 0'] = 0
Prediction['RFM 1'] = 0
Prediction['XGB 0'] = 0
Prediction['XGB 1'] = 0
Prediction['GBC 0'] = 0
Prediction['GBC 1'] = 0
Prediction['Logit 0'] = 0
Prediction['Logit 1'] = 0

Prediction[['SVM 0','SVM 1']]= svclassifier.predict_proba(TEST)
Prediction[['RFM 0','RFM 1']]= rfm.predict_proba(TEST)
Prediction[['XGB 0','XGB 1']] = xgb.predict_proba(TEST)
Prediction[['GBC 0','GBC 1']] = gbc.predict_proba(TEST)
Prediction[['Logit 0','Logit 1']] = logit.predict_proba(TEST)

#%%
# Final
Prediction['Probability of Account not likely to use FieldSense'] = (Prediction['SVM 0']+Prediction['RFM 0']+Prediction['XGB 0']+
          Prediction['GBC 0']+Prediction['Logit 0']) / 5

Prediction['Probability of Account likely to use FieldSense'] = (Prediction['SVM 1']+Prediction['RFM 1']+Prediction['XGB 1']+
          Prediction['GBC 1']+ Prediction['Logit 1']) / 5

Prediction = Prediction[['Date','company_name','Probability of Account likely to use FieldSense','Probability of Account not likely to use FieldSense',
               'SVM 0','RFM 0','XGB 0','GBC 0','Logit 0','SVM 1','RFM 1','XGB 1','GBC 1','Logit 1']]
#%%
import datetime
yesterday = str(datetime.date.today() - datetime.timedelta(days = 1))

Prediction1 = Prediction[['Date','company_name','Probability of Account likely to use FieldSense']]

Prediction1.Date = Prediction1.Date.astype('str')
Prediction1.Date = Prediction1.Date.apply(lambda x: x[:-9])
Prediction1.Date = Prediction1.Date.astype('category')
Prediction1 = Prediction1[Prediction1.Date ==  yesterday]
    
Prediction1['Probability of Account likely to use FieldSense_'] = pd.cut(Prediction1['Probability of Account likely to use FieldSense'],bins=[0,0.10,0.20,0.30,0.55],
       labels=['Low','Low-Medium','Medium-High','High'])

#Prediction1.drop(['Probability of Account likely to use FieldSense'],1,inplace=True)
accounts=accounts.merge(account_region,left_on='region_id_fk',right_on='id',how='inner')
Prediction1=Prediction1.merge(accounts,left_on='company_name',right_on='company_name',how='inner')
Prediction1=Prediction1[['Date', 'company_name',
       'Probability of Account likely to use FieldSense_','Probability of Account likely to use FieldSense','city', 'state', 'company_phone_number1', 'company_phone_number2','region_name']]
Prediction1.columns=['Date', 'company_name',
       'Probability','Probability of Account likely to use FieldSense','city', 'state', 'company_phone_number1', 'company_phone_number2','region_name']
#print(Prediction1)

# Pushing Predicted Values to DATABASE
from sqlalchemy import create_engine
engine = create_engine("")
con = engine.connect()
Prediction1.to_sql(con=con, name='table', if_exists='append')
con.close()