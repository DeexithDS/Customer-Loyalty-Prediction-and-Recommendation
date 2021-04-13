# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 14:51:54 2021

@author: DEEXITH REDDY
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
from scipy.stats import pearsonr
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

##Models:

from sklearn import metrics
from sklearn.metrics import confusion_matrix

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from xgboost import XGBClassifier #pip install xgboost
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process

##Loading tables and merging based on offer number
df=pd.read_csv("C:/Users/DEEXITH REDDY/Desktop/D&A Case Study/trainHistory.csv")
df1=pd.read_csv("C:/Users/DEEXITH REDDY/Desktop/D&A Case Study/offers.csv")
new=pd.merge(df,df1,left_on='offer',right_on='offer')

##Categorical variables
new["repeater"] =new["repeater"].astype('category')
new["repeater"] =new["repeater"].cat.codes

##Checking if the categorical variables worked
df.loc[df['id'] == 15753725]
new.loc[new['id'] == 15753725]

##Calculating percentage returned and not returned

amount_notreturned = new[new['repeater'] == 0]['repeater'].count() / new.shape[0] * 100
amount_returned = new[new['repeater'] == 1]['repeater'].count() / new.shape[0] * 100

sns.countplot(x='repeater', palette="Set3", data=new)
plt.xticks([0, 1], ['Not returned', 'Returned'])
plt.xlabel('Condition', size=15, labelpad=12, color='grey')
plt.ylabel('Amount of customers', size=15, labelpad=12, color='grey')
plt.title("Proportion of customers returned and not returned", size=15, pad=20)
plt.ylim(0, 160000)
plt.text(-0.15, 7000, f"{round(amount_notreturned, 2)}%", fontsize=12)
plt.text(0.85, 1000, f"{round(amount_returned, 2)}%", fontsize=12)
sns.despine()


##We see that majority have not returned. So we check and try to predict those who have returned.


#BRAND
##Checking brand vs. repeater correlation using one hot encoder

brandrepeater=new[['repeater','brand']]
brandrepeater
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(handle_unknown='ignore')
enc_df = pd.DataFrame(enc.fit_transform(brandrepeater[['brand']]).toarray())
enc_df
brandrepeater = brandrepeater.join(enc_df)



y=brandrepeater["repeater"]
x=brandrepeater.drop(["repeater","brand"],axis=1)


logit_model=sm.Logit(y,x)
result=logit_model.fit()
print(result.summary2())


##Except for one brand "3", all other brands were significant on the customers repeating
x=x.drop(3,axis=1)

##After "3" is dropped, all have high significance


#OFFERVALUE
##Checking if offer value has an effect on the returning of customer
##Six unique values
corr, _ = pearsonr(new['offervalue'], new['repeater'])
print('Pearsons correlation: %.3f' % corr)

new=new.drop('offervalue',axis=1)
#-0.044


##Company

companyrepeater=new[['repeater','company']]
enc = OneHotEncoder(handle_unknown='ignore')
enc_df = pd.DataFrame(enc.fit_transform(companyrepeater[['company']]).toarray())
enc_df
companyrepeater = companyrepeater.join(enc_df)



y=companyrepeater["repeater"]
x=companyrepeater.drop(["repeater","company"],axis=1)


logit_model=sm.Logit(y,x)
result=logit_model.fit()
print(result.summary2())

##Company 6 is not significant

##Quantity has only one value, "1". So we delete

new['quantity'].unique()
##array([1], dtype=int64)
new=new.drop('quantity',axis=1)

##Category:
categoryrepeater=new[['repeater','category']]
categoryrepeater
enc = OneHotEncoder(handle_unknown='ignore')
enc_df = pd.DataFrame(enc.fit_transform(categoryrepeater[['category']]).toarray())
categoryrepeater = categoryrepeater.join(enc_df)
categoryrepeater
y=categoryrepeater["repeater"]
x=categoryrepeater.drop(["repeater","category"],axis=1)

logit_model=sm.Logit(y,x)
result=logit_model.fit()
print(result.summary2())

##All categories are important

##Repeat trips

y=new["repeater"]
x=new["repeattrips"]

logit_model=sm.Logit(y,x)
result=logit_model.fit()
print(result.summary2())

new=new.drop('repeattrips',axis=1)

##Repeat trips has no significance

##Checking for market:
marketrepeater=new[['repeater','market']]
enc = OneHotEncoder(handle_unknown='ignore')
enc_df = pd.DataFrame(enc.fit_transform(marketrepeater[['market']]).toarray())
marketrepeater = marketrepeater.join(enc_df)
y=marketrepeater["repeater"]
x=marketrepeater.drop(["repeater","market"],axis=1)
logit_model=sm.Logit(y,x)
result=logit_model.fit()
print(result.summary2())

##All markets are important

marketrepeater=new[['repeater','market']]
enc = OneHotEncoder(handle_unknown='ignore')
enc_df = pd.DataFrame(enc.fit_transform(marketrepeater[['market']]).toarray())
marketrepeater = marketrepeater.join(enc_df)
y=marketrepeater["repeater"]
x=marketrepeater.drop(["repeater","market"],axis=1)
logit_model=sm.Logit(y,x)
result=logit_model.fit()
print(result.summary2())

##Offers:



##Offer 1194044 is not important as p-value is high

offersrepeater=new[['repeater','offer']]
enc = OneHotEncoder(handle_unknown='ignore')
enc_df = pd.DataFrame(enc.fit_transform(offersrepeater[['offer']]).toarray())
offersrepeater = offersrepeater.join(enc_df)
y=offersrepeater["repeater"]
x=offersrepeater.drop(["repeater","offer"],axis=1)
logit_model=sm.Logit(y,x)
result=logit_model.fit()
print(result.summary2())

##Chain

corr, _ = pearsonr(new['chain'], new['repeater'])
print('Pearsons correlation: %.3f' % corr)


new=new.drop('chain',axis=1)
##Not correlated

##Dates:
##All dates have both repeaters and non-repeaters



##Dropping the ID

new=new.drop('id',axis=1)

##Label Encoding:
##Offer:
new['offer'] = new.offer.astype('category')
df2=pd.get_dummies(new[["offer"]])
new=pd.concat([new,df2],axis=1)
new=new.drop('offer',axis=1)
new=new.drop('offer_1194044',axis=1) ##Dropping insignificant offer

##Company:

new['company'] = new.company.astype('category')
df2=pd.get_dummies(new[["company"]])
new=pd.concat([new,df2],axis=1)
new=new.drop('company',axis=1)
new=new.drop('company_107127979',axis=1) ##Dropping insignificant company

#Brand
new['brand'] = new.brand.astype('category')
df2=pd.get_dummies(new[["brand"]])
new=pd.concat([new,df2],axis=1)
new=new.drop('brand',axis=1)
new=new.drop('brand_6732',axis=1) ##Dropping insignificant brand


##Market
new['market'] = new.market.astype('category')
df2=pd.get_dummies(new[["market"]])
new=pd.concat([new,df2],axis=1)
new=new.drop('market',axis=1) 

##Category 
new['category'] = new.category.astype('category')
df2=pd.get_dummies(new[["category"]])
new=pd.concat([new,df2],axis=1)
new=new.drop('category',axis=1)

##Dropping dates

new=new.drop('offerdates',axis=1)

##Checking through categorical variables:

new=new.drop(['id','chain','repeattrips','offerdate','quantity','offervalue'],axis=1)

new["offer"] =new["offer"].astype('category')
new["offer"] =new["offer"].cat.codes
new['offer']
new["market"] =new["market"].astype('category')
new["market"] =new["market"].cat.codes
new["category"] =new["category"].astype('category')
new["category"] =new["category"].cat.codes
new["company"] =new["company"].astype('category')
new["company"] =new["company"].cat.codes
new["brand"] =new["brand"].astype('category')
new["brand"] =new["brand"].cat.codes

##Splitting
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)


##RECOMMENDATION ENGINE:

f=f.drop(['chain','repeattrips','offerdate','repeater','quantity'],axis=1)
f=f.drop(['offer','offervalue'],axis=1)

f['company'] = f.company.astype('category')
df2=pd.get_dummies(f[["company"]])
f=pd.concat([f,df2],axis=1)
f=f.drop('company',axis=1)

f['brand'] = f.brand.astype('category')
df2=pd.get_dummies(f[["brand"]])
f=pd.concat([f,df2],axis=1)
f=f.drop('brand',axis=1)

f['market'] = f.market.astype('category')
df2=pd.get_dummies(f[["market"]])
f=pd.concat([f,df2],axis=1)
f=f.drop('market',axis=1) 

f['category'] = f.category.astype('category')
df2=pd.get_dummies(f[["category"]])
f=pd.concat([f,df2],axis=1)
f=f.drop('category',axis=1)

f=f.drop(['market_1',
 'market_2',
 'market_4',
 'market_5',
 'market_6',
 'market_7',
 'market_8',
 'market_9',
 'market_10',
 'market_11',
 'market_12',
 'market_14',
 'market_15',
 'market_16',
 'market_17',
 'market_18',
 'market_20',
 'market_21',
 'market_22',
 'market_23',
 'market_24',
 'market_26',
 'market_27',
 'market_28',
 'market_33',
 'market_34',
 'market_35',
 'market_37',
 'market_39',
 'market_43',
 'market_45',
 'market_47',
 'market_93',
 'market_96',],axis=1)
##Preparing for cosive similarity

f1=f.drop('id',axis=1)
f1.info()
df3=pd.DataFrame()

##Creating basic dataframe with all information

df=pd.read_csv("C:/Users/DEEXITH REDDY/Desktop/D&A Case Study/trainHistory.csv")
df1=pd.read_csv("C:/Users/DEEXITH REDDY/Desktop/D&A Case Study/offers.csv")
t=pd.merge(df,df1,left_on='offer',right_on='offer')

k=f.drop('id',axis=1)
df_list=k.values.tolist()


def recommender(n):
 c=[]
 e=[]
 f=[]
 g=[]
 h=[]
 for i in range(0,160057):
  if i!=n:   
   cos_sim=np.dot(df_list[n],df_list[i])/(np.linalg.norm(df_list[n])*np.linalg.norm(df_list[i]))
   c.append(cos_sim)
   e.append(i)
   f.append(list(t.loc[t['index'] == i, 'brand']))
   g.append(list(t.loc[t['index'] == i, 'company']))
   h.append(list(t.loc[t['index'] == i, 'category']))
 return(sorted(zip(e,f,g,h), reverse=True)[:10])


