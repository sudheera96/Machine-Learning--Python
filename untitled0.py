# -*- coding: utf-8 -*-
"""
Created on Sat Jun  2 10:05:10 2018

@author: ADMIN
"""

import pandas as pd
import matplotlib as plt
import numpy as np

df=pd.read_csv("C:/Users/ADMIN/Desktop/train_u6lujuX_CVtuZ9i.csv")

df.head(10)
df.describe()
df['Property_Area'].value_counts()
df['Credit_History'].value_counts()

df['ApplicantIncome'].hist(bins=50)
df['CoapplicantIncome'].hist(bins=50)

df.boxplot(column='ApplicantIncome')

df.boxplot(column='ApplicantIncome',by='Education')
df.boxplot(column='ApplicantIncome',by='Self_Employed')

df['LoanAmount'].hist(bins=50)
df.boxplot(column='LoanAmount')

temp1=df['Credit_History'].value_counts(ascending=True)

temp2=df.pivot_table(values='Loan_Status',index=['Credit_History'],
                     aggfunc=lambda x:x.map({'Y':1,'N':0}).mean())

fig = plt.figure(figsize=(8,4))
ax1 = fig.add_subplot(121)
ax1.set_xlabel('Credit_History')
ax1.set_ylabel('Count of Applicants')
ax1.set_title("Applicants by Credit_History")
temp1.plot(kind='bar')

ax2 = fig.add_subplot(122)
temp2.plot(kind = 'bar')
ax2.set_xlabel('Credit_History')
ax2.set_ylabel('Probability of getting loan')
ax2.set_title("Probability of getting loan by credit history")

temp3 = pd.crosstab(df['Credit_History'], df['Loan_Status'])
temp3.plot(kind='bar', stacked=True, color=['red','blue'], grid=False)

 df.apply(lambda x: sum(x.isnull()),axis=0)

df['LoanAmount'].fillna(df['LoanAmount'].mean(), inplace=True)

df['Self_Employed'].value_counts()
df['Self_Employed'].fillna('No', inplace=True)
table = df.pivot_table(values='LoanAmount', index='Self_Employed' ,columns='Education', aggfunc=np.median)
def fage(x):
 return table.loc[x['Self_Employed'],x['Education']]

df['LoanAmount'].fillna(df[df['LoanAmount'].isnull()].apply(fage, axis=1), inplace=True)


df['LoanAmount_log'] = np.log(df['LoanAmount'])
df['LoanAmount_log'].hist(bins=20)

df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
df['TotalIncome_log'] = np.log(df['TotalIncome'])
df['LoanAmount_log'].hist(bins=20) 

