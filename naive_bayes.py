# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 11:09:03 2016

@author: AbreuLastra_Work
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

df = pd.read_csv("https://raw.githubusercontent.com/Thinkful-Ed/curric-data-001-data-sets/master/ideal-weight/ideal_weight.csv", header=0, index_col=0)

df.columns = [i.strip("'") for i in df.columns]
df['sex'] = [j.strip("'") for j in df['sex']]


#Plotting histograms
plt.hist(df['actual'], alpha = .4, label = "actual")
plt.hist(df['ideal'], label = "ideal")
plt.legend(loc='upper right')
plt.show()

plt.figure()
plt.hist(df['diff'], bins=20, alpha=.5, label = "ideal vs actual weight")
plt.legend(loc='upper right')
plt.show()

df['sex']= pd.get_dummies(df['sex'])


##Naive

model = GaussianNB()

y = df['sex']
X=df[df.columns[1:]]

model=model.fit(X,y)

y_pred = model.predict(X)

print metrics.accuracy_score(y, y_pred)

#Predict the sex for an actual weight of 145, an ideal weight of 160, and a diff of -15.
X_1= pd.Series([145, 160, -15])

print(model.predict(X_1))

#Predict the sex for an actual weight of 160, an ideal weight of 145, and a diff of 15.
X_2= pd.Series([160, 145, 15])
print(model.predict(X_2))