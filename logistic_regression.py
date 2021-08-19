import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
#getdata
url = "titanic.csv"
titanic_data=pd.read_csv(url)
print(titanic_data.shape)
print(titanic_data.head(10))

print('no of passenger :',len(titanic_data))
print("Information : \n",titanic_data.info())

print(
    'Null values info in percentage: \n',
     (titanic_data.isnull().sum()/len(titanic_data))*100
    )

#analyze data
#sns.countplot(x='survived',data=titanic_data)
#plt.show()

#sns.countplot(x='survived',hue='sex',data=titanic_data)
#plt.show()
#sns.countplot(x='survived',hue='pclass',data=titanic_data)
#plt.show()
#titanic_data["age"].plot.hist()
#plt.show()

#data wrangling
print(titanic_data.isnull())
titanic_data.drop('cabin',axis=1,inplace=True)
titanic_data.drop('body',axis=1,inplace=True)
titanic_data.drop('home.dest',axis=1,inplace=True)

print(titanic_data.head(10))
titanic_data.dropna(inplace=True)
print("Information : \n",titanic_data.info())
print((titanic_data.isnull().sum()/len(titanic_data))*100)
print(titanic_data.head(),"\n",titanic_data.shape)

sns.heatmap(titanic_data.isnull(),yticklabels=False,cbar=False)
plt.show()

s=pd.get_dummies(titanic_data['sex'],drop_first=True)
p=pd.get_dummies(titanic_data['embarked'],drop_first=True)
pcl=pd.get_dummies(titanic_data['pclass'],drop_first=True)

titanic_data=pd.concat([titanic_data,s,p,pcl],axis=1)
print(titanic_data.head(10),titanic_data.info())
print(titanic_data['sibsp'].head(20))
titanic_data.drop(['sex','pclass','boat','name','ticket','embarked'],axis=1,inplace=True)
print(titanic_data.head(10))



#train

x=titanic_data.drop('survived',axis=1)
y=titanic_data['survived']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.3,shuffle=True)
logmodel=LogisticRegression()
logmodel.fit(x_train,y_train)

pred=logmodel.predict(x_test)
#print(classification_report(y_test,pred))
print(accuracy_score(y_test,pred))