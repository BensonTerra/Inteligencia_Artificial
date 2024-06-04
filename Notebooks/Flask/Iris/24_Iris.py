import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

df=load_iris()
#df=pd.read_csv('_USA_Housing.csv')

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()

X = df.data
y = df.target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split (X, y, test_size=.3, random_state=23)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

#----------------------------------------
# save the model
import pickle

pickle.dump(clf, open('model.pkl','wb'))

#confirm
model=pickle.load(open('model.pkl', 'rb'))
print(model.predict([[1,1.2,0.8,0.9]]))