import pandas as pd
#import numpy as np

df=pd.read_csv("C:/Users/benso/Desktop/gitCentral/Inteligencia_Artificial/Datasets/USA_Housing.csv")
#df=pd.read_csv('_USA_Housing.csv')


# scale the attributes using standard scaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

colunas = ['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
       'Avg. Area Number of Bedrooms', 'Area Population']
df[colunas] = scaler.fit_transform(df[colunas])

X = df.drop(['Price','Address'], axis=1)
y = df['Price']

#X = df.drop(['Avg. Area Number of Bedrooms','Price','Address'], axis=1)
#y = df['Price']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split (X, y, test_size=.3, random_state=23)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()

lr.fit(X_train, y_train)

#----------------------------------------
# save the model
import pickle

pickle.dump(lr, open('model.pkl','wb'))

#confirm
model=pickle.load(open('model.pkl', 'rb'))
print(model.predict([[1,1.2,0.8,0.9,1.1]]))

# save the scaler
pickle.dump(scaler, open('scaler.pkl','wb'))

#confirm
scaler = pickle.load(open('scaler.pkl','rb'))
print(scaler.transform([[80000,5,6,4,23000]]))