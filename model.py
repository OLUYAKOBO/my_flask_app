import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


data = pd.read_csv('students_dropout.csv')
data

cols_needed = ['Target','Course','Application mode','Application order']

for c in data.columns:
    if c not in cols_needed:
        data.drop(c,
                  axis=1,
                  inplace= True)
        
#data

data.Target.replace({'Graduate':1,
                     'Dropout':0,
                     'Enrolled':0},
                     inplace = True)

#print(data.head(2))

x = data.drop('Target',
              axis = 1)

y = data.Target

scaler = StandardScaler()
x_scaled = scaler.fit(x)
x_scaled = scaler.fit_transform(x)
#x_scaled = pd.DataFrame(x_scaled, columns=x.columns)

np.random.seed(42)
#print(x)
print(x_scaled)
x_train,x_test,y_train,y_test = train_test_split(x_scaled,
                                                 y,
                                                 test_size = 0.1,
                                                 random_state = 42)


model = RandomForestClassifier()
model.fit(x_train,y_train)
print(model.score(x_test,y_test))

import pickle

pickle.dump(model,open('model.pkl','wb'))


from sklearn.metrics import confusion_matrix, classification_report
y_pred = model.predict(x_test)

print(confusion_matrix(y_test,y_pred))

print(classification_report(y_test,y_pred))