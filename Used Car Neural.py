import numpy as np
import pandas as pd
import sklearn

features = pd.read_csv("data\X_train.csv", sep=',')
targets = pd.read_csv('data\y_train.csv', sep=',')

# print (targets)
# print (features.head)



# one hot encoding of categorical data 
# transmission

# print(features.columns.tolist())
one_hot = pd.get_dummies(features['transmission'])
# print (one_hot)
# print (features['transmission'])
features = features.drop('transmission', axis = 1)

features = features.join(one_hot)


# remove brand and model
features = features.drop('brand', axis = 1)
features = features.drop('model', axis = 1)
# print (features)


# one hot fuelType
# Both fueltype and transmission have an "other" category, complicating thigs
#features['Other'].value_counts()


one_hot = pd.get_dummies(features['fuelType'])


features = features.drop('fuelType', axis = 1)
features = features.join(one_hot, how='left', lsuffix='_tr', rsuffix='_fuel')

# remove carID
features = features.drop('carID', axis = 1)
targets = targets.drop('carID', axis = 1)

# Test without/ with tax feature
#features = features.drop('tax', axis = 1) 


meantax = features['tax'].mean()
print('mean tax is', meantax)


#find 0s in tax
print('0 tax in',features[features.tax==0])
# replace 0s with mean

features['tax']=features['tax'].replace(0, meantax)

# scale the features 

from sklearn.preprocessing import StandardScaler

x = features.copy()
y = targets.copy()

scaler = StandardScaler()
scaler = scaler.fit(x)
scaled_x = scaler.transform(x)




from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(scaled_x, y, test_size=0.2, random_state=42)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

from keras.models import Sequential 
from keras.layers import Dense, Dropout


model = Sequential()

#first layer
model.add (Dense(units = 2048, input_dim = 14, kernel_initializer = 'normal', activation = 'relu'))

# next layers

model.add (Dense(units = 1024, kernel_initializer = 'normal', activation = 'relu'))
model.add (Dropout(0.4))
model.add (Dense(units = 1024, kernel_initializer = 'normal', activation = 'relu'))
model.add (Dropout(0.4))
model.add (Dense(1, kernel_initializer = 'normal'))
model.compile(loss='mean_absolute_error', optimizer='adam')
model.fit(X_train, y_train ,batch_size = 20, epochs = 100, verbose=1)


y_prediction =  model.predict(X_test)

y_prediction.shape
#percentage error
from sklearn.metrics import mean_absolute_percentage_error
print("mean absolute percentage error is ==",mean_absolute_percentage_error(y_test,y_prediction)*100, "%")

from sklearn.metrics import r2_score
print("r2 score is ==",  r2_score(y_test,y_prediction))