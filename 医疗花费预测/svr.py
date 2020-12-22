import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


training_data = pd.read_csv('public_dataset/train.csv')

training_data.loc[training_data['sex'] == 'female', 'sex'] = 0
training_data.loc[training_data['sex'] == 'male', 'sex'] = 1
training_data.rename(columns={'sex': 'is_male'}, inplace=True)

training_data.loc[training_data['smoker'] == 'yes', 'smoker'] = 1
training_data.loc[training_data['smoker'] == 'no', 'smoker'] = 0

training_data.loc[:, 'northeast'] = training_data['region'] == 'northeast'
training_data.loc[:, 'northwest'] = training_data['region'] == 'northwest'
training_data.loc[:, 'southeast'] = training_data['region'] == 'southeast'
training_data.loc[:, 'southwest'] = training_data['region'] == 'southwest'
del training_data['region']

x_train = training_data[['age', 'is_male', 'bmi', 'children', 'smoker', 'northeast', 'northwest', 'southeast', 'southwest']]\
    .to_numpy(dtype=np.float64)
y_train = training_data['charges'].to_numpy(dtype=np.float64)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
y_mean = np.mean(y_train)
y_std = np.std(y_train)
y_train = (y_train - y_mean) / y_std

parameters = {'C': np.logspace(0, 1, 10), 'epsilon': np.logspace(-1, 0, 10)}
model = GridSearchCV(SVR(), parameters, scoring='r2', n_jobs=4)
model.fit(x_train, y_train)
print('Best params:', model.best_params_)  # C=2.15, epsilon=0.13
print('Best score:', model.best_score_)  # 0.84

raw_test_data = pd.read_csv('public_dataset/test_sample.csv')
test_data = raw_test_data.copy()

test_data.loc[test_data['sex'] == 'female', 'sex'] = 0
test_data.loc[test_data['sex'] == 'male', 'sex'] = 1
test_data.rename(columns={'sex': 'is_male'}, inplace=True)

test_data.loc[test_data['smoker'] == 'yes', 'smoker'] = 1
test_data.loc[test_data['smoker'] == 'no', 'smoker'] = 0

test_data.loc[:, 'northeast'] = test_data['region'] == 'northeast'
test_data.loc[:, 'northwest'] = test_data['region'] == 'northwest'
test_data.loc[:, 'southeast'] = test_data['region'] == 'southeast'
test_data.loc[:, 'southwest'] = test_data['region'] == 'southwest'
del test_data['region']

x_test = test_data[['age', 'is_male', 'bmi', 'children', 'smoker', 'northeast', 'northwest', 'southeast', 'southwest']]\
    .to_numpy(dtype=np.float64)
x_test = scaler.transform(x_test) 
y_pred = model.predict(x_test) * y_std + y_mean

raw_test_data.loc[:, 'charges'] = y_pred
raw_test_data.to_csv('svr_result.csv', index=False)
