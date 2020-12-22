import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV


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

model = LassoCV(normalize=True)
model.fit(x_train, y_train)

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
y_pred = model.predict(x_test)

raw_test_data.loc[:, 'charges'] = y_pred
raw_test_data.to_csv('lasso_result.csv', index=False)
