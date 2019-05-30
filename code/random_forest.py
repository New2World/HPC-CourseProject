import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

# load the training and testing data 
npz_train = np.load("earthquake_train_enhanced.npz")
X_train = npz_train['X_train']
y_train = npz_train['y_train']

npz_test = np.load("earthquake_test_enhanced.npz")
X_test = npz_test['X_test']
y_test = npz_test['y_test']

# Use cross validation method to find the best parameters for the given data set
grid_model = RandomForestRegressor()

param_list = {'n_estimators': [50, 100, 150, 200, 250, 300, 320, 340, 360, 380, 400, 420, 440, 460, 480, 500],
              'bootstrap': [True, False],
              'max_features': ['auto', 'sqrt', 'log2'],
              'max_depth': [10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36]}

grid_search = GridSearchCV(grid_model, param_grid=param_list, n_jobs=16)

grid_search.fit(X_train, y_train)

print('-----------------------------------------------')
print("Best parameters\n{}\n".format(grid_search.best_params_))
print('-----------------------------------------------')

# use the best hyperparameters to initialize model
model = RandomForestRegressor(n_estimators=440, bootstrap=True, max_features='log2', max_depth=32)

# train the random forest
model.fit(X_train, y_train)

# for each segment in the test dataset, predict the time_to_failure
# and compare to the labeled value to calculate the mean absolute error (MAE)
y_predict = []
ctr = 0
for item in X_test:
    print(f'test number {ctr}')
    ctr += 1
    item = item.reshape(1, -1)
    y_predict.append(model.predict(item))
y_predict = np.array(y_predict)


print('-----------------------------------------------')
print("MAE is {}".format(np.mean(np.abs(y_predict - y_test))))
print('-----------------------------------------------')
