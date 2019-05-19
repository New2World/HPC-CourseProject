import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.svm import NuSVR, SVR
from sklearn.metrics import mean_absolute_error

npzfile = np.load("earthquake_train_enhanced.npz")
X_train = npzfile['X_train']
y_train = npzfile['y_train']
print "Load training data"

# grid search best hyperparameters
svr = NuSVR(gamma='scale')
param_grid = {
        'nu':[.2, .4, .5, .6, .8],
        'kernel':['poly','rbf'],
        'degree':[3,5,10],
        'C':[.1, .25, .5, .75, .9, 1.],
}
grid_search = GridSearchCV(svr, cv=10, param_grid=param_grid, n_jobs=16)
grid_search.fit(X_train, y_train)
# svr.fit(X_train, y_train)

del X_train, y_train

print ("Best parameters set:")
print (grid_search.best_params_)

npzfile = np.load('earthquake_test_enhanced.npz')
X_test = npzfile['X_test']
y_test = npzfile['y_test']
print "Load test data"

# use trained model to predict
y_test_pred = grid_search.predict(X_test)

# mean absolute error
mae = mean_absolute_error(y_test, y_test_pred)

print "MAE: {}".format(mae)
