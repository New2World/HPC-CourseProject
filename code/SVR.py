import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.svm import NuSVR, SVR

npzfile = np.load("earthquake_train.npz")
X_train = npzfile['X_train']
y_train = npzfile['y_train']
print "Load training data"

# grid search best hyperparameters
svr = SVR(gamma='scale')
param_grid = {
        'kernel':['poly','rbf'],
        'C':[.5, .9, 1.],
}
grid_search = GridSearchCV(svr, cv=10, param_grid=param_grid, n_jobs=16)
grid_search.fit(X_train, y_train)

del X_train, y_train

npzfile = np.load('earthquake_test.npz')
X_test = npzfile['X_test']
y_test = npzfile['y_test']
print "Load test data"

# use trained model to predict
y_test_pred = svr.predict(X_test)

# mean absolute error
mae = np.mean(np.abs(y_test-y_test_pred))

print "MAE: {}".format(mae)
