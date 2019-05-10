import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV
from sklearn.svm import NuSVR, SVR

def plot_ad_ttf_data(train_ad_sample, train_ttf_sample, title="Acoustic data and time to failure: sampled data"):
    _, ax1 = plt.subplots(figsize=(12, 8))
    plt.title(title)
    plt.plot(train_ad_sample, color='r')
    ax1.set_ylabel('acoustic data', color='r')
    plt.legend(['acoustic data'], loc=(0.01, 0.95))
    ax2 = ax1.twinx()
    plt.plot(train_ttf_sample, color='b')
    ax2.set_ylabel('time to failure', color='b')
    plt.legend(['time to failure'], loc=(0.01, 0.9))
    plt.grid(True)

def get_features(X):
    feat = []
    X_abs = np.abs(X)
    feat.append(X_abs.mean())
    feat.append(X_abs.std())
    feat.append(X_abs.max())
    feat.append(stats.kurtosis(X_abs))
    feat.append(stats.skew(X_abs))
    feat.append(np.quantile(X_abs, .01))
    feat.append(np.quantile(X_abs, .05))
    feat.append(np.quantile(X_abs, .95))
    feat.append(np.quantile(X_abs, .99))
    X_fft = np.fft.fft(X).astype(np.float64)
    feat.append(X_fft.mean())
    feat.append(X_fft.std())
    feat.append(X_fft.max())
    # add more
    return np.array(feat)

def feature_generator(data, chunksize=300, span=100):
    iters = 629145480 / span - chunksize + 1
    data_value = data.values[::span]
    for i in xrange(iters):
        chunk_data = data_value[i:i+chunksize]
        yield get_features(chunk_data[:,0]), chunk_data[:,1][-1]

training_data = pd.read_csv("../train.csv", chunksize=150000, 
                            dtype={'acoustic_data':np.int16,
                            'time_to_failure':np.float64})

episode = 0
X_train = []
X_test = []
y_train = []
y_test = []

# total episode: 4194
for each_chunk in training_data:
    episode += 1
    generate_feature = feature_generator(each_chunk)
    for i in xrange(1201):
        feature, ttf = generate_feature.next()
        if episode < 4000:
            X_train.append(feature)
            y_train.append(ttf)
        else:
            X_test.append(feature)
            y_test.append(ttf)
    print "episode #{}".format(episode)
    if episode >= 4196:
        break

y_test = np.array(y_test)

svr = SVR(gamma='scale')
param_grid = {
        'kernel':['poly','rbf'],
        'C':[.5, .9, 1.],
}
grid_search = GridSearchCV(svr, cv=10, param_grid=param_grid, n_jobs=16)
grid_search.fit(X_train, y_train)

y_test_pred = svr.predict(X_test)

mae = np.mean(np.abs(y_test-y_test_pred))

print "MAE: {}".format(mae)
