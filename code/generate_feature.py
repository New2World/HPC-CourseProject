import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt

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

training_data = pd.read_csv("../train.csv", chunksize=150000,
                         dtype={'acoustic_data':np.int16,
                         'time_to_failure':np.float64})
# episode = 0
X_train = []
X_test = []
y_train = []
y_test = []

# total episode: 4194
for data_chunk in training_data:
    x = data_chunk.values[:,0]
    y = data_chunk.values[:,1]
    if iter < 4000:
        X_train.append(get_features(x))
        y_train.append(y)
    else:
        X_test.append(get_features(x))
        y_test.append(y)

print ("training set size: {}".format(len(X_train)))
print ("test set size: {}".format(len(X_test)))

X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

np.savez("earthquake_train.npz", X_train=X_train, y_train=y_train)
print ("Train data saved")
np.savez("earthquake_test.npz", X_test=X_test, y_test=y_test)
print ("Test data saved")
