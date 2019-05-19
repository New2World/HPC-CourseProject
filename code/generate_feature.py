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
    full_len = X.shape[0]
    one_third = (full_len-1)/3
    two_third = one_third*2
    x1 = X[:one_third]
    x2 = X[one_third+1:two_third]
    x3 = X[two_third+1:]
    # X_abs = np.abs(X)
    feat.append(x1.mean())
    feat.append(x2.mean())
    feat.append(x3.mean())
    feat.append(x1.std())
    feat.append(x2.std())
    feat.append(x3.std())
    feat.append(x1.max())
    feat.append(x2.max())
    feat.append(x3.max())
    feat.append(x1.min())
    feat.append(x2.min())
    feat.append(x3.min())
    feat.append(stats.kurtosis(x1))
    feat.append(stats.kurtosis(x2))
    feat.append(stats.kurtosis(x3))
    feat.append(stats.skew(x1))
    feat.append(stats.skew(x2))
    feat.append(stats.skew(x3))
    feat.append(np.quantile(x1, .01))
    feat.append(np.quantile(x2, .01))
    feat.append(np.quantile(x3, .01))
    feat.append(np.quantile(x1, .05))
    feat.append(np.quantile(x2, .05))
    feat.append(np.quantile(x3, .05))
    feat.append(np.quantile(x1, .95))
    feat.append(np.quantile(x2, .95))
    feat.append(np.quantile(x3, .95))
    feat.append(np.quantile(x1, .99))
    feat.append(np.quantile(x2, .99))
    feat.append(np.quantile(x3, .99))
    x1_fft = np.fft.fft(x1).astype(np.float64)
    x2_fft = np.fft.fft(x2).astype(np.float64)
    x3_fft = np.fft.fft(x3).astype(np.float64)
    feat.append(x1_fft.mean())
    feat.append(x2_fft.mean())
    feat.append(x3_fft.mean())
    feat.append(x1_fft.std())
    feat.append(x2_fft.std())
    feat.append(x3_fft.std())
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
iter = 0

# total episode: 4194
for data_chunk in training_data:
    x = data_chunk.values[:,0]
    y = data_chunk.values[:,1]
    if iter < 4000:
        X_train.append(get_features(x[:75000]))
        y_train.append(y[74999])
        X_train.append(get_features(x[75001:]))
        y_train.append(y[-1])
    else:
        full_len = x.shape[0]
        X_test.append(get_features(x[:full_len]))
        y_test.append(y[(full_len-1)/2])
        X_test.append(get_features(x[(full_len-1)/2+1:]))
        y_test.append(y[-1])
    iter += 1
    print ("#{} finshed".format(iter))

print ("training set size: {}".format(len(X_train)))
print ("test set size: {}".format(len(X_test)))

X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

np.savez("earthquake_train_enhanced.npz", X_train=X_train, y_train=y_train)
print ("Train data saved")
np.savez("earthquake_test_enhanced.npz", X_test=X_test, y_test=y_test)
print ("Test data saved")
