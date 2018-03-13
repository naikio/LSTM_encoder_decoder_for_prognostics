import numpy as np
import os

import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.externals import joblib

def moving_average(x, N=20):
    return np.convolve(x, np.ones(N,)/N, mode='valid')
# def moving_average(a, n=15):
#     ret = np.cumsum(a, dtype=float)
#     ret[n:] = ret[n:] - ret[:-n]
#     return ret[n - 1:] / n

# def moving_average(x, N=15):
#     cumsum = np.cumsum(np.insert(x, 0, 0))
#     return (cumsum[N:] - cumsum[:-N]) / float(N)

e_curves_train = joblib.load('e_curves_train.pkl')
e_curves_test = joblib.load('e_curves_test.pkl')
hi_curves_train = joblib.load('hi_curves_train.pkl')
original_pts_train = joblib.load('original_pts_train.pkl')
original_pts_test = joblib.load('original_pts_test.pkl')

regr = linear_model.LinearRegression()

X = np.array([e for e_curve in original_pts_train for e in e_curve])
Y = np.array([hi for hi_curve in hi_curves_train for hi in hi_curve])
# Train the model using the hi_curves obtained from train instances
regr.fit(X.reshape(-1,3), Y.reshape(-1,1))
# save model to disk
if not os.path.isfile('linear_regressor.model'):
    joblib.dump(regr, 'linear_regressor.model')


hi_curve_all_engines = []
for l in original_pts_test:
    hi_curve = regr.predict(X=l)
    hi_curve_all_engines.append(hi_curve)

if not os.path.isfile('hi_curves_test.pkl'):
    joblib.dump(hi_curve_all_engines, 'hi_curves_test.pkl')

for n, l in enumerate(hi_curve_all_engines):
    plt.plot(moving_average(np.squeeze(l), 10))
    plt.ylim((0,1))
    if n%10 == 0:
        pass
plt.show()

for n, l in enumerate(hi_curves_train):
    plt.plot(moving_average(np.squeeze(l), 10))
    plt.ylim((0,1))
    if n%10 == 0:
        pass
plt.show()
