from sklearn.model_selection import train_test_split
from scipy.io import loadmat
import numpy as np
import sklearn
from RBF import RBF
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import pandas as pd


train_data = loadmat("data_train.mat")['data_train']  # [330 x 33]
train_label = loadmat("label_train.mat")['label_train']  # [330 x 1]
test_data = loadmat("data_test.mat")['data_test']  # [21 x 33]

# ------------------------------Normalization--------------------------------------
scaler = StandardScaler(copy=False)
train_data = scaler.fit_transform(train_data)
test_data = scaler.transform(test_data)

# -------------------------------training-------------------------------------------
rbf_som = RBF(33, 27, 1, centers_find=2)
svm = SVC(kernel='rbf', gamma=0.052)

rbf_som.fit(train_data, train_label)
svm.fit(train_data, train_label.ravel())

# -------------------------------predict test set-------------------------------------
y_pred_rbf = rbf_som.predict_class(test_data)
y_pred_svm = svm.predict(test_data)

output = pd.DataFrame()
output['RBF'] = np.array(y_pred_rbf.squeeze())
output['SVM'] = np.array(y_pred_svm.squeeze())
print(output)

output.to_csv("test_data_prediction.csv", index=True)



