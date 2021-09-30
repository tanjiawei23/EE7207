from scipy.io import loadmat
import numpy as np
import sklearn
from RBF import RBF
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC


# [Nsample x mFeature]
train_data = loadmat("data_train.mat")['data_train']  # [330 x 33]
train_label = loadmat("label_train.mat")['label_train']  # [330 x 1]
test_data = loadmat("data_test.mat")['data_test']  # [21 x 33]
# print(train_label)

# train_label = train_label + (train_label == -1)
# print(train_label)

# ------------------------------Normalization--------------------------------------
scaler = StandardScaler(copy=False)
train_data = scaler.fit_transform(train_data)
test_data = scaler.transform(test_data)

# ---------------------------4-fold cross validation--------------------------
kfold = KFold(n_splits=4)
print(kfold.get_n_splits(train_data), "-fold cross validation")

sigma = np.array([i for i in range(1, 1001)], dtype='float') * 0.001
svm_meanScore = 0
scores = []
for n in sigma:
    svm = SVC(C=1, kernel='rbf', gamma=n)
    scores_kfold = []
    for train_index, test_index in kfold.split(train_data):
        x_train, x_test = train_data[train_index], train_data[test_index]
        y_train, y_test = train_label[train_index], train_label[test_index]
        svm.fit(x_train, y_train.ravel())
        # score = np.mean(np.power(y_test - svm.predict(x_test), 2))
        y_pred = svm.predict(x_test).reshape(-1, 1)
        right = (y_test == y_pred)
        score = np.sum(right) / len(right)
        # print("decision_func=", svm.decision_function(x_test))
        # print("prediction=", svm.predict(x_test))
        scores_kfold.append(score)
    print("sigma=", n, "accuracy=", np.mean(scores_kfold))
    scores.append(np.mean(scores_kfold))
    if np.mean(scores_kfold) > svm_meanScore:
        svm_meanScore = np.mean(scores_kfold)
        opt_sigma = n
print("oprimun sigma=", opt_sigma, "accuracy: ", svm_meanScore)

output_data = pd.DataFrame()
output_data['sigma'] = np.array([0.001 * i for i in range(1, 1001)])
output_data['scores'] = np.array(scores).reshape(-1, 1)
print(output_data)
output_data.to_csv('./svm_result.csv', index=False)










