from scipy.io import loadmat
import numpy as np
import sklearn
from RBF import RBF
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import pandas as pd


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

numberHiddenNeurons = [i for i in range(2, 101, 1)]
rbf_accuracy = 0
optNeurons = 0
scores = []
for n in numberHiddenNeurons:
    rbf = RBF(33, n, 1, centers_find=2)
    scores_kfold = []
    for train_index, test_index in kfold.split(train_data):
        x_train, x_test = train_data[train_index], train_data[test_index]
        y_train, y_test = train_label[train_index], train_label[test_index]
        rbf.W = None
        rbf.fit(x_train, y_train)
        # score = np.mean(np.power(y_test - rbf.test(x_test), 2))
        y_pred = rbf.predict_class(x_test).reshape(-1, 1)
        right = (y_pred == y_test)
        score = np.sum(right) / right.shape[0]
        scores_kfold.append(score)
    print(n, "neurons:", np.mean(scores_kfold))
    scores.append(np.mean(scores_kfold))
    if np.mean(scores_kfold) > rbf_accuracy:
        rbf_accuracy = np.mean(scores_kfold)
        optNeurons = n
print("oprimun:", optNeurons, "neurons, accuracy: ", rbf_accuracy)

output_data = pd.DataFrame()
output_data['num_neurons'] = np.array([i for i in range(2, 101, 1)])
output_data['scores'] = np.array(scores).reshape(-1, 1)
print(output_data)
output_data.to_csv('./rbf_som_result.csv', index=False)

# plt.figure(figsize=(15, 10))
# plt.grid()
# plt.plot([i for i in range(2, 50, 5)], scores, '-')
# plt.show()





















