from sklearn.model_selection import train_test_split
from scipy.io import loadmat
import numpy as np
import sklearn
from RBF import RBF
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import auc


train_data = loadmat("data_train.mat")['data_train']  # [330 x 33]
train_label = loadmat("label_train.mat")['label_train']  # [330 x 1]
test_data = loadmat("data_test.mat")['data_test']  # [21 x 33]

# ------------------------------Normalization--------------------------------------
scaler = StandardScaler(copy=False)
train_data = scaler.fit_transform(train_data)
test_data = scaler.transform(test_data)

scores_RBF_random = []
scores_RBF_som = []
scores_RBF_kmeans = []
scores_svm = []
for epoch in range(100):
    x_train, x_test, y_train, y_test = train_test_split(
        train_data, train_label, test_size=0.33, shuffle=True)
    rbf_random = RBF(33, 93, 1, centers_find=0)
    rbf_kmeans = RBF(33, 66, 1, centers_find=1)
    rbf_som = RBF(33, 27, 1, centers_find=2)
    svm = SVC(kernel='rbf', gamma=0.052)
    rbf_random.fit(x_train, y_train)
    rbf_kmeans.fit(x_train, y_train)
    rbf_som.fit(x_train, y_train)
    svm.fit(x_train, y_train.ravel())
    y_pred_rr = rbf_random.predict_class(x_test).reshape(-1, 1)
    y_pred_rk = rbf_kmeans.predict_class(x_test).reshape(-1, 1)
    y_pred_rs = rbf_som.predict_class(x_test).reshape(-1, 1)
    y_pred_svm = svm.predict(x_test).reshape(-1, 1)
    right_rr = (y_test == y_pred_rr)
    right_rk = (y_test == y_pred_rk)
    right_rs = (y_test == y_pred_rs)
    right_svm = (y_test == y_pred_svm)
    scores_RBF_random.append(np.sum(right_rr) / right_rr.shape[0])
    scores_RBF_kmeans.append(np.sum(right_rk) / right_rk.shape[0])
    scores_RBF_som.append(np.sum(right_rs) / right_rs.shape[0])
    scores_svm.append(np.sum(right_svm) / right_svm.shape[0])
    if (epoch + 1) % 10 == 0:
        print(epoch+1, "epoch")
    if epoch == 0:
        # print(classification_report(y_test, y_pred_rr, target_names=["class -1", "class 1"]))
        # print(classification_report(y_test, y_pred_rk, target_names=["class -1", "class 1"]))
        # print(classification_report(y_test, y_pred_rs, target_names=["class -1", "class 1"]))
        # print(classification_report(y_test, y_pred_svm, target_names=["class -1", "class 1"]))
        fpr_svm, tpr_svm, thresholds = roc_curve(y_test, y_pred_svm, pos_label=1)
        svm_auc = auc(fpr_svm, tpr_svm)
        print("auc=", svm_auc)
        plt.plot(fpr_svm, tpr_svm, marker='o')
        plt.title("ROC curve of SVM")
        # plt.show()
    x_train, x_test, y_train, y_test = None, None, None, None
print(scores_RBF_random, "\n average=", np.mean(scores_RBF_random))
print(scores_RBF_kmeans, "\n average=", np.mean(scores_RBF_kmeans))
print(scores_RBF_som, "\n average=", np.mean(scores_RBF_som))
print(scores_svm, "\n average=", np.mean(scores_svm))

x_bar = ["RBF_random", "RBF_kmeans", "RBF_som", "SVM"]
y_bar = [np.mean(scores_RBF_random), np.mean(scores_RBF_kmeans),
         np.mean(scores_RBF_som), np.mean(scores_svm)]

fig = plt.figure(figsize=(10, 7))
plt.bar(x_bar, y_bar, 0.4, color="red", log=True)
x_position = 0
for y in y_bar:
    plt.text(x_position, y, '%.3f' % y, ha='center', va= 'bottom',fontsize=11)
    x_position += 1
plt.xlabel("4 different classifiers")
plt.ylabel("average accuracy")
plt.title("average accuracy of 100 independent experiments")
plt.show()
# plt.savefig("compare results")





