## Assignment 1 of EE7207 (Neural & Fuzzy Network) 2021
This article is the assignment for the EE7207 course of EEE College, Nanyang Technological University.

This experiment evaluates the performance of Radial Basis Function Neural Network and Gaussian Kernel Support Vector Machine in binary classification tasks.

---
### Setup
Tested on Python 3.6.8

Clone this directory and install Python requirements:
```
pip install -r requirements.txt
```

### Get results
1. Using 4-fold cross-validation to tune parameters and plot:
```
python3 cv_rbf.py
python3 cv_svm.py
python3 plot.py
```

2. evaluate the performance on traning data:
```
python3 trainingset_result.py
```

3. predict the labels of test data:
```
python3 testset_predict.py
```

### Results
[cv_svm]: fig_results/4-fold_crossvalidationforGaussian-kernelSVM.png
[cv_rbf]: fig_results/4-fold_crossvalidationforRBFnetwork.png
[compare]: fig_results/compare_trainresults.png
- 4-fold cross-validation of kernel SVM:
    ![alt text][cv_svm]
- 4-fold cross-validation of RBF network (with 3 center vectors selection):
    ![alt text][cv_rbf]
- average accuracy on training set:
    ![alt text][compare]




