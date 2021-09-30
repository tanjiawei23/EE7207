import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


rbf_random = pd.read_csv('results/rbf_random_result.csv')
rbf_kmeans = pd.read_csv('results/rbf_kmeans_result.csv')
rbf_som = pd.read_csv('results/rbf_som_result.csv')
svm = pd.read_csv('svm_result.csv')

num_neurons = rbf_random['num_neurons']
scores_random = rbf_random['scores']
scores_kmeans = rbf_kmeans['scores']
scores_som = rbf_som['scores']

sigma = svm['sigma']
scores_svm = svm['scores']

print("best:", end='\n')
print("random:", num_neurons[np.argmax(scores_random)], np.max(scores_random))
print("kmeans:", num_neurons[np.argmax(scores_kmeans)], np.max(scores_kmeans))
print("som:", num_neurons[np.argmax(scores_som)], np.max(scores_som))
print("svm:", sigma[np.argmax(scores_svm)], np.max(scores_svm))

plt.figure(figsize=(15, 10))
plt.xlabel('number of neurons')
plt.ylabel('accuracy')
plt.title('4-fold cross validation for RBF network')
plt.plot(num_neurons, scores_random, '-r')
plt.plot(num_neurons, scores_kmeans, '-b')
plt.plot(num_neurons, scores_som, '-k')
plt.legend(["rbf-random", "rbf-kmeans", "rbf-som"])

plt.figure(figsize=(15, 10))
plt.xlabel('sigma value')
plt.ylabel('accuracy')
plt.title('4-fold cross validation for Gaussian-kernel SVM')
plt.plot(sigma, scores_svm, '-r')
plt.legend(["Gaussian-kernel SVM"])
# plt.show()















