from numpy import *
from scipy.linalg import norm, pinv
from sklearn.cluster import KMeans
from minisom import MiniSom


class RBF:
    """
    centers_find = {
    0: random centers
    1: k-means
    2: SOM network
    }
    """
    def __init__(self, indim, numCenters, outdim, centers_find=0):
        self.indim = indim
        self.outdim = outdim
        self.numCenters = numCenters
        self.centers = [random.uniform(-1, 1, indim) for i in range(numCenters)]
        self.beta = None  # beta = 1 / (2sigma**2)
        self.W = random.random((self.numCenters, self.outdim))
        self.centers_find = centers_find

    def _basisfunc(self, c, d):
        assert len(d) == self.indim
        return exp(-self.beta * norm(c - d) ** 2)

    def _calcAct(self, X):
        # calculate activations of RBFs
        G = zeros((X.shape[0], self.numCenters), float)
        for ci, c in enumerate(self.centers):
            for xi, x in enumerate(X):
                G[xi, ci] = self._basisfunc(c, x)
        return G

    def fit(self, X, Y):
        """ X: matrix of dimensions n x indim
            y: column vector of dimension n x 1 """
        if self.centers_find == 0:
            # choose random center vectors from training set
            rnd_idx = random.permutation(X.shape[0])[:self.numCenters]
            self.centers = [X[i, :] for i in rnd_idx]
        elif self.centers_find == 1:
            kmeans = KMeans(n_clusters=self.numCenters).fit(X)
            self.centers = kmeans.cluster_centers_
        else:
            som = MiniSom(self.numCenters, 1, X.shape[1], sigma=0.1)
            som.train(X, num_iteration=1000)
            self.centers = np.squeeze(som.get_weights())

        # calculate sigma
        d_max = 0
        for i_center in self.centers:
            for j_center in self.centers:
                d_max = max(d_max, norm(i_center - j_center))
        if d_max != 0:
            sigma = d_max / sqrt(2 * self.numCenters)
            self.beta = 1 / (2 * sigma ** 2)
        else:
            print("d_max = 0")
            self.beta = 8

        # calculate activations of RBFs
        G = self._calcAct(X)

        # calculate output weights (pseudoinverse)
        self.W = dot(pinv(G), Y)

    def predict(self, X):
        """ X: matrix of dimensions n x indim """

        G = self._calcAct(X)
        Y = dot(G, self.W)
        return Y

    def predict_class(self, X):
        #  two class for Y = 1 / -1
        Y = self.predict(X)
        false_mask = Y < 0
        y_class = np.array(np.ones(Y.shape), dtype='int') + false_mask * (-2)
        return y_class

