"""
Unittesting for some machine learing algorithms.
"""

import unittest
import numpy as np
from scipy import optimize, io
import machine_learning as ml

class TestCase(unittest.TestCase):

    def setUp(self):
        self.x1 = np.array([[0,0,0]]).T
        self.y1 = np.array([[1,1,1]]).T
        self.x2 = np.array([[1,2,3,4,5]]).T
        self.y2 = np.array([[1,2,3,4,5]]).T
        self.ex1data1 = ml.load_data('LinearRegression/ex1data1.txt')
        self.ex1data2 = ml.load_data('LinearRegression/ex1data2.txt', norm=True)
        self.ex2data1 = ml.load_data('LogisticRegression/ex2data1.txt')
        self.ex2data2 = ml.load_data('LogisticRegression/ex2data2.txt')
        self.ex3data1 = io.loadmat('RegularizedLogReg/ex3data1.mat')
        self.ex3weights = io.loadmat('RegularizedLogReg/ex3weights.mat')
        self.ex4data1 = io.loadmat('NeuralNetworks/ex4data1.mat')
        self.ex4weights = io.loadmat('NeuralNetworks/ex4weights.mat')
        self.ex5data1 = io.loadmat('RegularizedLinReg/ex5data1.mat')
        self.ex7data1 = io.loadmat('KmeansPCA/ex7data1.mat')
        self.ex7data2 = io.loadmat('KmeansPCA/ex7data2.mat')

    def test_linreg_cost(self):
        init_cost = ml.linreg_cost(np.zeros((2,1)),
                                   self.ex1data1[0],
                                   self.ex1data1[1])
        self.assertEqual(round(init_cost, 2), 32.07)
        

    def test_gradient_descent(self):
        ex1 = ml.gradient_descent(np.zeros((1,1)), self.x1,
                                  self.y1, 0.01, 500)[0].flatten()
        ex2 = ml.gradient_descent(np.zeros((1,1)), self.x2,
                                  self.y2, 0.1, 500)[0].flatten()
        ex3 = ml.gradient_descent(np.zeros((2,1)), self.ex1data1[0],
                                  self.ex1data1[1], 0.01, 5000)[0].flatten()
        ex4 = ml.gradient_descent(np.zeros((3,1)), self.ex1data2[0],
                                  self.ex1data2[1], 0.01, 1500)[0].flatten()
        self.assertEqual(ex1[0], 0.0)
        self.assertEqual(ex2[0], 1.0)
        self.assertEqual(round(np.sum(ex3), 1), -2.7)
        self.assertEqual(round(np.sum(ex4), 1), 443282.0)

    def test_normal_eqn(self):
        ex1 = ml.normal_eqn(self.x1, self.y1)
        ex2 = ml.normal_eqn(self.x2, self.y2)
        ex3 = ml.normal_eqn(self.ex1data1[0], self.ex1data1[1]).flatten()
        ex4 = ml.normal_eqn(self.ex1data2[0], self.ex1data2[1]).flatten()
        self.assertEqual(ex1[0], 0.0)
        self.assertEqual(ex2[0], 1.0)
        self.assertEqual(round(np.sum(ex3),1), -2.7)
        self.assertEqual(round(np.sum(ex4), 1), 443282.1)

    def test_logreg_cost(self):
        ex1 = ml.logreg_cost(np.zeros(3), self.ex2data1[0], self.ex2data1[1])
        ex2 = optimize.fmin(ml.logreg_cost, np.zeros(3),
                            args=(self.ex2data1[0], self.ex2data1[1]))
        ex3 = ml.logreg_cost(ex2, self.ex2data1[0], self.ex2data1[1])
        self.assertEqual(round(ex1, 3), 0.693)
        self.assertEqual(round(np.sum(ex2), 2), -24.75)
        self.assertEqual(round(ex3, 3), 0.203)
        

    def test_logreg_grad(self):
        ex1 = ml.logreg_grad(np.zeros(3), self.ex2data1[0], self.ex2data1[1])
        ex2 = optimize.fmin_cg(ml.logreg_cost, np.ones(3),
                               args=(self.ex2data1[0], self.ex2data1[1]),
                               fprime = ml.logreg_grad)
        self.assertEqual(round(np.sum(ex1),1), -23.4)
        self.assertEqual(round(np.sum(ex2), 2), -24.75)

    def test_map_features(self):
        # split the data into 2 arrays
        m = self.ex2data2[0][:,1:2]
        n = self.ex2data2[0][:,2:3]
        #map onto 6th degree polynomial
        z = ml.map_feature(m, n, 6)
        ex1 = ml.rlogreg_cost(np.zeros(28), z, self.ex2data2[1], weight=1)
        self.assertEqual(round(ex1, 3), 0.693)

    def test_logreg_predict(self):
        pass

    def test_nn_predict(self):
        #add ones for the intercept
        nndata = np.concatenate((np.ones((5000,1)), self.ex3data1['X']), axis=1)
        ex1 = ml.nn_predict(self.ex3weights['Theta1'],
                           self.ex3weights['Theta2'], nndata)
        self.assertEqual(round(np.mean(ex1.reshape((5000,1)) +
                                       1 == self.ex3data1['y']) * 100, 2), 97.52)
   
    def test_nn_cost(self):
        #add ones for the intercept
        nndata = np.concatenate((np.ones((5000,1)),
                                 self.ex4data1['X']), axis=1)
        #unroll thetas into one vector
        thetas = np.append(self.ex4weights['Theta1'].flatten(),
                           self.ex4weights['Theta2'].flatten())
        ymat = np.array((np.arange(1, 11) == self.ex4data1['y']), dtype=int)
        #with no regularization
        ex1 = ml.nn_cost(thetas, 400, 25, 10, nndata, ymat)
        #with regularization
        ex2 = ml.nn_cost(thetas, 400, 25, 10, nndata, ymat, 1)
        self.assertEqual(round(ex1, 4), 0.2876)
        self.assertEqual(round(ex2, 4), 0.3838)

    def test_nn_sigmoid_gradient(self):
        ex1 = ml.nn_sigmoid_gradient(np.array([1, -0.5, 0, 0.5, 1]))
        self.assertEqual(ex1.all(), np.array([0.196612, 0.235004, 0.250000,
                                              0.235004, 0.196612]).all())

    def test_rlinreg_cost(self):
        x1 = np.concatenate((np.ones((self.ex5data1['X'].shape[0],1)),
                             self.ex5data1['X']), axis=1)
        ex1 = ml.rlinreg_cost(np.ones((2,1)), x1, self.ex5data1['y'], 1)
        self.assertEqual(round(ex1, 2), 303.99)

    def test_rlinreg_grad(self):
        x1 = np.concatenate((np.ones((self.ex5data1['X'].shape[0],1)),
                             self.ex5data1['X']), axis=1)
        ex1 = ml.rlinreg_grad(np.ones((2,1)), x1, self.ex5data1['y'], 1)
        self.assertEqual(ex1.all(), np.array([-15.303016, 598.250744]).all())

    def test_find_closest(self):
        closest = ml.find_closest(self.ex7data2['X'],
                                  np.array([[3, 3], [6, 2], [8, 5]]))
        self.assertEqual(closest[0][0:3], [0,2,1])

    def test_compute_centroids(self):
        closest = ml.find_closest(self.ex7data2['X'],
                                  np.array([[3, 3], [6, 2], [8, 5]]))
        ex1 = ml.compute_centroids(self.ex7data2['X'],
                                   np.asarray(closest[0]), 3)
        self.assertEqual(ex1.all(), np.array([[2.428301, 3.157924],
                                              [5.813503, 2.633656],
                                              [7.119387, 3.616684]]).all())

    def test_kmeans(self):
        ex1 = ml.kmeans(self.ex7data2['X'], 3, 10)
        self.assertEqual(ex1[-2].all(),np.array([[1.953995, 5.025570],
                                                 [3.043671, 1.015410],
                                                 [6.033667, 3.000525]]).all())

    def test_pca(self):
        x1 = ml.feature_normalize(self.ex7data1['X'])[0]
        ex1 = ml.pca(x1)
        self.assertEqual(ex1[0].all(), np.array([[-0.707107,-0.707107],
                                                 [-0.707107, 0.707107]]).all())

    def test_project_data_pca(self):
        x1 = ml.feature_normalize(self.ex7data1['X'])[0]
        ex1 = ml.project_data_pca(x1, ml.pca(x1)[0], 1)
        self.assertEqual(round(ex1[0][0], 3), 1.496)


unittest.main()
