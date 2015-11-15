"""
All of these functions are just some recapitulations of similar
projects which were completed in Octave for a class, and are NOT to be
considered a complete machine learning library. See scikitlearn or
caret (R), or similar for much more robust and fully featured
machine learning libraries.
"""

import numpy as np

#--------------------------------------------------------------------
#
# a few utility functions
#
#--------------------------------------------------------------------


def load_data(datafile, norm = False, add_incpt=True):
    """
    Data loader utility function. Choose if the intercept
    should be added (1's). The observed values are assumed to
    be in the last column.
    """

    data = np.loadtxt(datafile, delimiter = ",")

    # the observed y values should be in the last column
    y = data[:,-1]
    m = y.size
    yr = y.reshape((m,1))
    X = data[:,0:-1]
    
    # normalize the features if desired
    if norm:
        X = feature_normalize(X)[0]
    
    # add a column of ones to the x values for the intercept, if desired
    if add_incpt:
        X = np.concatenate((np.ones((m, 1)), X), axis = 1)
    
    return X, yr


def map_feature(x1, x2, degree):
    """
    Maps two variables onto a set of polynomials and adds a
    column of ones for the intercept.

    Parameters
    ----------
    x1 : array of independent variables to be mapped
    x2 : array of independent variables to be mapped
    degree: degree of the polynomial dersired

    Returns
    -------
    mapped variables : array
    """

    mapped = np.ones((len(x1), 1))
    for i in range(1,degree+1):
        for j in range(i+1):
            mapped = np.concatenate((mapped, (x1**(i-j)*x2**j)),axis = 1)
    return mapped


def convert_labels_one_v_all(classes, labels):
    """
    Takes an array of determined classification values and 
    and array of the possible classes and converts them to a one vs all
    binary array, with index of where 1 is being the label in the same
    index as the label array.
    """

    return np.array((labels == classes), dtype = int)


def feature_normalize(X):
    """
    Normalizes each variable/feature based on the mean and 
    standard deviation of the observations for that variable.

    Parameters
    ----------
    X : array of independent variables

    Returns
    -------
    X : array
    mu : float
    stdev : float
    """

    mu = X.mean(axis=0)
    stdev = X.std(axis=0)
    X = X - mu
    X = X / stdev
    return X, mu, stdev

def train(X, y, cost, grad, init_theta, weight = 0):
    """
    Train to retrieve fitted theta values for an input X and y 
    using conjgant gradient from scipy, and the input gradient function
    and cost function.

    Parameters
    ----------
    X : array of independent variables
    y : array of dependent variables
    cost : string, the cost function to use
    grad : string, the gradient function to use
    init_theta : array, an inital set of starting values 
    weight : scalar multiplicative value to scale the regulariation

    Returns
    -------
    theta : array
    """

    initial_theta = np.zeros(X.shape[1])
    theta = optimize.fmin_cg(cost, init_theta, args = (X, y, weight),
                             fprime = grad, maxiter = 100,
                             full_output = True, disp = False)
    return theta


#--------------------------------------------------------------------
#
# Simple Linear Regression by gradient descent with a cost
# function implemtation. Note that if the data are well behaved,
# the a solution can be found directly by solving the Normal Equations
# or using lstsq must more quickly and efficiently.
#
#--------------------------------------------------------------------


def linreg_cost(theta, X, y):
    """
    Standard linear regression cost function.

    Parameters
    ----------
    theta : array of fitting parameters
    X : array of independent variables
    y : array of dependent variable

    Returns
    -------
    cost : float
    """

    # set up some useful values and make sure theta is a column vector
    m = y.shape[0]
    t = theta.shape
    
    if theta.ndim > 1:
        theta = theta.reshape((t[0], t[1]))
    else:
        theta = theta[:,np.newaxis]
    
    cost = (1.0 / (2*m)) * np.dot((np.dot(X, theta) - y).T, 
                                  (np.dot(X, theta) - y))
    return np.asscalar(cost)


def gradient_descent(theta, X, y, alpha, num_iter):
    """
    Simple gradient descent

    Parameters
    ----------
    theta : array of fitting parameters
    X : array of independent variables
    y : array of dependent variable
    alpha : learning rate (step size)
    num_iter : number of iterations

    Returns
    -------
    theta : array
    cost_history : array
    """

    # set up some useful values and make sure theta is a column vector
    m = y.shape[0]
    t = theta.shape
    cost_history = []
    
    if theta.ndim > 1:
        theta = theta.reshape((t[0], t[1]))
    else:
        theta = theta[:,np.newaxis]
   
    for dummy_i in range(num_iter):
        gradient = (1.0/m) * np.dot(X.T, (np.dot(X, theta) - y))
        theta = theta - alpha * gradient
        cost_history.append(linreg_cost(theta, X, y))   
    return theta, cost_history


def normal_eqn(X, y):
    """
    Explicit solving of closed form normal equation.

    Parameters
    ----------
    X : array of independent variables
    y : array of dependent variables

    Returns
    -------
    theta : array
    """

    m = y.shape[0]
    return np.dot(np.dot(np.linalg.pinv(np.dot(X.T, X)), X.T), y)


#--------------------------------------------------------------------
#
# Regularized Linear Regression
#
#--------------------------------------------------------------------


def rlinreg_cost(theta, X, y, weight = 0):
    """
    Cost function for regularized linear regression.
    Parameters
    ----------
    theta : array of fitting parameters
    X : array of independent variables
    y : array of dependent variable
    weight : scalar multiplicative value to scale the regulariation

    Returns
    -------
    cost : float
    """
    
    # set up some useful values and make sure theta is a column vector
    m = y.shape[0]
    t = theta.shape
    
    if theta.ndim > 1:
        theta = theta.reshape((t[0], t[1]))
    else:
        theta = theta[:,np.newaxis]
    
    h = np.dot(X, theta)
    cost = (1.0 / (2*m)) * np.sum((h-y)**2) + (weight / (2.0*m)) * \
            np.sum(theta[1:,:]**2)
    return cost


def rlinreg_grad(theta, X, y, weight = 0):
    """
    Gradient function for regularized linear regression.

    Parameters
    ----------
    theta : array of fitting parameters
    X : array of independent variables
    y : array of dependent variable
    weight : scalar multiplicative value to scale the regulariation

    Returns
    -------
    gradient : array
    """

    # set up some useful values and make sure theta is a column vector
    m = y.shape[0]
    t = theta.shape
    
    if theta.ndim > 1:
        theta = theta.reshape((t[0], t[1]))
    else:
        theta = theta[:,np.newaxis]
    
    h = np.dot(X, theta)
    gradient = (1.0/m) * np.dot(X.T, (h-y)) + (weight/float(m)) * theta[1:,:]
    
    return gradient.flatten()


#--------------------------------------------------------------------
#
# Simple Logistic Regression
#
#--------------------------------------------------------------------


def sigmoid(z):
    """
    Computes the sigmoid of z.

    Parameters
    ----------
    z : scalar or array of values

    Returns
    -------
    sigmoid : float
    """

    return 1.0 / (1 + np.exp(-z))


def logreg_cost(theta, X, y):
    """
    Logistic regression cost function.

    Parameters
    ----------
    theta : array of fitting parameters
    X : array of independent variables
    y : array of dependent variable

    Returns
    -------
    cost : float
    """

    # set up some useful values and make sure theta is a column vector
    m = y.shape[0]
    t = theta.shape
    
    if theta.ndim > 1:
        theta = theta.reshape((t[0], t[1]))
    else:
        theta = theta[:,np.newaxis]
    
    h = sigmoid(np.dot(X, theta))
    cost = (1.0/m) * (np.dot(-y.T, np.log(h)) - 
                        np.dot((1.0 - y.T), np.log(1.0 - h)))
    return np.asscalar(cost)
    

def logreg_grad(theta, X, y):
    """
    Computes the gradient (f') of the logistic regression cost function.

    Parameters
    ----------
    theta : array of fitting parameters
    X : array of independent variables
    y : array of dependent variable

    Returns
    -------
    gradient : array
    """

    # set up some useful values and make sure theta is a column vector
    m = y.shape[0]
    t = theta.shape
    
    if theta.ndim > 1:
        theta = theta.reshape((t[0], t[1]))
    else:
        theta = theta[:,np.newaxis]

    h = sigmoid(np.dot(X, theta))
    gradient = (1.0/m) * np.dot(X.T, (h-y))
    return gradient.flatten()


#--------------------------------------------------------------------
#
# Logisitic Regression with regularization
#
#--------------------------------------------------------------------


def rlogreg_cost(theta, X, y, weight = 0):
    """
    Regularized logistic regression cost function
    
    Parameters
    ----------
    theta : array of fitting parameters
    X : array of independent variables
    y : array of dependent variable
    weight : scalar multiplicative value to scale the regulariation

    Returns
    -------
    cost : float
    """

    # set up some useful values and make sure theta is a column vector
    m = y.shape[0]
    t = theta.shape
    
    if theta.ndim > 1:
        theta = theta.reshape((t[0], t[1]))
    else:
        theta = theta[:,np.newaxis]

    h = sigmoid(np.dot(X, theta))
    cost = ((1.0/m) * (np.dot(-y.T, np.log(h)) - \
                         np.dot((1.0 - y.T), np.log(1.0 - h)))) + \
            ((weight / (2.0 * m)) * np.dot(theta[1:,:].T, theta[1:,:]))
    return np.asscalar(cost)


def rlogreg_grad(theta, X, y, weight = 0):
    """
    Computes the gradient (f') of the logistic regression cost 
    function with regularization.

    Parameters
    ----------
    theta : array of fitting parameters
    X : array of independent variables
    y : array of dependent variable
    weight : scalar multiplicative value to scale the regulariation

    Returns
    -------
    gradient : array
    """

    # set up some useful values and make sure theta is a column vector
    m = y.shape[0]
    t = theta.shape
    
    if theta.ndim > 1:
        theta = theta.reshape((t[0], t[1]))
    else:
        theta = theta[:,np.newaxis]
    
    # make a new theta paramter for regularization because scoping works 
    # oddly in scipy.optimize
    thetaR = np.copy(theta)
    thetaR[0] = 0
    
    h = sigmoid(np.dot(X, theta))
    gradient = (1.0/m) * np.dot(X.T, (h-y)) + (weight / float(m)) * thetaR
    return gradient.flatten()


def logreg_predict(theta, X):
    """
    Predicts the binary class to which a new observation
    belongs to based on the learned theta.

    Parameters
    ----------
    theta : array of learned parameters
    X : array of indepedented variables used to predict

    Returns
    -------
    classes : array of True or False values
    """
    return sigmoid(np.dot(X, theta)) >= 0.5


#--------------------------------------------------------------------
#
# Neural Network Classifcation
#
#--------------------------------------------------------------------


def nn_predict(theta1, theta2, X):
    """
    Neural network forward propagation by providing the 
    prediction label given trained theta values from a 3 layer
    neural network of 10 units in the output layer
    r network, with 10 units (for each of 10 digits) in the output layer.

    Parameters
    ----------
    theta1 : array of fitted parameters for activation layer
    theta2 : array of fitted parameters for output layer

    Returns
    -------
    predictions : array
    """

    # set useful vals
    m = len(X)
    num_labels = len(theta2)
    
    # calculate hidden layer activations
    a2 = sigmoid(np.dot(X,theta1.T))
    
    # add ones for the bias unit
    a2 = np.concatenate((np.ones((len(a2),1)), a2), axis=1)
    
    # calculate the output layer
    h = sigmoid(np.dot(a2, theta2.T))
    
    # find the max value (highest probability) for prediction
    return np.argmax(h, axis=1)


def nn_sigmoid_gradient(z):
    """
    Calculates the gradient (f') of the sigmoid function at each point.
    
    Parameters
    ----------
    z : array of values

    Returns
    -------
    gradient : array
    """

    return sigmoid(z) * (1 - sigmoid(z))


def nn_rand_initial_theta(l_in, l_out):
    """
    Initialize inital guesses based on the size layers.

    Parameters
    ----------
    l_in : int, number of inputs to the layer
    l_out : int, number of outputs the layer emits

    Returns
    -------
    random theta : array
    """

    # range for initial values
    r_init = np.sqrt(6) / np.sqrt(l_in+l_out)
    return np.random.rand(l_out, 1+l_in) * 2 * r_init - r_init


def nn_cost(thetas, input_size, hidden_size, num_labels, X, y, weight = 0):
    """
    Cost function for training a neural network (feedforward)
    Three layer network, one input, one hidden, one output only.

    Parameters
    ----------
    thetas : array of theta values, input layer followed by hidden layer
    X : array of independent variables
    y : array of dependent variable classes

    Returns
    -------
    cost : float
    """

    # set up some useful values and reshape the theta arrays
    m = X.shape[0]
    theta1 = np.reshape(thetas[:hidden_size * (input_size+1)],
                        (hidden_size, input_size + 1))
    theta2 = np.reshape(thetas[(hidden_size * (input_size+1)):],
                        (num_labels, hidden_size + 1))
    
    # compute the activation of the hidden layers
    a2 = sigmoid(np.dot(X, theta1.T))
    a2 = np.concatenate((np.ones((a2.shape[0], 1)), a2), axis = 1)
    h = sigmoid(np.dot(a2, theta2.T))
    
    cost = (1.0/m) * np.sum(-y * np.log(h) - (1-y) * np.log(1-h)) + \
           (weight / (2.0 * m)) * (np.sum(theta1[:,1:]**2) + 
                                   np.sum(theta2[:,1:]**2))
    return cost


def nn_grad(thetas, input_size, hidden_size, num_labels, X, y, weight = 0):
    """
    Calculates the back progagation and gradient of a neural network.

    Parameters
    ----------
    thetas : array of theta values, input layer followed by hidden layer
    X : array of independent variables
    y : array of dependent variable classes

    Returns
    -------
    gradient : array
    """

    # set up some useful values and reshape the theta arrays
    m = X.shape[0]
    theta1 = np.reshape(thetas[:hidden_layer_size * (input_layer_size+1)],
                        (hidden_layer_size, input_layer_size + 1))
    theta2 = np.reshape(thetas[(hidden_layer_size * (input_layer_size+1)):],
                        (num_labels, hidden_layer_size + 1))
    
    z2 = np.dot(X, theta1.T)
    a2 = sigmoid(z2)
    a2 = np.concatenate((np.ones((a2.shape[0], 1)), a2), axis = 1)
    h = sigmoid(np.dot(a2, theta2.T))
    d3 = h - y
    d2 = np.dot(d3, theta2[:, 1:]) * nn_sigmoid_gradient(z2)
    D1, D2 = np.dot(d2.T, X), np.dot(d3.T, a2)
    
    g1 = (D1 / float(m)) + (weight / float(m)) * np.concatenate((
        np.zeros((theta1.shape[0],1)), theta1[:, 1:]), axis = 1)
    g2 = (D2 / float(m)) + (weight / float(m)) * np.concatenate((
        np.zeros((theta2.shape[0],1)), theta2[:, 1:]), axis = 1)
    
    gradient = np.append(g1.flatten(), g2.flatten())
    return gradient


#--------------------------------------------------------------------
#
# K-means clustering
#
#--------------------------------------------------------------------

def pair_distance(x, y, norm = True):
    """
    Computes the euclidean distance between two pairs. 
    Either squared L2-norm or euclidean norm can
    be used. When norm = True the L2 norm of the distance is used.
    
    Parameters
    ----------
    x : float, centroid location on one axis
    y : float, centroid location on one axis
    norm : whether to use the L2 norm for distance

    Returns
    -------
    distance : float
    """
    if norm:
        return np.dot(x,x) - 2 * np.dot(x, y) + np.dot(y,y)
    else:
        return np.sqrt(np.dot(x,x) - 2 * np.dot(x, y) + np.dot(y,y))


def find_closest(X, centroids):
    """
    Finds the closest centroid for each observation.
    Use for k-means cllustering only, as there are faster 
    ways for hierarchical clustering.
    
    Parameters
    ----------
    X : array of variables
    centroids : array of centroids

    Returns
    -------
    closest centroid for each variable : array
    """

    # set up some useful values
    size_x = X.shape[0]
    k = centroids.shape[0]
    
    results = [min([(pair_distance(X[i,:], centroids[j]), j)
                    for j in range(k)]) for i in range(size_x)]
    vals = [result[0] for result in results]
    indices = [result[1] for result in results]
    return indices, vals


def compute_centroids(X, idx, num_centroids):
    """
    Calculates the new centroids based on the mean center
    of the points assigned to an old centroid.
    
    Parameters
    ----------
    X : array of variables
    idx : index
    num_centroids : int, number of centroids

    Returns
    -------
    centroids : array
    """

    centroids = np.zeros((num_centroids, X.shape[1]))
    for i in range(num_centroids):
        centroids[i,:] = np.mean(np.compress(idx == i, X, axis=0), axis=0)
    return centroids

    
def kmeans(X, num_clusters, num_iters = 20, thresh = 1e-5):
    """
    Classifies data points into clusters based on closest centroid 
    for num_clusters and num_iters
    
    Parameters
    ----------
    X : array of variables
    num_clusters : int, number of clusters to make
    num_iters : int, number of iterations
    thresh : threshhold

    Returns
    -------
    clusters : array
    """
    
    init_centroids = np.array([X[np.random.randint(0,X.shape[0]),:]
                               for i in range(num_clusters)])
    centroids = init_centroids
    iterations = 0
    prev_distortion, current_distortion = 1000, 0
    
    for dummy_i in range(num_iters):
        if abs(prev_distortion - current_distortion) <= thresh:
            return closest, centroids, iterations
        else:
            closest = find_closest(X, centroids)
            prev_distortion, current_distortion = current_distortion, sum(closest[1])
            centroids = compute_centroids(X, np.asarray(closest[0]), num_clusters)
            iterations +=1
    return closest, centroids, iterations


#--------------------------------------------------------------------
#
# Principal compnents analysis
#
#--------------------------------------------------------------------

def pca(X):
    """
    Computes the eigenvectors of the covariance matrix.
    
    Parameters
    ----------
    X :  array of variables

    Returns
    -------
    eigenvectors, eigenvalues : array    
    """

    sigma = (1.0 / X.shape[0]) * np.dot(X.T, X)
    return np.linalg.svd(sigma)


def project_data_pca(X, U, k):
    """
    Reduces data by computing the projection of normalized X onto
    the first k dimensions of eigenvectors.
     
    Parameters
    ----------
    X : array of variables
    U : array of eigenvectors
    k : int, number of dimensions 

    Returns
    -------
    projection : array
    """

    return np.dot(X, U[:,:k])

def recover_projected_data(Z, U, k):
    """
    Recover a best approximation of data which has been
    projected into lower dimensions using PCA.
    
    Parameters
    ----------
    Z : array of projected data
    U : array of eigenvectors
    k : same k as used to projection
    
    Returns
    -------
    data : array
    """

    return np.dot(Z, U[:,:k].T)


#--------------------------------------------------------------------
#
# Anomaly detection and collaborative filtering
#
#--------------------------------------------------------------------

def multivarpdf(X, mean, cov):
    """
    Computes the probability density function of an array of variables.
    Same as stats.describe from scipy.

    Parameters
    ----------
    X : array of variables
    mean : mean
    cov : covariance matrix or vector

    Returns
    -------
    pdf : float
    """

    # set up some values and decide if cov is a covariance matrix
    # or a vector to be treated as the diagonal cov matrix
    k = len(mean)
    if cov.shape[0] == 1 or cov.ndim == 1:
        cov = np.diag(cov)
    X -= mean
    pdf = (2.0 * np.pi) ** (-k / 2.0) * np.linalg.det(cov) ** (-0.5) * \
           np.exp(-0.5 * np.sum(np.dot(X, np.linalg.pinv(cov)) * X, axis = 1)) 
    return pdf
    

def thresh_select(yval, pval):
    """
    Use cross-validation set to determine a good threshold 
    value for anomaly detection (outliers)
    
    Parameters
    ----------
    yval : array of ground truth variables
    pval : validation set results
    
    Returns
    -------
    threshold : float
    """

    # set up some useful values
    best_thresh = 0
    best_f1, f1 = 0, 0
    min_p, max_p = np.min(pval), np.max(pval)
    stepsize = (max_p - min_p) / 1000.0
    
    def frange(start, stop, step):
        while start < stop:
            yield start
            start += step
    
    for threshold in frange(min_p, max_p, stepsize):
        # if the probability is less than espison, it is considered an anomaly
        predictions = np.array((pval < threshold), dtype = int)
        
        # true positives
        tp = sum([predictions[i] == 1 and yval[i] == 1
                  for i in range(len(predictions))])
        
        # false positives
        fp = sum([predictions[i] == 1 and yval[i] == 0
                  for i in range(len(predictions))])
        
        # false negatives
        fn = sum([predictions[i] == 0 and yval[i] == 1
                  for i in range(len(predictions))])
        
        if tp+fp == 0:
            precision = np.inf
        else:
            precision = float(tp) / (tp + fp)
        if tp+fn == 0:
            recall = np.inf
        else:
            recall = float(tp) / (tp + fn)
        f1 = (2.0 * precision * recall) / (precision + recall)
    
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = threshold
            
    return best_thresh, np.asscalar(best_f1)


def recommend_cost(theta, X, y, r, num_users, num_items, num_features, weight = 0):
    """
    Implememtation of a cost function for recommending an items 
    based on user ratings and features of users and items
    using collaborative filtering.
    
    Parameters
    ----------
    theta : array of fitted coefficients
    X : array of variables
    y : array of ratings
    r : binary indicator matrix
    num_users : number of users of the system
    num_items : number of items in the system
    num_features : number of variables
    weight : regularization weight

    Returns
    -------
    cost : float    
    """

    # reshape because optimize wants thetas to be flat
    theta = theta.reshape((num_users, num_features))
    cost = (1.0 / 2) * np.sum((np.dot(X, theta.T) - y)**2 * r) + \
           ((weight / 2.0) * np.sum(theta**2) + (weight / 2.0) * np.sum(X**2))
    return cost


def recommend_grad(theta, X, y, r, num_users, num_items, num_features, weight = 0):
    """
    Computes the gradient (f') for the const function above.

    Parameters
    ----------
    theta : array of fitted coefficients
    X : array of variables
    y : array of ratings
    r : binary indicator matrix
    num_users : number of users of the system
    num_items : number of items in the system
    num_features : number of variables
    weight : regularization weight

    Returns
    -------
    grad : float
    """

    theta = theta.reshape((num_users, num_features))
    x_grad =     np.dot(((np.dot(X, theta.T) - y) * r), theta) + weight * X
    theta_grad = np.dot(((np.dot(X, theta.T) - y) * r).T, X) + (weight * theta)
    gradient = np.append(x_grad, theta_grad)
    return theta_grad.flatten()





