"""
A set of small projects on cluster analysis and algorithms. 
Most function names and return structures were required for
the machine grader. All other work is my own unless otherwise
noted. Written for python 2.7.
"""

import math
import random
import time
import timeit
import alg_cluster
import matplotlib.pyplot as plt


def pair_distance(cluster_list, idx1, idx2):
    """
    Helper function to take a list of cluster objects and return 
    the Euclidean distance and ordered indicies of two of them
    as a tuple of the distance, idx1, idx2.
    """
    
    return (cluster_list[idx1].distance(cluster_list[idx2]), min(idx1, idx2), max(idx1, idx2))

def slow_closest_pairs(cluster_list):
    """
    Takes a list of cluster objects and returns a set with the clusters that have
    minimum distance between them. A set of tuples (dist, idx1, idx2)
    is returned whith the minimum distance of the clusters in the list.

    This function is a implementation of the brute-force algorithm (and will be O(n^2))
    """

    current_min_dist = float("inf")
    result = set([])
    for idx1 in range(len(cluster_list) - 1):
        for idx2 in range(idx1+1, len(cluster_list)):
            dist = pair_distance(cluster_list, idx1, idx2)
            if dist[0] == current_min_dist:
                result.add(dist)
            elif dist[0] < current_min_dist:
                result = set([dist])
                current_min_dist = dist[0]
    return result           

def fast_closest_pair(cluster_list):
    """
    Takes a list of cluster objects and return a tuple
    (distance, idx1, idx2) where the indicies are ordered
    and cluster_list[idx1], cluster_list[idx2] have the
    smallest distance of the clusters in the list

    An implementaion of the recursive 
    divide and conquer algorithm (O(n log n)).
    """

    def fast_helper(cluster_list, horiz_order, vert_order):
        """
        Does the actual divide and conquer while the outer function 
        sets up the ordering of the horizontal and verticle indices.
        """
        if len(horiz_order) <= 3:
            dist1 = tuple(slow_closest_pairs([cluster_list[idx]
                                              for idx in horiz_order]))[0]
            dist2 = (dist1[0], horiz_order[dist1[1]], horiz_order[dist1[2]])
            return dist2
        else:
            mid = len(horiz_order) / 2
            mid_line = (cluster_list[horiz_order[mid-1]].horiz_center() +
                        cluster_list[horiz_order[mid]].horiz_center()) / float(2)
            horz_l = horiz_order[:mid]
            horz_r = horiz_order[mid:]
            vert_l = [idx for idx in vert_order if idx in set(horz_l)]
            vert_r = [idx for idx in vert_order if idx in set(horz_r)]
            
            r_dist = fast_helper(cluster_list, horz_l, vert_l)
            l_dist = fast_helper(cluster_list, horz_r, vert_r)
            dist = min(r_dist, l_dist)
            
            some_list = [idx for idx in vert_order if
                         abs(cluster_list[idx].horiz_center() - 
                             mid_line) < dist[0]]
            if some_list:
                for elem1 in range(len(some_list)-1):
                    for elem2 in range(elem1 + 1, min(elem1+3, len(some_list))):
                            dist = min(dist, pair_distance(
                                cluster_list, some_list[elem1],
                                some_list[elem2]))
                return dist
            else:
                return dist
        

    #horizontal indicies , ordered
    hcoord_idx = [(cluster_list[idx].horiz_center(), idx)
                  for idx in range(len(cluster_list))]
    
    hcoord_idx.sort()
    horiz_order = [hcoord_idx[idx][1] for idx in range(len(hcoord_idx))]

    #vertical indicies, ordered
    vcoord_idx = [(cluster_list[idx].vert_center(), idx)
                  for idx in range(len(cluster_list))]
    
    vcoord_idx.sort()
    vert_order = [vcoord_idx[idx][1] for idx in range(len(vcoord_idx))]

    result = fast_helper(cluster_list, horiz_order, vert_order)
    return (result[0], min(result[1:]), max(result[1:]))

def hierarchical_clustering(cluster_list, num_clusters):
    """
    Takes a list of clusters and a the number of new clusters
    desired and returns a list of clusters (lenght num_clusters)
    that are the new clusters reclustered in a hierarchical manner.
    """

    while len(cluster_list) > num_clusters:
        closest = fast_closest_pair(cluster_list)
        cluster_list[closest[1]].merge_clusters(cluster_list[closest[2]])
        cluster_list.pop(closest[2])
    return cluster_list 

def kmeans_clustering(cluster_list, num_clusters, num_iterations):
    """
    Takes a list of clusters, the number of desired clusters, and 
    the number of iterations, and returns a new list of clusters.
    the new clusters are computed using kmeans algorithm.
    """

    #make a list of the population totals and index of clusters
    #and sort them (ascending order of population, beacuse it is easy)
    population_list = [(cluster_list[idx].total_population(), idx)
                       for idx in range(len(cluster_list))]
    
    population_list.sort()
    
    #initialize num_cluster centers using the highest populations
    init_center_list = []

    for center in range(-1, -1*num_clusters-1, -1):
        init_center_list.append(alg_cluster.Cluster(
            set([]),cluster_list[population_list[center][1]].horiz_center(),
            cluster_list[population_list[center][1]].vert_center(), 0, 0))
    
    for dummy_i in range(num_iterations):
        init_cluster_list = [alg_cluster.Cluster(
            set([]), cluster.horiz_center(), cluster.vert_center(), 0, 0) 
            for cluster in init_center_list]

        for cluster_idx in range(len(cluster_list)):
            dist = min([(init_center_list[idx].distance(
                cluster_list[cluster_idx]), idx, cluster_idx)
                for idx in range(len(init_center_list))])
            init_cluster_list[dist[1]].merge_clusters(cluster_list[cluster_idx].copy())
        
        init_center_list = init_cluster_list 
    return init_center_list

def gen_random_clusters(num_clusters):
    """
    Generates num_clusters random clusters in a +-1 square grid
    at a random point in range(100, -10, -20).
    """

    cluster_list = []
    for cluster in range(num_clusters):
        cluster_list.append(alg_cluster.Cluster(
            set([]), random.uniform(-1, 1), random.uniform(-1,1), 0, 0))
    
    return cluster_list

def compute_distortion(data_file, num_clusters, c_type):
    """
    Uses the cluster error function in the Cluster class to 
    compute a distortion factor between two clusters.
    """

    data_table = load_data(data_file)
    singleton_list = []
    for line in data_table:
        singleton_list.append(alg_cluster.Cluster(
            set([line[0]]), line[1], line[2], line[3], line[4]))
    
    if c_type == 'hierarchical':
        cluster_list = hierarchical_clustering(singleton_list, num_clusters)
    elif c_type == 'kmeans':
        cluster_list = kmeans_clustering(singleton_list, num_clusters, 5)
    else:
        print 'wrong cluster type'

    errors = []
    for cluster in cluster_list:
        errors.append(cluster.cluster_error(data_table))
    return sum(errors)  

def load_data(data_file):
    """
    A function to load a data file.
    """
    
    with open(data_file) as data:
        clusters = data.read().split('\n')
    print 'Loaded', len(clusters), 'data points'
    data_tokens = [line.split(',') for line in clusters]
    
    return [[tokens[0], float(tokens[1]), float(tokens[2]),
             int(tokens[3]), float(tokens[4])] for tokens in data_tokens]


def timing():
	slow = []
	fast = []
	result = []

	for n in range(2, 201):
		c_list = gen_random_clusters(n)
		
		
		a = timeit.timeit(lambda: slow_closest_pairs(c_list), number=1)

		slow.append(a)

		b = timeit.timeit(lambda: fast_closest_pair(c_list), number=1)

		fast.append(b)

	result.append(slow)
	result.append(fast)
	return result

def distortion(data):
    """
    Calculates the distortions using two different clustering
    methods (k-means and hierarchical).
    """

	distortions_h = []
	distortions_k = []
	for i in range(6,21):
		a = compute_distortion(data, i, 'hierarchical')
		distortions_h.append(a)

		b = compute_distortion(data, i, 'kmeans')
		distortions_k.append(b)

	results = []
	results.append(distortions_h)
	results.append(distortions_k)

	return results


#--------------------------------------------------------------------
#
# generate some plots for the project
#
#--------------------------------------------------------------------

# comparison of the slow and fast versions of closest pairs
x_points = range(2,201)
y_points = timing()
y_slow = y_points[0]
y_fast = y_points[1]

plt.plot(x_points, y_slow, linewidth=2.0, color='Teal', label='Slow Closest Pairs')
plt.plot(x_points, y_fast, linewidth=2.0, color='Orange', label='Fast Closest Pairs')
plt.legend(loc = 'upper left')
plt.title('Running Time of Closest Pair Algorithms (Desktop Python)')
plt.xlabel('Number of Initial Random Clusters')
plt.ylabel('Time of Computation (s)')
plt.savefig('slow_fast.png')
plt.show()


# comparison of the distortions between k-means and hierarchical
# clustering methods for different size data sets

def create_distortion_plots():
    x_points = range(6,21)
    for num in ['111', '290', '896', '3108']:
        y_points = distortion('unifiedCancerData_' + '.csv')
        plt.plot(x_points, y_points[0], linewidth=2.0, color='Maroon',
                 label='Hierarchical Clustering')
        plt.plot(x_points, y_points[1], linewidth=2.0, color='Blue',
                 label='K-means Clustering')
        plt.legend(loc='upper right')
        plt.title('Clustering Distortion of ' + num + ' County Dataset')
        plt.xlabel('Output Clusters')
        plt.ylabel('Distortion')
        plt.savefig(num + 'distortion.png')
        plt.show()

create_distortion_plots()

