"""
Simple graph analysis project for a class.
Most of the function names and return structures are set specifically
for use with a machine grader, but the rest of the code is my own work,
unless otherwise noted. Written for python 2.7.
"""

import matplotlib.pyplot as plt
import random

EX_GRAPH0 = {0: set([1, 2]), 1: set([]), 2: set([])}
EX_GRAPH1 = {0: set([1, 4, 5]), 1: set([2, 6]), 2: set([3]),
             3: set([0]), 4: set([1]), 5: set([2]), 6: set([])}
EX_GRAPH2 = {0: set([1, 4, 5]), 1: set([2, 6]), 2: set([3, 7]),
             3: set([7]), 4: set([1]), 5: set([2]), 6: set([]),
             7: set([3]), 8: set([1, 2]), 9: set([0, 3, 4, 5, 6, 7])}


def make_complete_graph(num_nodes):
    """
    Returns a dictionary representing a complete
    directed graph with num_nodes vertices.
    """
    
    graph = {}
    if num_nodes > 0 and isinstance(num_nodes, int):
        for vertex in range(num_nodes):
            edges = range(num_nodes)
            edges.remove(vertex)
        graph[vertex] = set(edges)
    return graph
    else:
        return graph

def compute_in_degrees(digraph):
    """
    Returns the in-degree of each of the nodes
    in a directed graph represented by a dict.
    """
    
    vals = []
    for verticies in digraph:
        for values in digraph[verticies]:
            vals.append(values)
    degrees = {}
    for vertex in digraph.keys():
        degrees[vertex] = vals.count(vertex)
    
    return degrees

def in_degree_distribution(digraph):
    """
    Returns the unnormalized in-degree distributions of a directed graph.
    """
    
    dist = {}
    degrees = compute_in_degrees(digraph).values()
    for degree in degrees:
       dist[degree] = degrees.count(degree)

    return dist

def load_graph(graph_file):
    """
    Loads a graph given text representation of the graph.
    Returns a dictionary that models the graph.
    """
    
    with open(graph_file) as graph:
        graph_lines = graph.read().split('\n')[:-1]
    
    print "Loaded graph with", len(graph_lines), "nodes"
    answer_graph = {}
    for line in graph_lines:
        neighbors = line.split(' ')
        node = int(neighbors[0])
        answer_graph[node] = set([])
        for neighbor in neighbors[1 : -1]:
            answer_graph[node].add(int(neighbor))
    return answer_graph

def normalize_in_degree(digraph):
    """
    Returns the normailzed distribution of in-degrees of graph nodes.
    """
    normal_dist = {}
    total_nodes = len(digraph.keys())
    in_degrees = in_degree_distribution(digraph)
    for degree in in_degrees.keys():
        normal_dist[degree] = in_degrees[degree] / float(total_nodes)
    
    return normal_dist

def er(n, p):
    """
    Creates a dict representation of a canonical ER graph.
    """
    
    graph = {}
    for i in range(n):
        edges = []
        for j in range(n):
            a = random.random()
            if a < p:
                edges.append(j)
        if i in edges:
            edges.remove(i)
        graph[i] = set(edges)
    return graph


class DPATrial:
    """
    Encapsulates optimized trials for the DPA algorithm. Here nodes
    are generated according to specified probabilities.
    """

    def __init__(self, num_nodes):
        """
        Initializes a DPATrial object as a complete graph with num_nodes.
        Part of this code was provided by the instructors.
        """

        self._num_nodes = num_nodes
        self._node_numbers = [node for node in range(num_nodes) 
                              for dummy_idx in range(num_nodes)]

        def run_trial(self, num_nodes):
            """
            Runs num_node trials using random.choice(), and updates
            the list of node numbers for that each node number occurs
            according to input probabilities.
            """

            new_node_neighbors = set()
            for dummy_idx in range(num_nodes):
                new_node_neighbors.add(random.choice(self._node_numbers))

            self._node_numbers.append(self._num_nodes)
            self._node_numbers.extend(list(new_node_neighbors))

            self._num_nodes += 1
            return new_node_neighbors

def dpa_graph(num,num_nodes):
    """
    Uses DPATrials to construct a DPA graph.
    """
    graph = make_complete_graph(num_nodes)
    dpa_obj = DPATrial(num_nodes)
    for idx in range(num_nodes, num):
        graph[idx] = dpa_obj.run_trial(num_nodes)
    return graph


#--------------------------------------------------------------------
#
# Several plots were to be generated as part of the project.
#
#--------------------------------------------------------------------

# load the provided citations data
citation_graph = load_graph('alg_phys-cite.txt')

# create a log-log plot of the normalized citation distribution
cite_dist = normalize_in_degree(citation_graph)
plt.loglog(cite_dist.keys(), cite_dist.values(), 'bo')
plt.title(r'Normalized Distribution of Citations (log/log)')
plt.ylabel(r'Normalized Degree Distribution ($log_{10}$)')
plt.xlabel(r'In-Degree ($log_{10}$)')
plt.savefig('cite_plot.png')
plt.show()

# create a plot of ER Graph node distribution
er_dist = normalize_in_degree(er(1000, 0.1))
plt.loglog(er_dist.keys(), er_dist.values(), 'bo')
plt.title(r'Normalized Distribution of Random ER Graph with p= 0.1 (log/log)')
plt.xlabel(r'In-Degree ($log_{10}$)')
plt.ylabel(r'Normalized Degree Distribution ($log_{10}$)')
plt.savefig('er_plot.png')
plt.show()

# create plot of the DPA graps distributions
dpa_dist = normalize_in_degree(dpa_graph(27770, 13))
plt.loglog(dpa_dist.keys(), dpa_dist.values(), 'bo')
plt.title(r'Normalized Distribution of DPA Graph (log/log)')
plt.xlabel(r'In-Degree ($log_{10}$)')
plt.ylabel(r'Normalized Degree Distribution ($log_{10}$)')
plt.savefig('dpa_plot.png')
plt.show()
