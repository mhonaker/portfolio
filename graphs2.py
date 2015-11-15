"""
Second set of graph analysis projects for a class.
Most of the function names and return structures were required
for the machine grader, but all of the work is my own
unless otherwise noted. Throughout, 'attack' refers to node removal.
Written for python 2.7.
"""

import random
import time
import math
import matplotlib.pyplot as plt
from collections import deque


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
    
def make_complete_graph(num_nodes):
    """
    Returns a dictionary representing a complete
    directed graph with num_nodes vertices.
    """

    graph = {}
    for node in range(num_nodes):
        graph[node] = set(range(num_nodes))
        graph[node].remove(node)
    return graph

def copy_graph(graph):
    """
    Makes an independent copy of a graph.
    """
    new_graph = {}
    for node in graph:
        new_graph[node] = set(graph[node])
    return new_graph

def delete_node(ugraph, node):
    """
    Delete a node from an undirected graph.
    """

    neighbors = ugraph[node]
    ugraph.pop(node)
    for neighbor in neighbors:
        ugraph[neighbor].remove(node)


def bfs_visited(ugraph, start_node):
    """
    Takes an undirected graph and start node and returns the set of all
    nodes visited following a breadth first search algorithm.
    Returns the set of nodes connected to the start node incuding 
    the start node.
    """

    the_queue = deque()
    visited = set([start_node])
    the_queue.append(start_node)
    while the_queue:
        node  = the_queue.pop()
        for neighbor in ugraph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                the_queue.append(neighbor)
    return visited

def cc_visited(ugraph):
    """
    Takes a graph and returns the set of connected components.
    """

    remaining_nodes = set(ugraph)
    connected_components = []
    while remaining_nodes:
        component = bfs_visited(ugraph, remaining_nodes.pop())
        connected_components.append(component)
        remaining_nodes.difference_update(component)
    return connected_components

def largest_cc_size(ugraph):
    """
    Takes an undirected graph and returns the size (number of nodes)
    of the largest connected component
    """

    biggest = 0
    for component in cc_visited(ugraph):
        if len(component) > biggest:
            biggest = len(component)
    return biggest

def er(n, p):
    """
    Creates a dict representation of the canonical ER graph.
    """

    graph = {}
    for node in range(n):
        graph[node] = set([]) 
    for node_a in range(0, n-1):
        for node_b in range(node_a+1, n):
            if random.random() < p:
                graph[node_a].add(node_b)
                graph[node_b].add(node_a)
    return graph

class UPATrial:
    """
    Simple class to implement optimized trials for UPA algorithm
    Part of this class was provided by the instructors.
    """
    
    def __init__(self, num_nodes):
        """
        Initialize a UPATrial object cooresponding to a complete graph
        with num_node verticies.
        """

        self._num_nodes = num_nodes
        self._node_numbers = [node for node in range(num_nodes)
                              for dummy_idx in range(num_nodes)]

    def run_trial(self, num_nodes):
        """
        Conduct num_nodes trials by appying random.choice()
        to the list of node numbers.
        Updates the list of node numbers so that each one appears in the 
        correct ratio and returns a set of nodes.
        """

        #compute the neighbors for the new node
        new_node_neighbors = set()
        for _ in range(num_nodes):
            new_node_neighbors.add(random.choice(self._node_numbers))
        
        #update the list of node numbers so that each one appears in the
        #correct ratio
        self._node_numbers.append(self._num_nodes)
        for dummy_idx in range(len(new_node_neighbors)):
            self._node_numbers.append(self._num_nodes)
        self._node_numbers.extend(list(new_node_neighbors))

        self._num_nodes += 1

        return new_node_neighbors
    
def UPAgraph(n,m):
    """
    A random UPA graph is iteratively created where in each iteration a new node is created
    added to the graph, and connected to a subset of existing nodes. the subset is chosen based
    on the connection degree of the existing nodes. n is the final number of nodes in the graph
    and m is is the number od existing nodes to which the new node is connected
    during each iteration.
    """

    #start with a complete graph of m nodes
    graph = make_complete_graph(m)
    for node in range(m,n):
        graph[node] = set()
    #use the UPATrial class to create the graph
    upa_obj = UPATrial(m)
    for i in range(m,n):
        new_neighbors = upa_obj.run_trial(m)
        graph[i].update(new_neighbors)
        for j in new_neighbors:
            graph[j].add(i)
    return graph

def random_order(ugraph):
    """
    Takes a graph and returns a list of its nodes in a random order.
    """

    a = ugraph.keys()
    random.shuffle(a)
    return a

def targeted_order(ugraph):
    """
    Compute a targeted attack order consisting of nodes
    of maximum degree. Returns a list of nodes.
    """

    new_graph = copy_graph(ugraph)
    order = []
    while len(new_graph) > 0:
        max_degree = -1
        for node in new_graph:
            if len(new_graph[node]) > max_degree:
                max_degree = len(new_graph[node])
                max_degree_node = node

        neighbors = new_graph[max_degree_node]
        new_graph.pop(max_degree_node)
        for neighbor in neighbors:
            new_graph[neighbor].remove(max_degree_node)

        order.append(max_degree_node)
    return order    

def fast_targeted_order(ugraph):
    """
    As above, provides a list of nodes for an attack order, but with
    a more efficient implementation.
    """

    degree_sets = [set([]) for x in range(len(ugraph))]
    degree_dict = {}

    #create a list of sets of nodes with each degree. include an 
    #empty set for a degree that has no examples
    for node in ugraph: 
        degree_dict[node] = len(ugraph[node])
        degree_sets[len(ugraph[node])].add(node)
    
    order = []
    #iterate though the list of sets of nodes with each degree
    #in reverse order (highest to lowest)
    for degree_a in range(len(ugraph)-1, -1, -1):
        #if the set at that degree is not empty
        while degree_sets[degree_a]:
            node_a = degree_sets[degree_a].pop()
            for neighbor in ugraph[node_a]:
                ugraph[neighbor].remove(node_a)
                #get the degree of the neigbor and shift it down one
                d = degree_dict[neighbor] 
                degree_sets[d].remove(neighbor)
                degree_sets[d-1].add(neighbor)
                degree_dict[neighbor] -= 1
            order.append(node_a)    
    return order            

def compute_resilience(ugraph, attack_order):
    """
    Takes an undirected graph and computes the resilience
    it returns a list of the largest connected components 
    following removal of each node in the attcak order list.
    """

    graph = copy_graph(ugraph)
    res_list = [largest_cc_size(ugraph)]
    for node in attack_order:
        edges = graph.pop(node)
        for node_b in edges:
            graph[node_b].remove(node)
        res_list.append(largest_cc_size(graph))
    return res_list

def generate_edges(graph):
    """
    Returns the number of edges in an input graph.
    """
    edges = []
    for node in graph:
        for neighbor in graph[node]:
            edges.append((node, neighbor))
    return len(edges)/2.0

def timing():
    nodes = []
    tar_time = []
    fast_tar_time = []
    for n in range(10, 1000, 10):
        nodes.append(n)

        graph = UPAgraph(n, 5)
        
        time1 = time.time()
        a = targeted_order(graph)
        time2 = time.time()
        tar_time.append(time2-time1)

        time3 = time.time()
        b = fast_targeted_order(graph)
        time4 = time.time()
        fast_tar_time.append(time4-time3)
    
    result = []
    result.append(nodes)
    result.append(tar_time)
    result.append(fast_tar_time)

    return result

#--------------------------------------------------------------------
#
# Several plots were genrated as part of the project
#
#--------------------------------------------------------------------

# the resilience of three types of graphs were determined and
# plots were generated based on random and targeted attacks

verticies = 1347
edges = 3112
prob = float(edges) / ((verticies * (verticies-1)) / 2.0) 
m_val = 2

er_graph = er(verticies, prob)
upa_graph = UPAgraph(verticies, m_val)
net_graph = load_graph("alg_rf7.txt")


x_points = range(verticies+1)

# random attck order
y_er = compute_resilience(ergraph, random_order(er_graph))
y_upa = compute_resilience(upagraph, random_order(upa_graph))
y_net = compute_resilience(net_graph, random_order(net_graph))

plt.plot(x_points, y_er, linewidth=2.0, label=r'ER graph ($p=0.003$)')
plt.plot(x_points, y_upa, linewidth=2.0, label=r'UPA graph($m=2$)')
plt.plot(x_points, y_net, linewidth=2.0, label=r'Network graph')
plt.legend(loc = 'upper right')
plt.title('Resilience of Graphs to Random Attack')
plt.xlabel('Total Number of Graph Nodes Removed')
plt.ylabel('Size of Largest Connected Component (# of nodes)')
plt.savefig('random_attack.png')
plt.show()

# targeted attck order
y_er = compute_resilience(ergraph, fast_targeted_order(er_graph))
y_upa = compute_resilience(upagraph, fast_targeted_order(upa_graph))
y_net = compute_resilience(net_graph, fast_targeted_order(net_graph))

plt.plot(x_points, y_er, linewidth=2.0, label=r'ER graph ($p=0.003$)')
plt.plot(x_points, y_upa, linewidth=2.0, label=r'UPA graph($m=2$)')
plt.plot(x_points, y_net, linewidth=2.0, label=r'Network graph')
plt.title('Resilience of Graphs to Targeted Attack')
plt.xlabel('Total Number of Graph Nodes Removed')
plt.ylabel('Size of Largest Connected Component (# of nodes)')
plt.savefig('targeted_attack.png')
plt.show()


# a comparison of the time involved for the targeted_order (O(n^2))
# vs fast_targeted_order (O(n))

points = timing()
print points
x_points = points[0]
y_points_target = points[1]
y_points_fast_target = points[2]

plt.plot(x_points, y_points_target, linewidth = 2.0, label = r'Targeted Order')
plt.plot(x_points, y_points_fast_target, linewidth = 2.0, label = r'Fast Targeted Order')
plt.legend(loc = 'upper left')
plt.title('Time of Calculating Attack Order in Desktop Python')
plt.xlabel('Number of Nodes in Graph')
plt.ylabel('Time for Computation (s)')
plt.savefig('target_time.png')
plt.show()
