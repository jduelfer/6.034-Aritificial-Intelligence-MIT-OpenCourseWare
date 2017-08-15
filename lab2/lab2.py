# Fall 2012 6.034 Lab 2: Search
#
# Your answers for the true and false questions will be in the following form.  
# Your answers will look like one of the two below:
#ANSWER1 = True
#ANSWER1 = False

# 1: True or false - Hill Climbing search is guaranteed to find a solution
#    if there is a solution
ANSWER1 = False

# 2: True or false - Best-first search will give an optimal search result
#    (shortest path length).
#    (If you don't know what we mean by best-first search, refer to
#     http://courses.csail.mit.edu/6.034f/ai3/ch4.pdf (page 13 of the pdf).)
ANSWER2 = False

# 3: True or false - Best-first search and hill climbing make use of
#    heuristic values of nodes.
ANSWER3 = True

# 4: True or false - A* uses an extended-nodes set.
ANSWER4 = True

# 5: True or false - Breadth first search is guaranteed to return a path
#    with the shortest number of nodes.
ANSWER5 = True

# 6: True or false - The regular branch and bound uses heuristic values
#    to speed up the search for an optimal path.
ANSWER6 = False

# Import the Graph data structure from 'search.py'
# Refer to search.py for documentation
from search import Graph
from sets import Set

## Optional Warm-up: BFS and DFS
# If you implement these, the offline tester will test them.
# If you don't, it won't.
# The online tester will not test them.
class Search:
	def __init__(self, graph, start, goal):
		self.graph = graph
		self.start = start
		self.goal = goal

		self.agenda = [start]
		self.visited = []
		self.parent = {}

	def search(self):
		while len(self.agenda) > 0:
			node = self.agenda.pop(0)
			self.visited += [node]
			if node == self.goal:
				return self.backtrack()
			connected_nodes = self.graph.get_connected_nodes(node)
			if connected_nodes is not None:
				self.add_to_agenda(node, connected_nodes)

	def validate_possible_extensions(self, node):
		validated = True
		if self.agenda and node in Set(self.agenda):
			validated = False
		if self.visited and node in Set(self.visited):
			validated = False
		return validated

	def validate_not_visited(self, node):
		validated = True
		if self.visited and node in Set(self.visited):
			validated = False
		return validated

	def backtrack(self):
		"""Backtracks from an end to a start given a dictionary of
		nodes as keys with their parents as values."""
		path = [self.goal]
		while path[-1] != self.start:
			path.append(self.parent[path[-1]])
		path.reverse()
		return path

class BFS(Search):
	def add_to_agenda(self, node, connected_nodes):
		nodes_to_add = []
		for connected_node in connected_nodes:
			if self.validate_possible_extensions(connected_node):
				nodes_to_add.append(connected_node)
				self.parent[connected_node] = node
		if nodes_to_add:
			self.agenda += nodes_to_add

class DFS(Search):
	def add_to_agenda(self, node, connected_nodes):
		order_connected_nodes = []
		for connected_node in connected_nodes:
			if self.validate_possible_extensions(connected_node):
				order_connected_nodes.append(connected_node)
				self.parent[connected_node] = node
		if order_connected_nodes:
			self.agenda = order_connected_nodes + self.agenda

class HillClimbing(Search):
	"""Almost the same as DFS but with two differences. We have to
	order the values before putting them to the top of the stack based
	upon distance to goal, and we don't care if a node is in the agenda
	for the future, we just care about visited nodes."""
	def add_to_agenda(self, node, connected_nodes):
		unordered_node_heuristics = []
		for connected_node in connected_nodes:
			if self.validate_not_visited(connected_node):
				heuristic = self.graph.get_heuristic(connected_node, self.goal)
				node_heuristic = NodeHeuristic(connected_node, heuristic)
				unordered_node_heuristics.append(node_heuristic)
				self.parent[connected_node] = node
		if unordered_node_heuristics:
			ordered_node_heuristics = sorted(unordered_node_heuristics, key=lambda x: x.heuristic)
			ordered_nodes = []
			for ordered in ordered_node_heuristics:
				ordered_nodes.append(ordered.node)
			self.agenda = ordered_nodes + self.agenda

class BeamSearch(Search):
	def __init__(self, graph, start, goal, beam_width):
		Search.__init__(self, graph, start, goal)
		self.beam_width = beam_width
		self.layer_by_node = dict()
		self.nodes_by_layer = dict()
		self.is_filtered_by_layer = dict()
		print('graph : ' + str(graph))
		print('beam width : ' + str(beam_width))

	def search(self):
		while len(self.agenda) > 0:
			self.filter_agenda()
			print('agenda after filter : ' + str(self.agenda))
			node = self.agenda.pop(0)
			self.visited += [node]
			if node == self.goal:
				return self.backtrack()
			connected_nodes = self.graph.get_connected_nodes(node)
			if connected_nodes is not None:
				self.add_to_agenda(node, connected_nodes)
		return []

	def add_to_agenda(self, node, connected_nodes):
		nodes_to_add = []
		for connected_node in connected_nodes:
			if self.validate_possible_extensions(connected_node):
				nodes_to_add.append(connected_node)
				self.parent[connected_node] = node
				if not self.layer_by_node.has_key(node):
					self.layer_by_node[node] = 0
					self.nodes_by_layer[0] = [node]
				self.layer_by_node[connected_node] = self.layer_by_node[node] + 1
				if not self.nodes_by_layer.has_key(self.layer_by_node[connected_node]):
					self.nodes_by_layer[self.layer_by_node[connected_node]] = []
				self.nodes_by_layer[self.layer_by_node[connected_node]].append(connected_node)
		if nodes_to_add:
			self.agenda += nodes_to_add

	def filter_agenda(self):
		node_to_pop = self.agenda[0]
		if self.layer_by_node.has_key(node_to_pop):
			current_layer = self.layer_by_node[node_to_pop]
			if not self.is_filtered_by_layer.has_key(current_layer):
				unordered_node_heuristics = []
				for node in self.agenda:
					if node in self.nodes_by_layer[current_layer]:
						heuristic = self.graph.get_heuristic(node, self.goal)
						node_heuristic = NodeHeuristic(node, heuristic)
						unordered_node_heuristics.append(node_heuristic)
				end_of_reorder = len(unordered_node_heuristics)
				if end_of_reorder > 0:
					self.agenda = self.agenda[end_of_reorder:] # temp until we sort
					ordered_node_heuristics = sorted(unordered_node_heuristics, key=lambda x: x.heuristic)
					ordered_nodes = []
					for index, value in enumerate(ordered_node_heuristics):
						if index < self.beam_width:
							ordered_nodes.append(value.node)
					self.agenda = ordered_nodes + self.agenda # reinsert ordered layer
				self.is_filtered_by_layer[current_layer] = True

class NodeHeuristic:
	def __init__(self, node, heuristic):
		self.node = node
		self.heuristic = heuristic


def bfs(graph, start, goal):
	bfs_search = BFS(graph, start, goal)
	return bfs_search.search()

def dfs(graph, start, goal):
	dfs_search = DFS(graph, start, goal)
	return dfs_search.search()


## Now we're going to add some heuristics into the search.  
## Remember that hill-climbing is a modified version of depth-first search.
## Search direction should be towards lower heuristic values to the goal.
def hill_climbing(graph, start, goal):
	hill_climbing_search = HillClimbing(graph, start, goal)
	return hill_climbing_search.search()

## Now we're going to implement beam search, a variation on BFS
## that caps the amount of memory used to store paths.  Remember,
## we maintain only k candidate paths of length n in our agenda at any time.
## The k top candidates are to be determined using the 
## graph get_heuristic function, with lower values being better values.
def beam_search(graph, start, goal, beam_width):
	beam_search_algo = BeamSearch(graph, start, goal, beam_width)
	return beam_search_algo.search()

## Now we're going to try optimal search.  The previous searches haven't
## used edge distances in the calculation.

## This function takes in a graph and a list of node names, and returns
## the sum of edge lengths along the path -- the total distance in the path.
def path_length(graph, node_names):
    raise NotImplementedError


def branch_and_bound(graph, start, goal):
    raise NotImplementedError

def a_star(graph, start, goal):
    raise NotImplementedError


## It's useful to determine if a graph has a consistent and admissible
## heuristic.  You've seen graphs with heuristics that are
## admissible, but not consistent.  Have you seen any graphs that are
## consistent, but not admissible?

def is_admissible(graph, goal):
    raise NotImplementedError

def is_consistent(graph, goal):
    raise NotImplementedError

HOW_MANY_HOURS_THIS_PSET_TOOK = ''
WHAT_I_FOUND_INTERESTING = ''
WHAT_I_FOUND_BORING = ''
