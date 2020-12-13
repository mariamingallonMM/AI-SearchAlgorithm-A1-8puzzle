"""
Project 1 of Columbia University's AI EdX course (8-puzzle).
Python 3
Author: Maria Mingallon
"""

# import required libraries for collection of metrics for outputs.txt file
import argparse
import math
import time
import sys
import os

# Specific libraries import for queues
## used for breadth first search function, FIFO queue     
from queue import Queue 
## used for A start algorithm, variant of queue that retrieves entries in priority order (lowest first)
from queue import PriorityQueue 

## The Class that Represents the Puzzle

class PuzzleState(object):

    """PuzzleState a class that generates the board for the 8-puzzle from:
    - config: a tuple representing the puzzle configuration, e.g. (0,1,2,3,4,5,6,7,8)
    - n: the number of keys per row and column which for the 8-puzzle should be equal to '3'
    - parent: a tuple representing the puzzle configuration preceeding the current one, default is 'None'
    - action: movement taken from parent to config states of the current puzzle configuration, default is 'Initial'
    - cost: keep track of the path cost, default is '0' """

    def __init__(self, config, n, parent=None, action="Initial", cost=int(0)):

        if n*n != len(config) or n < 2:
            raise Exception("the length of config is not correct!")
        
        self.config = config
        self.n = n
        self.parent = parent
        self.action = action
        self.cost = cost

        self.dimension = n
        self.children = []
        self.goal_state = (0,1,2,3,4,5,6,7,8)

        if parent:
            self.cost = int(parent.cost) + int(cost)
        else:
            self.cost = int(cost)

        for i, item in enumerate(self.config):

            if item == 0:

                self.blank_row = i // self.n
                self.blank_col = i % self.n

                break
 
    def __str__(self):
        """
        Style the plotting of the puzzle state into an array of 3 rows and 3 columns
        """
        return f'The current state of Puzzle is:\n [{self.config[0:3]} \n {self.config[3:6]} \n {self.config[6:9]}] \n' 

    def __file__(self):
        return driver.py
  
    def display(self):
        """
        Style the display of the puzzle state
        """
        for i in range(self.n):

            line = []
            offset = i * self.n

            for j in range(self.n):

                line.append(self.config[offset + j])

            #print(line)

    #Stack of functions for the possible moves of the '0' key: "Up", "Down", "Left" and "Right" (UDLR)
    def move_up(self):
            
        """move blank up"""

        #when blank key is on top edge of board already, do nothing
        if self.blank_row == 0:
            return None
        else:
            blank_index = self.blank_row * self.n + self.blank_col
            target = blank_index - self.n
            new_config = list(self.config)
            new_config[blank_index], new_config[target] = new_config[target], new_config[blank_index]
            return PuzzleState(tuple(new_config), self.n, parent=self, action="Up", cost=self.cost + 1)

    def move_down(self):
            
        """move blank down"""

        #when blank key is on bottom edge of board already, do nothing
        if self.blank_row == self.n - 1:
            return None
        else:
            blank_index = self.blank_row * self.n + self.blank_col
            target = blank_index + self.n
            new_config = list(self.config)
            new_config[blank_index], new_config[target] = new_config[target], new_config[blank_index]
            return PuzzleState(tuple(new_config), self.n, parent=self, action="Down", cost=self.cost + 1)

    def move_left(self):
            
        """action move blank left"""
        
        #when blank key is on left edge of board already, do nothing
        if self.blank_col == 0:
            return None
        else:
            blank_index = self.blank_row * self.n + self.blank_col
            target = blank_index - 1
            new_config = list(self.config)
            new_config[blank_index], new_config[target] = new_config[target], new_config[blank_index]
            return PuzzleState(tuple(new_config), self.n, parent=self, action="Left", cost=self.cost + 1)


    def move_right(self):
            
        """action move blank right"""

        #when blank key is on right edge of board already, do nothing
        if self.blank_col == self.n -1:
            return None
        else:
            blank_index = self.blank_row * self.n + self.blank_col
            target = blank_index + 1
            new_config = list(self.config)
            new_config[blank_index], new_config[target] = new_config[target], new_config[blank_index]
            return PuzzleState(tuple(new_config), self.n, parent=self, action="Right", cost=self.cost + 1)


    def expand(self):

        """Expands a node from:
        - a given (parent) node
        - the node is expanded in UDLR sequence
        - this is the function that generates the children from a given node (parent)
        """

        if len(self.children) == 0:
            up_child = self.move_up()

            if up_child is not None:
                self.children.append(up_child)

            down_child = self.move_down()

            if down_child is not None:
                self.children.append(down_child)

            left_child = self.move_left()

            if left_child is not None:
                self.children.append(left_child)
                
            right_child = self.move_right()
                
            if right_child is not None:
                self.children.append(right_child)
        
        return self.children


    def find_solution(self):
        
        """
        Get the path of the solution from:
        - a given node (e.g. the goal state)
        - work out its parent nodes recursively looking upwards in the search tree
        """

        solution = []
        solution.append(self.action)
        path = self
        while path.parent != None:
            path = path.parent
            solution.append(path.action)
        solution = solution[:-1]
        #reverse solution to display it from start to end
        solution.reverse()
        return solution

    def test_goal(self):
        """
        Tests a given state against the goal_state (0,1,2,3,4,5,6,7,8)
        """
        if self.config == tuple(self.goal_state):
            return True
        return False


# Main Search Algorithm Functions 
    
def bfs_search(start_node):
    """
    BFS search function takes as arguments:
    - a given puzzle state, e.g. start_node = ((1, 2, 5, 3, 4, 0, 6, 7, 8)

    Its outputs are: 
    - Path: the sequence of moves taken to reach the goal
    - CostOfPath: the number of moves taken to reach the goal
    - NodesExpanded: the number of nodes that have been expanded; 
      starts at '1' to count the expansion of the first node
    - SearchDepth: the depth within the search tree when the goal node is found;
      starts at '-1' so that level 0 is the start node,
      depth = 1 being the first expansion of the start node
    - MaxSearchDepth: the maximum depth of the search tree in the lifetime of the algorithm;
      starts at '-1' so that level 0 is the start node, 
      depth = 1 being the first expansion of the start node
    - Resources: the maximum RAM usage in the lifetime of the process as measured by the ru_maxrss attribute in the resource module, reported in megabytes

    """

    global Path, CostOfPath,  NodesExpanded, SearchDepth, MaxSearchDepth, Resources

    Path =[]
    CostOfPath = 0
    NodesExpanded = 1 
    SearchDepth = -1
    MaxSearchDepth = -1 
    Resources = 0

    if start_node.test_goal():
        return start_node.find_solution()

    q = Queue() #initialise queue as FIFO queue
    q.put(start_node) # put start node into FIFO queue
    explored=[] #start up explored

    while not(q.empty()):
        node=q.get()
        explored.append(node.config)
        children=node.expand()
        for child in children:
            if child.config not in explored:
                if child.test_goal():
                    #print('Success! Solution found using BFS: ')
                    #print(child)
                    Path = child.find_solution()
                    CostOfPath = len(Path)
                    NodesExpanded = len(explored) + len(children) + 1
                    SearchDepth = len(children)
                    MaxSearchDepth = calculate_total_cost(child) - CostOfPath
                    Resources = calculate_resources()
                    return child.find_solution()
                q.put(child)
    return


def dfs_search(start_node):
    
    """DFS search function takes as arguments:
    - a given puzzle state, e.g. start_node = ((1, 2, 5, 3, 4, 0, 6, 7, 8)

    Its outputs are: 
    - Path: the sequence of moves taken to reach the goal
    - CostOfPath: the number of moves taken to reach the goal
    - NodesExpanded: the number of nodes that have been expanded; 
      starts at '1' to count the expansion of the first node
    - SearchDepth: the depth within the search tree when the goal node is found;
      starts at '-1' so that level 0 is the start node,
      depth = 1 being the first expansion of the start node
    - MaxSearchDepth: the maximum depth of the search tree in the lifetime of the algorithm;
      starts at '-1' so that level 0 is the start node, 
      depth = 1 being the first expansion of the start node
    - Resources: the maximum RAM usage in the lifetime of the process as measured by the ru_maxrss attribute in the resource module, reported in megabytes
    """

    global MaxFrontier, MaxSearchDeph, Path, CostOfPath,  NodesExpanded, SearchDepth, MaxSearchDepth, Resources

    MaxFrontier = 0
    MaxSearchDeep = 0
    Path =[]
    CostOfPath = 0
    NodesExpanded = 1
    SearchDepth = -1 
    MaxSearchDepth = -1 
    Resources = 0

    explored = set()
    frontier = list([start_node])

    while frontier: 
        node = frontier.pop()
        explored.add(node.config)
        if node.test_goal():
            print('Success! Solution found using DFS: ')
            print(node)
            Path = find_solution_dfs(node, start_node)
            CostOfPath = len(Path)
            NodesExpanded = len(explored)
            SearchDepth = len(children)
            MaxSearchDepth = calculate_total_cost(node) - CostOfPath
            Resources = calculate_resources()
            return node
        
        children = node.expand()
        #reverse children to explore deepest node first
        children = children[::-1]
        for child in children:
            if child.config not in explored:
                frontier.append(child)
                explored.add(child.config)
        
    return 


def A_star_search(start_node):

    """A * search function takes as arguments:
    - a given puzzle state, e.g. start_node = ((1, 2, 5, 3, 4, 0, 6, 7, 8)

    Its outputs are: 
    - Path: the sequence of moves taken to reach the goal
    - CostOfPath: the number of moves taken to reach the goal
    - NodesExpanded: the number of nodes that have been expanded; 
      starts at '1' to count the expansion of the first node
    - SearchDepth: the depth within the search tree when the goal node is found;
      starts at '-1' so that level 0 is the start node,
      depth = 1 being the first expansion of the start node
    - MaxSearchDepth: the maximum depth of the search tree in the lifetime of the algorithm;
      starts at '-1' so that level 0 is the start node, 
      depth = 1 being the first expansion of the start node
    - Resources: the maximum RAM usage in the lifetime of the process as measured by the ru_maxrss attribute in the resource module, reported in megabytes
    
    """

    global Path, CostOfPath,  NodesExpanded, SearchDepth, MaxSearchDepth, Resources
    Path =[]
    CostOfPath = 0
    NodesExpanded = 1
    SearchDepth = -1 
    MaxSearchDepth = -1 
    Resources = 0

    if start_node.test_goal():
        #print("Solution found using A* search!")
        return start_node.find_solution()
    
    # initialize explored
    explored=[] 

    # initialise queue as Priority Queue to prioritise calling of elements based on value
    # note that PriorityQueue stores data as follows: 
        #[0]: evaluation value
        #[1]: count
        #[2]: puzzle configuration
    q = PriorityQueue()

    # next, we put start node into PriorityQueue alongside its cost
    # and start count as zero
    count = 0
    q.put((calculate_total_cost(start_node),count,start_node)) 

    while not(q.empty()):
        node = q.get()
        # get the actual node configuration, stored in item(2) of the PriorityQueue class
        node = node[2]
        explored.append(node.config)
        children=node.expand()
        for child in children:
            if child.config not in explored:
                if child.test_goal():
                    print('Success! Solution found using A*: ')
                    print(child)
                    Path = child.find_solution()
                    CostOfPath = len(Path)
                    NodesExpanded = len(explored) + len(children) + 1
                    SearchDepth = len(children)
                    MaxSearchDepth = calculate_total_cost(child) - CostOfPath
                    Resources = calculate_resources()
                    return child.find_solution()
                count += 1
                q.put((calculate_total_cost(child),count,child))
    return


# Helper Functions (# TO DO: should probably sit in separate helpers.py file)

def writeOutput(path, cost_of_path, nodes_expanded, search_depth, max_search_depth, resources, time_count, print_results = None):

    """writes the file output.txt with the following parameters:

        - path_to_goal: the sequence of moves taken to reach the goal
        - cost_of_path: the number of moves taken to reach the goal
        - nodes_expanded: the number of nodes that have been expanded
        - search_depth: the depth within the search tree when the goal node is found
        - max_search_depth:  the maximum depth of the search tree in the lifetime of 
            the algorithm
        - running_time: the total running time of the search instance, reported 
            in seconds
        - max_ram_usage: the maximum RAM usage in the lifetime of the process as 
            measured by the ru_maxrss attribute in the resource module, 
            reported in megabytes

        Note: for the output.txt file to be saved in the directory where the main.py file is, 
        ensure you are running the code on the correct virtual environment (e.g. AI-SearchAlgorithm-A1-8puzzle)

    """
    #Print results
    #if print_results is not None:
    #    print("path_to_goal: ",str(path)) # the sequence of moves taken to reach the goal
    #    print("cost_of_path: ",str(cost_of_path))  # the number of moves taken to reach the goal
    #    print("nodes_expanded: ",str(nodes_expanded)) # the number of nodes that have been expanded
    #    print("search_depth: ",str(search_depth)) # the depth within the search tree when the goal node is found
    #    print("max_search_depth: ",str(max_search_depth)) # the maximum depth of the search tree in the lifetime of the algorithm
    #    print("running_time: ",format(time_count, '.8f')) # the total running time of the search instance in seconds
    #    print("max_ram_usage: ",format(resources, '.8f'), " megabytes")

    #Generate outputs.txt document
    current_path = os.getcwd() #.split('\\')[1:-2]
    filename = 'outputs' + '.txt'
    filepath = os.path.join(current_path,filename) # add * if joining list items
    file = open(filepath, 'w')
    file.write("path_to_goal: " + str(path) + "\n")
    file.write("cost_of_path: " + str(cost_of_path) + "\n")
    file.write("nodes_expanded: " + str(nodes_expanded) + "\n")
    file.write("search_depth: " + str(search_depth) + "\n")
    file.write("max_search_depth: " + str(max_search_depth) + "\n")
    file.write("running_time: " + format(time_count, '.8f') + "\n")
    file.write("max_ram_usage: " + format(resources, '.8f') + " megabytes" + "\n")
    file.close()


def calculate_resources():
    """
    calculates memory resources used
    Note: RSS results are provided transformed to megabytes from bytes
    """
    if sys.platform == "win32":
        import psutil 
        RSS_Mb = float (psutil.Process().memory_info().rss) / 10**6  
        return RSS_Mb
    else:
        # Note: if you execute Python from cygwin,
        # the sys.platform is "cygwin"
        # the grading system's sys.platform is "linux2"
        import resource
        #print("resource", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def calculate_total_cost(config, heuristics = None):

        """
        Calculate the total estimated cost of a state from:
        - config: the state to calculate the cost of
        - heuristics: whether to consider heuristics or not, default is 'None';
          if heuristics are used (heuristics = True), then the cost will include 
          the manhattan distance calculation.
          Use heuristics = True when calling A* search function

        """

        if heuristics == True:
            # return the sum of h(n) + g(n)
            return abs(calculate_manhattan_dist(config) + config.cost)  # h(n) + g(n)
        else:
            # return only g(n) if heuristics are not used
            return config.cost 


def calculate_manhattan_dist(config):

    """calculate the manhattan distance from:
    - a given puzzle configuration for the A* search function
    
    """
    hn = 0 # start with the heuristics function value being 0

    for tile in range(1, config.n ** 2):
        # evalute the distance from each tile in the current state to the goal state
        # this is done by getting the index position of each tile in either state
        dist = abs(config.index(tile) - config.goal_state.index(tile)) 

        # estimate the number of movements from 'UDLR' which the tile needs to 
        # do to 'travel the distance' to its position in the goal_state
        i = dist // config.n # integer division by 3 because 3 is max number of movements
        j = int(dist % config.n) # get the remainder of the division
        hn = hn + i + j # add i and j to the heuristics function value over the loop

    return hn


def find_solution_dfs(node, start_node):
    solution = []
    solution.append(node.action)
    path = node
    while path.parent != start_node:
        path = path.parent
        solution.append(path.action)
    solution = solution[::-1]
    return solution


# Main Function that reads in Input and Runs corresponding Algorithm
# TO DO: should probably sit in separate main.py file

def main():

    """ calling the function is as follows:

    Test Case #1
    python3 driver.py bfs 3,1,2,0,4,5,6,7,8
    python3 driver.py dfs 3,1,2,0,4,5,6,7,8
    python3 driver.py ast 3,1,2,0,4,5,6,7,8

    Test Case #2
    python3 driver.py bfs 1,2,5,3,4,0,6,7,8
    python3 driver.py dfs 1,2,5,3,4,0,6,7,8
    python3 driver.py ast 1,2,5,3,4,0,6,7,8

    Test Case #3
    python3 driver.py bfs 6,1,8,4,0,2,7,3,5

    """

    #take initial configuration from first argument in calling of function
    sm = sys.argv[1].lower() 
    
    #take initial configuration from second argument in calling of function
    begin_config = sys.argv[2].split(",") 
    begin_config = tuple(map(int, begin_config))
    #calculate size of puzzle from length of tuple
    size = int(math.sqrt(len(begin_config))) 

    #call the class PuzzleState
    hard_config = PuzzleState(begin_config, size) 
    #print(hard_config)

    if sm == "bfs":
        t0 = time.time()
        bfs_results = bfs_search(hard_config)
        time_count = time.time() - t0
        writeOutput(Path, CostOfPath,  NodesExpanded, SearchDepth, MaxSearchDepth, Resources, time_count, "Y")
      
    elif sm == "dfs":
        t0 = time.time()
        dfs_results = dfs_search(hard_config)
        time_count = time.time() - t0
        writeOutput(Path, CostOfPath,  NodesExpanded, SearchDepth, MaxSearchDepth, Resources, time_count, "Y")

    elif sm == "ast":
        t0 = time.time()
        A_results = A_star_search(hard_config)
        time_count = time.time() - t0
        writeOutput(Path, CostOfPath,  NodesExpanded, SearchDepth, MaxSearchDepth, Resources, time_count, "Y")
        
    else:
        print("Enter valid command arguments !")

if __name__ == '__main__':

    main()
