from audioop import reverse
import sys
import numpy as np

sys.path.extend(['../'])
from graph import tools_hdgcn
# import tools_hdgcn

num_node = 22

class Graph:
    def __init__(self,labeling_mode='spatial'):
        self.num_node = num_node
        if labeling_mode != 'spatial':
            self.A = tools_hdgcn.get_m(num_node)
        else: self.A = self.get_adjacency_matrix(labeling_mode)
        

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools_hdgcn.get_hierarchical_graph(num_node, tools_hdgcn.get_edgeset()) # L, 3, 25, 25
        else:
            raise ValueError()
        return A



if __name__ == '__main__':
    # import graph.tools_hdgcn as tools_ctr
    g = Graph().A
    g.shape