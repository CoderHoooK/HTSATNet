from audioop import reverse
import sys
import numpy as np

sys.path.extend(['../'])
from graph import tools_hdgcn
# import tools_hdgcn



class Graph:
    def __init__(self,labeling_mode='spatial',num_node = 22,L = 6):
        self.num_node = num_node
        self.A = self.get_adjacency_matrix(L,labeling_mode)
        

    def get_adjacency_matrix(self, L,labeling_mode=None,):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = np.ones((L,3,self.num_node,self.num_node)) # L, 3, 25, 25
        else:
            raise ValueError()
        return A


if __name__ == '__main__':
    # import graph.tools_hdgcn as tools_ctr
    g = Graph().A
    g.shape