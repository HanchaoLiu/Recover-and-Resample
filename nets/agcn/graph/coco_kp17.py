import os,sys

# sys.path.extend(['../'])

# print(__file__)
sys.path.append(os.path.join(os.path.dirname(__file__),".."))
# print(sys.path)

from graph import tools


num_node = 17
self_link = [(i, i) for i in range(num_node)]
# select left shoulder 6 as the base joint, although it is not symmetric
inward_ori_index = [
    (1,0), (3,1), (4,2), (2,0),(0,6), (0,5), (5,6),
    (10,8), (8,6), (9,7), (7,5),
    (16,14), (14,12), (12, 6), (11,12),
    (15,13), (13,11), (11, 5)
]
inward = [(i, j) for (i, j) in inward_ori_index]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward


class Graph:
    def __init__(self, labeling_mode='spatial'):
        self.A = self.get_adjacency_matrix(labeling_mode)
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(num_node, self_link, inward, outward)
        else:
            raise ValueError()
        return A


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import os

    # os.environ['DISPLAY'] = 'localhost:11.0'
    A = Graph('spatial').get_adjacency_matrix()
    for i in A:
        plt.imshow(i, cmap='gray')
        plt.show()
    print(A)
