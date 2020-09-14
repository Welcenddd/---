from math import sqrt
from collections import namedtuple
from random import random

#参考https://github.com/fengdu78/lihang-code/blob/master/
class kd_node(object):
    # 用数据结构保存kd树中每个结点所包含的信息(保存实例)
    def __init__(self,dom_elt,split,left,right):
        self.dom_elt = dom_elt #
        self.split = split #切分坐标轴
        self.left = left #左子树
        self.right = right #右子树

class kd_tree(object):
    def __init__(self,data):
        k = len(data[0]) #特征个数

        def create_node(split,dataset):
            if not dataset:
                return None
            dataset.sort(key=lambda  x: x[split])
            split_pos = len(dataset) // 2 #求len(dataset)/2的整除数
            median = dataset[split_pos] #切分点
            split_next = (split+1) % k
            return kd_node(median,split,create_node(split_next,dataset[:split_pos]),create_node(split_next,dataset[split_pos+1:])) #递归创建kd树
        self.root = create_node(0,data)

def pre_order(root):
    print(root.dom_elt)
    if root.left:
        pre_order(root.left)
    if root.right:
        pre_order(root.right)

result = namedtuple('Result_tuple', 'nearest_point nearest_dist nodes_visited')

def find_nearest(tree,point):
    k = len(point) # 目标点的特征维数
    def travel(kd_node,target,max_dist):
        if kd_node is None:
            return result([0]*k, float('inf'),0) #如果结点为空，返回无意义的tuple
        nodes_visited = 1 #否则，访问结点数+1

        s = kd_node.split # 该结点的切分维度
        pivot = kd_node.dom_elt #保存的实例结点的坐标

        if target[s] <= pivot[s]:
            nearer_node = kd_node.left
            further_node = kd_node.right
        else:
            nearer_node = kd_node.right
            further_node = kd_node.left

        temp1 = travel(nearer_node,target,max_dist)
        nearest = temp1.nearest_point #递归结束，找到包含目标点的叶结点，以此叶结点为“当前最近点”
        dist = temp1.nearest_dist #将相应的距离设为超球体的半径

        nodes_visited += temp1.nodes_visited #遍历结点数+1

        if dist < max_dist:
            max_dist = dist

        temp_dist = abs(pivot[s] - target[s]) #求分割点与目标点的距离
        if max_dist < temp_dist: # max_dist<temp_dist说明实例结点(分割点)不与以目标点为球心、以目标点和“当前最近点”间的距离为半径的超球体相交，直接返回
            return result(nearest,dist,nodes_visited)
        temp_dist = sqrt(sum((p1-p2)**2 for p1,p2 in zip(pivot,target))) #如果相交，计算结点和目标点的欧氏距离

        if temp_dist < dist: # 如果结点与目标点的欧氏距离小于“当前最近点”与目标点的距离，将该结点设置为新的“当前最近点”
            nearest = pivot
            dist = temp_dist
            max_dist = dist

        #检查该子结点的父结点的另一子结点对应的区域是否有更近的点
        temp2 = travel(further_node,target,max_dist)

        nodes_visited += temp2.nodes_visited
        if temp2.nearest_dist < dist: #如果有就更新最近点
            nearest = temp2.nearest_point
            dist = temp2.nearest_dist

        return result(nearest,dist,nodes_visited)
    return travel(tree.root,point,float('inf')) #从根结点开始递归

def random_point(k):
    return [random() for _ in range(k)]

def random_points(k,n):
    return [random_point(k) for _ in range(n)]


if __name__ == '__main__':
    data = [[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]]
    kd = kd_tree(data)
    pre_order(kd.root)
    ret = find_nearest(kd, [3,4.5])
    print(ret)

    N = 400000
    k = 3
    kd2 = kd_tree(random_points(k,N))
    ret2 = find_nearest(kd2,[0.1,0.5,0.8])
    print(ret2)
