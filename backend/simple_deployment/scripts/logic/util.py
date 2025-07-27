# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 13:17:33 2022

@author: emmij
"""

import numpy as np
from sklearn.cluster import DBSCAN
def cluster(coordinate_list):
    X = np.array(coordinate_list)
    clustering = DBSCAN(eps=100, min_samples=2).fit(X)
    return clustering.labels_
def map_label(people, clusters):
    # [(person_id, cluster_label), ...]
    return list(zip(people, clusters))

def group_counter():
    x = 0
    while True: 
        yield x
        x += 1

def left_most(coordinates):
    # pseudo bottom left
    return sorted(coordinates, key=lambda c: c.x + c.y)[0]
    '''
    maxi = float("-inf")
    for coordinate in coordinates:
        s = coordinate.x + coordinate.y
        if s > maxi:
            maxi = s
            
    maxiCoordinate = []
    for coordinate in coordinates:
        s = coordinate.x + coordinate.y
        if s == maxi:
            maxiCoordinate.append(coordinate)
    
    return sorted(maxiCoordinate, key=lambda coordinate: coordinate.y)[-1]
    '''

def right_most(coordinates):
    # pseudo top right
    return sorted(coordinates, key=lambda c: c.x + c.y)[-1]
    '''
    mini = float("inf")
    for coordinate in coordinates:
        s = coordinate.x + coordinate.y
        if s < mini:
            mini = s
            
    miniCoordinate = []
    for coordinate in coordinates:
        s = coordinate.x + coordinate.y
        if s == mini:
            miniCoordinate.append(coordinate)
    
    return sorted(miniCoordinate, key=lambda coordinate: coordinate.y)[0]
    '''