# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 13:11:59 2022

@author: emmij
"""

import logic.util as util

class Coordinate():
    def __init__(self, x, y):
        self.x = int(x)
        self.y = int(y)
    
    def centre(self, c):
        return Coordinate((self.x + c.x) // 2, (self.y + c.y) // 2)
    
    def update(self, x, y):
        # propogate to super 1
        self.x = int(x)
        self.y = int(y)
    
    def __str__(self):
        return '(' + str(self.x) + ', ' + str(self.y) + ')'

class Box():
    def __init__(self, loc):
        self.tl = Coordinate(loc[0], loc[1])
        self.br = Coordinate(loc[2], loc[3])
        self.cc = self.tl.centre(self.br)
    
    def update(self, loc):
        # propogate to super 2
        self.tl.update(loc[0], loc[1])
        self.br.update(loc[2], loc[3])
        self.cc = self.tl.centre(self.br)
        

class Person():
    def __init__(self, pid, loc, grp = -1):
        self.pid = pid
        self.grp = grp
        self.box = Box(loc)
    
    def update(self, loc):
        # propogate to super 3
        self.box.update(loc)
    
    def getCentre(self):
        centre = self.box.cc
        return (centre.x, centre.y)
        

class Group():
    def __init__(self):
        self.grpid = util.groupCounter().__next__()
        self.members = []
        self.passive = []