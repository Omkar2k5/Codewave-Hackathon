# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 15:00:15 2022

@author: emmij
"""
import re
from logic.model import Person
from logic.util import cluster, map_label, left_most, right_most

# global
frames = {}
cv_input = {}

def generate_group(frame_id, clusters):
    # IMPORTANT: Lose of object data
    # Tailor to openCV standard
    
    # Single Frame Activation Code
    global cv_input
    cv_input = {}
    for cluster_id, people in clusters.items():
        tl_coordinates = [person.box.tl for person in people]
        br_coordinates = [person.box.br for person in people]
        lm = left_most(tl_coordinates)
        rm = right_most(br_coordinates)
        cv_input[cluster_id] = [lm.x, lm.y, rm.x, rm.y]
    
    '''
    cv_input[frame_id] = {}
    for cluster_id, people in clusters.items():
        tl_coordinates = [person.box.tl for person in people]
        br_coordinates = [person.box.br for person in people]
        lm = left_most(tl_coordinates)
        rm = right_most(br_coordinates)
        cv_input[frame_id][cluster_id] = [lm.x, lm.y, rm.x, rm.y]
    '''
        

def analyze_frame(frame_id, people):
    coordinate_list = [list(person.getCentre()) for person in people]
    cluster_map = map_label(people, cluster(coordinate_list))
    clusters = {}
    for point in cluster_map:
        person, cluster_id = point[0], point[1]
        if cluster_id != -1:
            if cluster_id not in clusters:
                clusters[cluster_id] = [person]
            else:
                clusters[cluster_id].append(person)
    frames[frame_id] = generate_group(frame_id, clusters)

def analyze_file(frame_data = None):
    global x
    # file = open('out.txt', 'r')
    file = frame_data.split('\n')
    people = []
    frame_id = 0
    print('Frame data', frame_data)
    for line in file:
        if re.search('\AFrame', line):
            # Set new frame_id
            frame_id = int(line[9:].strip())
            
        elif re.search('\ATracker', line):
            # YOLO Person as Tracker: id
            tracker_id = int(line[11: line.find(',')].strip())
            # Find content in first parenthesis
            coordinates = re.findall(r'\((.*?)\)', line)[1].split(', ')
            # Generate Person 
            p = Person(tracker_id, coordinates)
            people.append(p)
                
        elif re.search('\AEOF', line) or re.search('\AFPS', line):
            # Frame complete
            if people == []:
                continue
            analyze_frame(frame_id, people)
    return cv_input

#analyze_file()
#print(cv_input)
#from draw import drawBox