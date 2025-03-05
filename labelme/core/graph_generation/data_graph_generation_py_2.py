"""
CODE to generate artery graphs.
"""
import json
import io
import shutil
import time
import networkx as nx
import sys
import os

import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import argparse

from tqdm import tqdm
from skimage import measure
from glob import glob

from multiprocessing.pool import ThreadPool
from labelme.core.utils.module import Node, VesselSegment
from labelme.core.graph_generation import visualization as visualizer
from labelme.core.graph_generation import networkx_graph_visualizer as nxg
from PIL import Image


from labelme.core.graph_generation import constants

from labelme.core.graph_generation.data_processing_2 import remove_isolated_segment, \
    skeleton_by_matlab, calculate_end_and_joint_points_by_matlab, diameter_map, get_spacing, \
    skeleton_by_python, calculate_end_and_joint_points_by_python, diameter_map_python, vessel_segment_point_python
from labelme.core.graph_generation.data_processing_2 import __generate_vessel_binary_map, __interpolate_centerline_points, \
    __calculate_perpendicular_line_slope, __find_vessel_pixel, __add_adjacent_points


##### STEP 3 GENERATE GRAPH ACCORDING TO GENERATED NODES #################
def generate_images(semantic_images_path, image_size, colormap, search_priority):
    semantic_image_paths = glob(os.path.join(semantic_images_path, "*.png"))
    binary_image   = np.zeros([image_size, image_size], dtype=np.uint8)
    semantic_image = np.zeros([image_size, image_size, 3], dtype=np.uint8)

    for pattern in search_priority:
        for semantic_image_path in semantic_image_paths:
            semantic_label = semantic_image_path[semantic_image_path.rfind("_") + 1: semantic_image_path.rfind(".png")]
            if semantic_label.rfind(pattern) !=-1:
                semantic_image_segment = cv2.imread(semantic_image_path, cv2.IMREAD_GRAYSCALE)
                if semantic_image_segment.shape[0] != image_size:
                    semantic_image_segment = cv2.resize(semantic_image_segment, (image_size, image_size), cv2.INTER_NEAREST)

                semantic_image[np.where(semantic_image_segment > 0)[0], np.where(semantic_image_segment > 0)[1], :] = colormap[semantic_label]
                binary_image[np.where(semantic_image_segment > 0)[0], np.where(semantic_image_segment > 0)[1]] = 255

    return semantic_image, binary_image


def graph_generation_by_connectivity(patient_name, nodes: list, skeleton_img,
                                     semantic_image, semantic_mapping, eng=None):
    """
    creating VesselSegment objects according to the detected nodes
    :param nodes:
    :param skeleton_img:
    :param eng:
    :return:
    """
    g = nx.Graph()
    for idx, node in enumerate(nodes):
        g.add_node(idx, data=node)
        for i in range(node.x-1, node.x+2):
            for j in range(node.y-1, node.y+2):
                if i < skeleton_img.shape[0] and j < skeleton_img.shape[1] and i >=0 and j>= 0:
                   skeleton_img[i, j] = 0
        #skeleton_img[node.x, node.y] = 0

    #labeling = measure.label(skeleton_img, neighbors=8, connectivity=1)
    labeling, n_segmnets = measure.label(skeleton_img, return_num=True)

    for idx, label in enumerate(np.unique(labeling)[1:]):
        vessel_segment_centerline = np.zeros_like(labeling)
        vessel_segment_centerline[labeling == label] = 1

        # find end points by matlab
        if not os.path.isdir(os.path.join("temp", patient_name)):
            os.makedirs(os.path.join("temp", patient_name))
        cv2.imwrite(os.path.join("temp", patient_name, "temp.png"), vessel_segment_centerline)
        ret = eng.vessel_segment_point(os.path.join("temp", patient_name, "temp.png"), nargout=4, stdout=io.StringIO())
        rj, cj, re, ce = np.array(ret[0], dtype=np.int32).flatten() - 1, \
                         np.array(ret[1], dtype=np.int32).flatten() - 1, \
                         np.array(ret[2], dtype=np.int32).flatten() - 1, \
                         np.array(ret[3], dtype=np.int32).flatten() - 1

        if re.shape[0] == 2:
            edge_points = []
            for i in range(2):
                near_dist = np.inf
                near_node = None
                for idx, node in enumerate(nodes):
                    end_point_x, end_point_y = re[i], ce[i]
                    dist = np.sqrt((end_point_x - node.x) ** 2 + (end_point_y - node.y) ** 2)
                    if dist < near_dist:
                        near_dist = dist
                        near_node = node
                edge_points.append(near_node)

            fig, ax = plt.subplots()
            plt.axis('off')
            fig.set_size_inches(skeleton_img.shape[0]/100, skeleton_img.shape[1]/100)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0.0, wspace=0.0)
            plt.margins(0, 0)
            plt.imshow(vessel_segment_centerline, cmap="gray")
            plt.scatter(edge_points[0].y, edge_points[0].x, marker='+')
            plt.scatter(edge_points[1].y, edge_points[1].x, marker='+')
            plt.savefig(f"temp/{patient_name}/edge_{idx}.png")
            plt.close()

            vessel_object = VesselSegment(edge_points[0], edge_points[1], vessel_segment_centerline)
            vessel_object.vessel_class = assign_class_on_vessel_segment(vessel_object, semantic_image, semantic_mapping)

            if g.has_edge(nodes.index(edge_points[0]), nodes.index(edge_points[1])):
                existing_vessel_object = g.edges[nodes.index(edge_points[0]), nodes.index(edge_points[1])]['data']
                #if np.sum(existing_vessel_object.vessel_centerline) > np.sum()
                # TODO compare assigned class first, this needs to be modified.
                if constants.search_sequence.index(existing_vessel_object.vessel_class) < constants.search_sequence.index(vessel_object.vessel_class):
                    g.edges[nodes.index(edge_points[0]), nodes.index(edge_points[1])]['data'] = vessel_object
            else:
                # add a new edge and assign vessel object in data field
                g.add_node(nodes.index(edge_points[0]), data=edge_points[0])
                g.add_node(nodes.index(edge_points[1]), data=edge_points[1])
                g.add_edge(nodes.index(edge_points[0]), nodes.index(edge_points[1]), data=vessel_object)

    return g, skeleton_img


def graph_generation_by_connectivity_python(patient_name, nodes: list, skeleton_img, semantic_image, semantic_mapping):
    """
    creating VesselSegment objects according to the detected nodes
    :param nodes:
    :param skeleton_img:
    :param eng:
    :return:
    """
    g = nx.Graph()
    for idx, node in enumerate(nodes):
        g.add_node(idx, data=node)
        for i in range(node.x-1, node.x+2):
            for j in range(node.y-1, node.y+2):
                if i < skeleton_img.shape[0] and j < skeleton_img.shape[1] and i >=0 and j>= 0:
                   skeleton_img[i, j] = 0
        #skeleton_img[node.x, node.y] = 0

    #labeling = measure.label(skeleton_img, neighbors=8, connectivity=1)
    labeling, n_segmnets = measure.label(skeleton_img, return_num=True)

    for idx, label in enumerate(np.unique(labeling)[1:]):
        vessel_segment_centerline = np.zeros_like(labeling)
        vessel_segment_centerline[labeling == label] = 1

        # find end points by matlab
        if not os.path.isdir(os.path.join("temp", patient_name)):
            os.makedirs(os.path.join("temp", patient_name))
        cv2.imwrite(os.path.join("temp", patient_name, "temp.png"), vessel_segment_centerline)
        # ret = eng.vessel_segment_point(os.path.join("temp", patient_name, "temp.png"), nargout=4, stdout=io.StringIO())
        rj, cj, re, ce = vessel_segment_point_python(os.path.join("temp", patient_name, "temp.png"))

        if re.shape[0] == 2:
            edge_points = []
            for i in range(2):
                near_dist = np.inf
                near_node = None
                for idx, node in enumerate(nodes):
                    end_point_x, end_point_y = re[i], ce[i]
                    dist = np.sqrt((end_point_x - node.x) ** 2 + (end_point_y - node.y) ** 2)
                    if dist < near_dist:
                        near_dist = dist
                        near_node = node
                edge_points.append(near_node)

            fig, ax = plt.subplots()
            plt.axis('off')
            fig.set_size_inches(skeleton_img.shape[0]/100, skeleton_img.shape[1]/100)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0.0, wspace=0.0)
            plt.margins(0, 0)
            plt.imshow(vessel_segment_centerline, cmap="gray")
            plt.scatter(edge_points[0].y, edge_points[0].x, marker='+')
            plt.scatter(edge_points[1].y, edge_points[1].x, marker='+')
            plt.savefig(f"temp/{patient_name}/edge_{idx}.png")
            plt.close()

            vessel_object = VesselSegment(edge_points[0], edge_points[1], vessel_segment_centerline)
            vessel_object.vessel_class = assign_class_on_vessel_segment(vessel_object, semantic_image, semantic_mapping)

            if g.has_edge(nodes.index(edge_points[0]), nodes.index(edge_points[1])):
                existing_vessel_object = g.edges[nodes.index(edge_points[0]), nodes.index(edge_points[1])]['data']
                #if np.sum(existing_vessel_object.vessel_centerline) > np.sum()
                # TODO compare assigned class first, this needs to be modified.
                if constants.search_sequence.index(existing_vessel_object.vessel_class) < constants.search_sequence.index(vessel_object.vessel_class):
                    g.edges[nodes.index(edge_points[0]), nodes.index(edge_points[1])]['data'] = vessel_object
            else:
                # add a new edge and assign vessel object in data field
                g.add_node(nodes.index(edge_points[0]), data=edge_points[0])
                g.add_node(nodes.index(edge_points[1]), data=edge_points[1])
                g.add_edge(nodes.index(edge_points[0]), nodes.index(edge_points[1]), data=vessel_object)

    return g, skeleton_img

def reconnect(graph: nx.Graph, radius=5):
    while(not nx.is_connected(graph)):
        components = list(nx.connected_components(graph)) # [{n1,n2},{n3,n4}]...
        comp_dist = np.zeros([len(components), len(components)])
        comp_node_pairs = np.array([[(0, 0) for i in range(len(components))] for j in range(len(components))])
        for ci in range(len(components)):
            for cj in range(ci+1, len(components)):
                component_1, component_2 = list(components[ci]), list(components[cj])
                distance = np.zeros([len(graph.nodes), len(graph.nodes)])
                for i in component_1:
                    for j in component_2:
                        distance[i][j] = np.sqrt((graph.nodes[i]['data'].x - graph.nodes[j]['data'].x)**2 +
                                                 (graph.nodes[i]['data'].y - graph.nodes[j]['data'].y)**2)

                distance[distance==0] = np.inf
                node_idx_in_comp1 = np.where(distance == np.amin(distance))[0][0]
                node_idx_in_comp2 = np.where(distance == np.amin(distance))[1][0]
                #print(f"[x] reconnect, node {node_idx_in_comp1} in comp1 and node {node_idx_in_comp2} in comp2, dist = {distance[node_idx_in_comp1][node_idx_in_comp2]}")

                comp_dist[ci][cj] = distance[node_idx_in_comp1][node_idx_in_comp2]
                comp_node_pairs[ci][cj] = (node_idx_in_comp1, node_idx_in_comp2)

        comp_dist[comp_dist==0] = np.inf
        selected_comp_idx_1 = np.where(comp_dist == np.amin(comp_dist))[0][0]
        selected_comp_idx_2 = np.where(comp_dist == np.amin(comp_dist))[1][0]
        node_distance = comp_dist[selected_comp_idx_1][selected_comp_idx_2]
        comp_node_pair = comp_node_pairs[selected_comp_idx_1][selected_comp_idx_2]
        # connected these two components on node pair
        print(f"[x] reconnect, node {comp_node_pair[0]} <-> {comp_node_pair[1]} with distance of {node_distance}")
        if node_distance > radius:
            raise ValueError("[!] cannot reconnect two components")
        else:
            graph.add_edge(comp_node_pair[0], comp_node_pair[1])
    return graph

# STEP 4 ANN 6: filter out artery segments if:
# 1) the length is smaller than threshold
# 2) max diameter is smaller than threshold
def filter_vessel(graph: nx.Graph, dist_map, pixel_spacing, radius_threshold, centerline_length_threshold):
    print(f"[x] before filter out artery, # nodes = {len(graph.nodes)}, # edges = {len(graph.edges)}")
    for edge in list(graph.edges):
        if len(graph.edges[edge[0], edge[1]].keys()) > 0:
            vessel_obj = graph.edges[edge[0], edge[1]]['data']
            # print(np.sum(vessel_obj.vessel_centerline>0))
            delete = True
            if np.sum(vessel_obj.vessel_centerline) > centerline_length_threshold:
                segment_dist_map = vessel_obj.vessel_centerline * dist_map
                # assign distance map to each vessel object
                vessel_obj.vessel_centerline_dist = segment_dist_map
                # mean_dist = np.mean(segment_dist_map[segment_dist_map > 0])
                # min_dist = np.min(segment_dist_map[segment_dist_map > 0])
                max_dist = np.max(segment_dist_map[segment_dist_map > 0])

                if max_dist * pixel_spacing >= radius_threshold:
                    # keep this kind of vessel object and retain graph structure
                    delete = False

            if delete:
                # remove vessel_object
                if graph.degree[edge[0]] == 1:
                    # remove edge with end points
                    graph.remove_node(edge[0])
                elif graph.degree[edge[1]] == 1:
                    # remove node with end points: in implementation, remove node will automatically remove edge
                    graph.remove_node(edge[1])
                else:
                    # remove edge without end points
                    # vessel_segment_t = graph.edges[edge[0], edge[1]]['data']
                    if __is_connected_after_remove(graph, [edge[0], edge[1]]):
                        graph.remove_edge(edge[0], edge[1])
    print(f"[x] after filter out artery, # nodes = {len(graph.nodes)}, # edges = {len(graph.edges)}")
    return graph


def __is_connected_after_remove(graph: nx.Graph, node_index: list):
    g = graph.copy(as_view=False)
    if len(node_index) == 1:
        g.remove_node(node_index[0])
        return nx.is_connected(g)
    else:
        g.remove_edge(node_index[0], node_index[1])
        return nx.is_connected(g)

# 1. remove degree-2 nodes, and merge vessel objects and related graph objects
def merge_degree_two(graph: nx.Graph):
    def __merge_vessel_objs(v_obj1: VesselSegment, v_obj2: VesselSegment, merged_node):
        v_obj1.vessel_centerline += v_obj2.vessel_centerline
        v_obj1.vessel_centerline = np.clip(v_obj1.vessel_centerline, 0, 1)
        v_obj1.vessel_mask += v_obj2.vessel_mask
        v_obj1.vessel_mask = np.clip(v_obj1.vessel_mask, 0, 1)
        # merge nodes from two vessel segments
        if v_obj1.node1.x == merged_node.x and v_obj1.node1.y == merged_node.y:
            v_obj1.node1 = Node(v_obj1.node2.degree, v_obj1.node2.x, v_obj1.node2.y)
        else:
            v_obj1.node1 = Node(v_obj1.node1.degree, v_obj1.node1.x, v_obj1.node1.y)

        if v_obj2.node1.x == merged_node.x and v_obj2.node1.y == merged_node.y:
            v_obj1.node2 = Node(v_obj2.node2.degree, v_obj2.node2.x, v_obj2.node2.y)
        else:
            v_obj1.node2 = Node(v_obj2.node1.degree, v_obj2.node1.x, v_obj2.node1.y)

        return v_obj1

    nodes_to_be_merged = []
    n_nodes_to_be_merged = 0
    for node_idx in list(graph.nodes):
        if graph.degree[node_idx] == 2:
            # get edge
            nodes_to_be_merged.append(node_idx)

    print(f"[x] merge_degree_two, before remove degree-2 nodes, # nodes = {len(graph.nodes)}, # edges = {len(graph.edges)}")
    for node_idx in nodes_to_be_merged:
        if graph.degree[node_idx] == 2:
            connected_edges = list(graph.edges(node_idx))
            if len(connected_edges) == 2:
                adj_node_indexes = []
                for connected_edge in connected_edges:
                    if connected_edge[0] != node_idx:
                        adj_node_indexes.append(connected_edge[0])
                    else:
                        adj_node_indexes.append(connected_edge[1])

                if len(graph.edges[connected_edges[0][0], connected_edges[0][1]].keys()) > 0:
                    vessel_segment_0 = graph.edges[connected_edges[0][0], connected_edges[0][1]]['data']
                else:
                    vessel_segment_0 = None
                if len(graph.edges[connected_edges[1][0], connected_edges[1][1]].keys()) > 0:
                    vessel_segment_1 = graph.edges[connected_edges[1][0], connected_edges[1][1]]['data']
                else:
                    vessel_segment_1 = None

                if vessel_segment_0 is None and vessel_segment_1 is not None:
                    graph.remove_node(node_idx)
                    graph.add_edge(adj_node_indexes[0], adj_node_indexes[1], data=vessel_segment_1)
                elif vessel_segment_1 is None and vessel_segment_0 is not None:
                    graph.remove_node(node_idx)
                    graph.add_edge(adj_node_indexes[0], adj_node_indexes[1], data=vessel_segment_0)
                elif vessel_segment_0 is None and vessel_segment_1 is None:
                    graph.remove_node(node_idx)
                    graph.add_edge(adj_node_indexes[0], adj_node_indexes[1])
                else:
                    if vessel_segment_1.vessel_class == vessel_segment_0.vessel_class:
                        # merge vessel segments with the same classes
                        merged_vessel_segment = __merge_vessel_objs(vessel_segment_0, vessel_segment_1, graph.nodes[node_idx]['data'])
                        graph.remove_node(node_idx)
                        graph.add_edge(adj_node_indexes[0], adj_node_indexes[1], data=merged_vessel_segment)
                        n_nodes_to_be_merged += 1
    print(f"[x] merge_degree_two, after remove degree-2 nodes, # nodes = {len(graph.nodes)}, # edges = {len(graph.edges)}, {n_nodes_to_be_merged} nodes have been merged")

    return graph

# Find the arterial segment mapping relationship between centerline and arterial binary contours
def find_vessel_pixel_in_centerline(graph, binary_image):
    mapped_mask = np.zeros_like(binary_image)

    for edge in list(graph.edges):
        if len(graph.edges[edge[0], edge[1]].keys()) > 0:
            vessel_obj = graph.edges[edge[0], edge[1]]['data']
            vessel_points = __interpolate_centerline_points(vessel_obj, 'slinear', interp_num=5)
            vessel_mask = np.zeros_like(binary_image)
            for vessel_point in vessel_points:
                slope = __calculate_perpendicular_line_slope(vessel_point, vessel_points)
                vessel_pixel_coordinates = __find_vessel_pixel(vessel_point, slope, binary_image,
                                                               offset=10, interpolation_interval=0.5, adj_size=-1)
                vessel_mask = __generate_vessel_binary_map(binary_image, vessel_pixel_coordinates, vessel_mask)

            graph.edges[edge[0], edge[1]]['data'].vessel_mask = vessel_mask
            mapped_mask += vessel_mask

    mapped_mask = np.clip(mapped_mask, 0, 1)
    print(f"# pixels in binary image = {np.sum(binary_image > 0)}, # pixels in mapped image = {np.sum(mapped_mask > 0)}, ratio = {np.sum(mapped_mask > 0) / np.sum(binary_image > 0)}")

    return mapped_mask


def find_vessel_pixel_in_centerline_dist(graph, binary_image, dist_map):
    def _find_nearest_greater_than_zero(coord, array):
        nonzero_coords = np.argwhere(array > 0)  # Get coordinates of elements greater than 0
        if len(nonzero_coords) == 0:
            return None  # No elements greater than 0

        distances = np.linalg.norm(nonzero_coords - coord, axis=1)  # Compute distances
        nearest_index = np.argmin(distances)  # Find index of nearest coordinate

        return array[nonzero_coords[nearest_index][0], nonzero_coords[nearest_index][1]]

    mapped_mask = np.zeros_like(binary_image)

    for edge in list(graph.edges):
        if len(graph.edges[edge[0], edge[1]].keys()) > 0:
            vessel_obj = graph.edges[edge[0], edge[1]]['data']
            vessel_points = __interpolate_centerline_points(vessel_obj, 'slinear', interp_num=5)
            vessel_mask = np.zeros_like(binary_image)
            for vessel_point in vessel_points:
                slope = __calculate_perpendicular_line_slope(vessel_point, vessel_points)
                offset = _find_nearest_greater_than_zero(vessel_point, dist_map)
                vessel_pixel_coordinates = __find_vessel_pixel(vessel_point, slope, binary_image,
                                                               offset=offset, interpolation_interval=0.2, adj_size=-1)
                vessel_mask = __generate_vessel_binary_map(binary_image, vessel_pixel_coordinates, vessel_mask)

            graph.edges[edge[0], edge[1]]['data'].vessel_mask = vessel_mask
            mapped_mask += vessel_mask

    mapped_mask = np.clip(mapped_mask, 0, 1)
    print(
        f"# pixels in binary image = {np.sum(binary_image > 0)}, # pixels in mapped image = {np.sum(mapped_mask > 0)}, ratio = {np.sum(mapped_mask > 0) / np.sum(binary_image > 0)}")

    return mapped_mask


def __get_class(coord, semantic_image, semantic_mapping):
    for key in semantic_mapping.keys():
        # color to label
        if semantic_image[coord[0], coord[1], 0] == semantic_mapping[key][0]:
            if semantic_image[coord[0], coord[1], 1] == semantic_mapping[key][1]:
                if semantic_image[coord[0], coord[1], 2] == semantic_mapping[key][2]:
                    return key
    return constants.VESSEL_OTHER_COLOR_KEY


def assign_class_on_vessel_segment(vessel_obj: VesselSegment, semantic_image: np.array, semantic_mapping: dict):
    num_pixels = np.sum(vessel_obj.vessel_centerline > 0)
    count_dict = {k: 0 for k in semantic_mapping.keys()}

    for i in range(num_pixels):
        try:
            vessel_pixel_coord = (
            np.where(vessel_obj.vessel_centerline == 1)[0][i], np.where(vessel_obj.vessel_centerline == 1)[1][i])
            clazz = __get_class(vessel_pixel_coord, semantic_image, semantic_mapping)
            count_dict[clazz] = count_dict[clazz] + 1
        except:
            vessel_obj.vessel_centerline[vessel_pixel_coord[0], vessel_pixel_coord[1]] = 0
            print("coord = {}, color = {}".format(vessel_pixel_coord, semantic_image[vessel_pixel_coord[0], vessel_pixel_coord[1]]))
    # print(count_dict)
    count_dict = sorted(count_dict.items(), key=lambda x: x[1])[::-1]
    return count_dict[0][0]


def assign_class_on_graph(graph, semantic_image, semantic_mapping: dict):
    for edge in list(graph.edges):
        vessel_obj = graph.edges[edge[0], edge[1]]['data']
        num_pixels = np.sum(vessel_obj.vessel_centerline>0)
        count_dict = {k: 0 for k in semantic_mapping.keys()}

        for i in range(num_pixels):
            try:
                vessel_pixel_coord = (np.where(vessel_obj.vessel_centerline == 1)[0][i], np.where(vessel_obj.vessel_centerline == 1)[1][i])
                clazz = __get_class(vessel_pixel_coord, semantic_image, semantic_mapping)
                count_dict[clazz] = count_dict[clazz]+1
            except:
                vessel_obj.vessel_centerline[vessel_pixel_coord[0], vessel_pixel_coord[1]] = 0
                print("coord = {}, color = {}".format(vessel_pixel_coord, semantic_image[vessel_pixel_coord[0], vessel_pixel_coord[1]]))
        #print(count_dict)
        count_dict = sorted(count_dict.items(), key=lambda x: x[1])[::-1]
        #clazz = count_dict[0][0]
        # if count_dict[0][0] == constants.VESSEL_OTHER_COLOR_KEY:
        #     graph.remove_edge(edge[0], edge[1])
        # else:
        graph.edges[edge[0], edge[1]]['data'].vessel_class = count_dict[0][0]

    return graph


def __find_all_cycles(G, source=None, cycle_length_limit=None):
    """forked from networkx dfs_edges function. Assumes nodes are integers, or at least
    types which work with min() and > ."""
    if source is None:
        # produce edges for all components
        nodes = [list(i)[0] for i in nx.connected_components(G)]
    else:
        # produce edges for components with source
        nodes = [source]
    # extra variables for cycle detection:
    cycle_stack = []
    output_cycles = set()

    def get_hashable_cycle(cycle):
        """cycle as a tuple in a deterministic order."""
        m = min(cycle)
        mi = cycle.index(m)
        mi_plus_1 = mi + 1 if mi < len(cycle) - 1 else 0
        if cycle[mi - 1] > cycle[mi_plus_1]:
            result = cycle[mi:] + cycle[:mi]
        else:
            result = list(reversed(cycle[:mi_plus_1])) + list(reversed(cycle[mi_plus_1:]))
        return tuple(result)

    for start in nodes:
        if start in cycle_stack:
            continue
        cycle_stack.append(start)

        stack = [(start, iter(G[start]))]
        while stack:
            parent, children = stack[-1]
            try:
                child = next(children)

                if child not in cycle_stack:
                    cycle_stack.append(child)
                    stack.append((child, iter(G[child])))
                else:
                    i = cycle_stack.index(child)
                    if i < len(cycle_stack) - 2:
                        output_cycles.add(get_hashable_cycle(cycle_stack[i:]))

            except StopIteration:
                stack.pop()
                cycle_stack.pop()

    return [list(i) for i in output_cycles]


##### STEP 7 after calculate_end_and_joint_points_by_matlab ##############
def merge_splitting_nodes(graph: nx.Graph, distance_threshold):
    # Node contraction identifies the two nodes as a single node incident to any edge
    # that was incident to the original two nodes
    def __has_joint_nodes_adj():
        # to check if there are two joint nodes located within a circle with a radius of 'distance_threshold'
        for node_i_idx in graph.nodes:
            for node_j_idx in graph.nodes:
                if node_i_idx != node_j_idx:
                    if graph.degree[node_i_idx] >= 2 and graph.degree[node_j_idx] >= 2:
                        dist = np.sqrt((graph.nodes[node_i_idx]['data'].x - graph.nodes[node_j_idx]['data'].x)**2 +
                                       (graph.nodes[node_i_idx]['data'].y - graph.nodes[node_j_idx]['data'].y)**2)
                        if dist < distance_threshold:
                            return True
        return False


    merge_splitting_node = True

    while __has_joint_nodes_adj() and merge_splitting_node:
        min_dist = np.inf
        splitting_nodes_to_be_merge = []
        for node_i_idx in graph.nodes:
            for node_j_idx in graph.nodes:
                if node_i_idx != node_j_idx:
                    if graph.degree[node_i_idx] >= 2 and graph.degree[node_j_idx] >= 2:
                        dist = np.sqrt((graph.nodes[node_i_idx]['data'].x - graph.nodes[node_j_idx]['data'].x) ** 2 +
                                       (graph.nodes[node_i_idx]['data'].y - graph.nodes[node_j_idx]['data'].y) ** 2)
                        if dist < min_dist:
                            min_dist = dist
                            splitting_nodes_to_be_merge = [node_i_idx, node_j_idx]

        n_node_before = len(graph.nodes)
        # remove node_j
        if len(splitting_nodes_to_be_merge) == 2 and min_dist < distance_threshold:
            node_idx_to_delete = splitting_nodes_to_be_merge[1]
            node_idx_to_retain = splitting_nodes_to_be_merge[0]

            neighbor_nodes = list(graph.neighbors(node_idx_to_delete))
            for neighbor_node in neighbor_nodes:
                if neighbor_node != node_idx_to_retain:
                    if len(graph.edges[neighbor_node, node_idx_to_delete].keys()) > 0:
                        graph.add_edge(node_idx_to_retain, neighbor_node, data=graph.edges[neighbor_node, node_idx_to_delete]['data'])
                        graph.remove_edge(neighbor_node, node_idx_to_delete)
                    else:
                        graph.add_edge(node_idx_to_retain, neighbor_node)
                        graph.remove_edge(neighbor_node, node_idx_to_delete)

            graph.remove_node(node_idx_to_delete)
            print(f"[x] merge_splitting_nodes, {(node_idx_to_delete, node_idx_to_retain)} with distance {min_dist}, node {node_idx_to_delete} has been deleted")
        n_node_after = len(graph.nodes)

        if n_node_after == n_node_before:
            merge_splitting_node = False
    return graph


def remove_isolated_nodes(graph):
    # remove degree one node generated by re-connection
    nodes_to_be_delete = []
    for edge in list(graph.edges):
        if len(graph.edges[edge[0], edge[1]].keys()) == 0:
            if graph.degree[edge[0]] == 1:
                nodes_to_be_delete.append(edge[0])
            if graph.degree[edge[1]] == 1:
                nodes_to_be_delete.append(edge[1])

    print(f"[x] nodes_to_be_delete = {nodes_to_be_delete}")
    graph.remove_nodes_from(nodes_to_be_delete)
    return graph

# step 8
def remove_cycle(graph: nx.Graph, cycle_length_threshold: int):
    cycles_in_graph = __find_all_cycles(graph, cycle_length_limit=cycle_length_threshold)
    for cycle_in_graph in cycles_in_graph:
        pass
    return graph


def save_pickle_objs(obj, file_path):
    pickle.dump(obj, open(file_path, 'wb'))


def assign_unique_label(g: nx.Graph, save_path, patient_name, original_image):
    g_switch = nx.Graph()
    for node in g.nodes():
        print(node)

    # add new node
    for idx, edge in enumerate(g.edges()):
        vessel_segment = g.edges[edge[0], edge[1]]['data']
        g_switch.add_node(idx, data=vessel_segment, ori_node_index=[edge[0], edge[1]])

    for idx, edge in enumerate(g.edges()):
        neighbor_nodes = g.neighbors(edge[0])
        # find current node index in g_swith
        # current_node_index_in_g_swith = 0
        for node in g_switch.nodes:
            ori_node_index = g_switch.nodes[node]['ori_node_index']
            if ori_node_index[0] == edge[0] and ori_node_index[1] == edge[1]:
                current_node_index_in_g_swith = node
            elif ori_node_index[1] == edge[0] and ori_node_index[0] == edge[1]:
                current_node_index_in_g_swith = node

        for neighbor_node in neighbor_nodes:
            if neighbor_node != edge[1]:
                for node in g_switch.nodes:
                    ori_node_index = g_switch.nodes[node]['ori_node_index']
                    if ori_node_index[0] == edge[0] and ori_node_index[1] == neighbor_node:
                        g_switch.add_edge(node, current_node_index_in_g_swith)
                    elif ori_node_index[1] == edge[0] and ori_node_index[0] == neighbor_node:
                        g_switch.add_edge(node, current_node_index_in_g_swith)

        neighbor_nodes = g.neighbors(edge[1])
        # find current node index in g_swith
        # current_node_index_in_g_swith = 0
        for node in g_switch.nodes:
            ori_node_index = g_switch.nodes[node]['ori_node_index']
            if ori_node_index[0] == edge[1] and ori_node_index[1] == edge[1]:
                current_node_index_in_g_swith = node
            elif ori_node_index[1] == edge[1] and ori_node_index[0] == edge[1]:
                current_node_index_in_g_swith = node

        for neighbor_node in neighbor_nodes:
            if neighbor_node != edge[0]:
                for node in g_switch.nodes:
                    ori_node_index = g_switch.nodes[node]['ori_node_index']
                    if ori_node_index[0] == edge[1] and ori_node_index[1] == neighbor_node:
                        g_switch.add_edge(node, current_node_index_in_g_swith)
                    elif ori_node_index[1] == edge[1] and ori_node_index[0] == neighbor_node:
                        g_switch.add_edge(node, current_node_index_in_g_swith)
    print(g_switch)

    # find LMA first
    lma_segment = None
    for idx, node in enumerate(g_switch.nodes()):
        print(f"node index = {idx}, ori = {g_switch.nodes[node]['ori_node_index']}")
        if g_switch.nodes[node]['data'].vessel_class == 'LMA':
            lma_segment = g_switch.nodes[node]['data']
            if g.degree[g_switch.nodes[node]['ori_node_index'][0]] != 1:
                ori_lma_bi_node_index = g_switch.nodes[node]['ori_node_index'][0]
            else:
                ori_lma_bi_node_index = g_switch.nodes[node]['ori_node_index'][1]
            break

    # 1. handle LAD first
    n_lad_segments = 0
    for idx, node in enumerate(g_switch.nodes()):
        ori_node_index = g_switch.nodes[node]['ori_node_index']
        if g_switch.nodes[node]['data'].vessel_class == 'LAD':
            n_lad_segments += 1

    current_connected_node_idx = ori_lma_bi_node_index
    for j in range(n_lad_segments):
        for idx, node in enumerate(g_switch.nodes()):
            if g_switch.nodes[node]['data'].vessel_class == 'LAD':
                ori_node_indexes = g_switch.nodes[node]['ori_node_index']
                if ori_node_indexes[0] == current_connected_node_idx:
                    g_switch.nodes[node]['data'].vessel_class = f"LAD{j + 1}"
                    current_connected_node_idx = ori_node_indexes[1]
                    assign_branches(current_connected_node_idx, g_switch, j + 1, "D")
                    break
                elif ori_node_indexes[1] == current_connected_node_idx:
                    g_switch.nodes[node]['data'].vessel_class = f"LAD{j + 1}"
                    current_connected_node_idx = ori_node_indexes[0]
                    assign_branches(current_connected_node_idx, g_switch, j + 1, "D")
                    break

    # 2.handle LCX artery branches
    n_lcx_segments = 0
    for idx, node in enumerate(g_switch.nodes()):
        ori_node_index = g_switch.nodes[node]['ori_node_index']
        if g_switch.nodes[node]['data'].vessel_class == 'LCX':
            n_lcx_segments += 1

    current_connected_node_idx = ori_lma_bi_node_index
    for j in range(n_lcx_segments):
        for idx, node in enumerate(g_switch.nodes()):
            if g_switch.nodes[node]['data'].vessel_class == 'LCX':
                ori_node_indexes = g_switch.nodes[node]['ori_node_index']
                if ori_node_indexes[0] == current_connected_node_idx:
                    g_switch.nodes[node]['data'].vessel_class = f"LCX{j + 1}"
                    current_connected_node_idx = ori_node_indexes[1]
                    assign_branches(current_connected_node_idx, g_switch, j + 1, "OM")
                    break
                elif ori_node_indexes[1] == current_connected_node_idx:
                    g_switch.nodes[node]['data'].vessel_class = f"LCX{j + 1}"
                    current_connected_node_idx = ori_node_indexes[0]
                    assign_branches(current_connected_node_idx, g_switch, j + 1, "OM")
                    break

    print("[x] renamed artery tree")
    for idx, node in enumerate(g_switch.nodes()):
        print(f"node index = {idx}, class = {g_switch.nodes[node]['data'].vessel_class}, ori = {g_switch.nodes[node]['ori_node_index']}")

    labeldict = {}
    for node_index in g_switch.nodes():
        # labeldict[node_index] = f"{node_index}|{g_switch.nodes[node_index]['ori_node_index']}\n{g_switch.nodes[node_index]['data'].vessel_class}"
        labeldict[node_index] = f"{g_switch.nodes[node_index]['data'].vessel_class}"
    nx.draw(g_switch, labels=labeldict, with_labels=True, node_color="#3498DB", width=2.0, font_size=12, font_color="w", node_size=900)

    plt.savefig(os.path.join(save_path, patient_name, f"{patient_name}_step12_g_switch_unique.png"))
    plt.close()

    nxg.visualize_semantic_image_unique(g_switch,
                                        original_image=original_image,
                                        semantic_mapping=constants.semantic_mapping,
                                        save_path=os.path.join(save_path, patient_name, f"{patient_name}_step12_g_switch_unique_semantic_image.png"))
    return g_switch


def assign_branches(current_connected_node_idx, switch_graph, branch_index, pattern):
    for idx, node in enumerate(switch_graph.nodes()):
        if switch_graph.nodes[node]['data'].vessel_class == pattern:
            ori_node_indexes = switch_graph.nodes[node]['ori_node_index']
            if ori_node_indexes[0] == current_connected_node_idx:
                switch_graph.nodes[node]['data'].vessel_class = f"{pattern}{branch_index}"
                break
            elif ori_node_indexes[1] == current_connected_node_idx:
                switch_graph.nodes[node]['data'].vessel_class = f"{pattern}{branch_index}"
                break
    return switch_graph


def process_single_patient(data_dir,
                           spacing_csv_path, patient_name, save_dir,
                           radius_threshold,
                           centerline_length_threshold,
                           image_dst_size=512,
                           visualize=False):
    print("-------------------------------------------------------------------------------------------------------------")
    print(f"[x] process patient [{patient_name}]")

    original_file_path = os.path.join(data_dir, 'training/{}.png'.format(patient_name))
    # copy this image to the folder
    shutil.copy(src=original_file_path, dst=os.path.join(save_dir, f"{patient_name}.png"))
    semantic_files_path = os.path.join(data_dir, 'image', patient_name)
    pixel_spacing = get_spacing(spacing_csv_path, patient_name)

    original_image = cv2.imread(original_file_path, cv2.IMREAD_GRAYSCALE)
    if original_image.shape[0] != image_dst_size:
        original_image = cv2.resize(original_image, (image_dst_size, image_dst_size), cv2.INTER_NEAREST)

    semantic_image, binary_image = generate_images(semantic_files_path, original_image.shape[0],
                                                   constants.color_map, constants.artery_keep_dict_tw)
    Image.fromarray(semantic_image).save(os.path.join(save_dir, f"{patient_name}_semantic_image.png"))
    Image.fromarray(binary_image).save(os.path.join(save_dir, f"{patient_name}_binary_image.png"))

    # step 1: generate center lines
    skeleton_image = skeleton_by_python(os.path.join(save_dir, f"{patient_name}_binary_image.png"))
    # skeleton_image = skeleton_by_matlab(os.path.join(save_dir, f"{patient_name}_binary_image.png"), eng)
    skeleton_image = remove_isolated_segment(skeleton_image)
    visualizer.visualize_binary_image([skeleton_image], [1.0], os.path.join(save_dir, f"{patient_name}_step1_skeleton.png"))

    # step 2: calculate end points and joint points
    nodes, joint_nodes, end_nodes = calculate_end_and_joint_points_by_python(os.path.join(save_dir, f"{patient_name}_step1_skeleton.png"))
    #nodes, joint_nodes, end_nodes = calculate_end_and_joint_points_by_matlab(os.path.join(save_dir, f"{patient_name}_step1_skeleton.png"), eng)
    visualizer.visualize_artery_tree_with_points(skeleton_image, end_nodes, joint_nodes,
                                                 os.path.join(save_dir, f"{patient_name}_step2_skeleton_with_points.png"))

    # step3:
    # generate vessel objects: generate objects according to extracted artery tree
    # filter vessel objects: remove the vessel segment with the maximum diameter less than threshold
    # graph, skeleton = graph_generation_by_connectivity(patient_name, nodes, skeleton_image,
    #                                                    semantic_image, constants.semantic_mapping, eng)

    graph, skeleton = graph_generation_by_connectivity_python(patient_name, nodes, skeleton_image,
                                                              semantic_image, constants.semantic_mapping)

    if visualize:
        nxg.visualize_semantic_image(graph, original_image, constants.semantic_mapping,
                                     save_path=os.path.join(save_dir, f"{patient_name}_step3_generated_graph.png"),
                                     centerline=True)

        visualizer.visualize_binary_image([skeleton], [1.0], save_path=os.path.join(save_dir, f"{patient_name}_step3_generated_skeleton.png"))

        nxg.draw_simple_graph(graph, save_path=os.path.join(save_dir, f"{patient_name}_step3_simple_graph.png"))

    graph = reconnect(graph, radius=centerline_length_threshold)

    if visualize:
        nxg.draw_simple_graph(graph, save_path=os.path.join(save_dir, f"{patient_name}_step3_simple_reconnected_graph.png"))

    dist_map = diameter_map_python(os.path.join(save_dir, f"{patient_name}_binary_image.png"))
    graph = filter_vessel(graph, dist_map, pixel_spacing,
                          radius_threshold=radius_threshold,
                          centerline_length_threshold=centerline_length_threshold)

    # step 4: map each centerline to vessel pixels
    # mapped_artery_mask = find_vessel_pixel_in_centerline(graph, binary_image)
    mapped_artery_mask = find_vessel_pixel_in_centerline_dist(graph, binary_image, dist_map)
    if visualize:
        nxg.visualize_networkx_graph(graph, mapped_artery_mask,
                                     save_path=os.path.join(save_dir, f"{patient_name}_step4_mapped_artery_mask.png"))

    #                              os.path.join(save_dir, f"{patient_name}_step5_mapped_semantic_centerline.png"), centerline=True)
    # step 6: merge nodes
    merge_and_filter = True
    merge_iter = 0
    while merge_and_filter:
        n_node = len(list(graph.nodes))
        graph = merge_degree_two(graph)
        graph = filter_vessel(graph, dist_map, pixel_spacing, radius_threshold, centerline_length_threshold)
        if len(list(graph.nodes)) == n_node:
            merge_and_filter = False
        merge_iter += 1

        if visualize:
            nxg.visualize_semantic_image(graph, original_image, constants.semantic_mapping,
                                         os.path.join(save_dir, f"{patient_name}_step6_merged{merge_iter}_semantic_centerline.png"), centerline=True)

    if visualize:
        nxg.draw_simple_graph(graph, save_path=os.path.join(save_dir, f"{patient_name}_step6_simple_graph.png"))
    # step 7: merge splitting node
    graph = merge_splitting_nodes(graph, distance_threshold=centerline_length_threshold)
    if visualize:
        nxg.draw_simple_graph(graph, save_path=os.path.join(save_dir, f"{patient_name}_step7_simple_graph.png"))
        nxg.visualize_semantic_image(graph, original_image, constants.semantic_mapping,
                                     save_path=os.path.join(save_dir, f"{patient_name}_step7_merged_semantic_centerline.png"),
                                     centerline=True)

    # step 8: remove isolated nodes
    graph = remove_isolated_nodes(graph)
    if visualize:
        nxg.draw_simple_graph(graph, save_path=os.path.join(save_dir, f"{patient_name}_step8_simple_graph.png"))
        nxg.visualize_semantic_image(graph, original_image, constants.semantic_mapping,
                                     save_path=os.path.join(save_dir, f"{patient_name}_step8_merged_semantic_centerline.png"),
                                     centerline=True)

    # step 9: merge degree-two node again for just one iteration
    graph = merge_degree_two(graph)
    if visualize:
        nxg.draw_simple_graph(graph, save_path=os.path.join(save_dir, f"{patient_name}_step9_simple_graph.png"))
        nxg.visualize_semantic_image(graph, original_image, constants.semantic_mapping,
                                     save_path=os.path.join(save_dir, f"{patient_name}_step9_merged_semantic_centerline.png"),
                                     centerline=True)
        nxg.draw_simple_graph_with_coord(graph, save_path=os.path.join(save_dir, f"{patient_name}_step9_simple_graph_coord.png"))

    # step 10: reindex nodes and edges
    mapping = {}
    for idx, node in enumerate(graph.nodes):
        mapping[node] = idx
    g_new = nx.relabel_nodes(graph, mapping, copy=True)
    # nxg.draw_simple_graph(g_new, save_path=os.path.join(save_dir, f"{patient_name}_step10_simple_graph.png"))
    if visualize:
        nxg.draw_simple_graph_with_coord(g_new, save_path=os.path.join(save_dir, f"{patient_name}_step10_simple_graph_coord.png"))
        nxg.visualize_semantic_image(g_new, original_image, constants.semantic_mapping,
                                     save_path=os.path.join(save_dir, f"{patient_name}_step10_merged_semantic_centerline.png"),
                                     centerline=True, point=True)
        nxg.visualize_semantic_image(g_new, original_image, constants.semantic_mapping,
                                     save_path=os.path.join(save_dir, f"{patient_name}_step10_merged_semantic_centerline_clean.png"),
                                     centerline=True, point=False)
        nxg.visualize_semantic_image(g_new, original_image, constants.semantic_mapping,
                                     save_path=os.path.join(save_dir, f"{patient_name}_step10_merged_semantic_artery_clean.png"),
                                     centerline=False, point=False)

    # save objects
    save_pickle_objs(g_new, file_path=os.path.join(save_dir, "{}.pkl".format(patient_name)))
    plt.close()

    # step 11. switch edge and node
    g_switch = nx.Graph()
    for node in g_new.nodes():
        print(node)

    # add new node
    for idx, edge in enumerate(g_new.edges()):
        vessel_segment = g_new.edges[edge[0], edge[1]]['data']
        g_switch.add_node(idx, data=vessel_segment, ori_node_index=[edge[0], edge[1]])

    for idx, edge in enumerate(g_new.edges()):
        neighbor_nodes = g_new.neighbors(edge[0])
        # find current node index in g_swith
        # current_node_index_in_g_swith = 0
        for node in g_switch.nodes:
            ori_node_index = g_switch.nodes[node]['ori_node_index']
            if ori_node_index[0] == edge[0] and ori_node_index[1] == edge[1]:
                current_node_index_in_g_swith = node
            elif ori_node_index[1] == edge[0] and ori_node_index[0] == edge[1]:
                current_node_index_in_g_swith = node

        for neighbor_node in neighbor_nodes:
            if neighbor_node != edge[1]:
                for node in g_switch.nodes:
                    ori_node_index = g_switch.nodes[node]['ori_node_index']
                    if ori_node_index[0] == edge[0] and ori_node_index[1] == neighbor_node:
                        g_switch.add_edge(node, current_node_index_in_g_swith)
                    elif ori_node_index[1] == edge[0] and ori_node_index[0] == neighbor_node:
                        g_switch.add_edge(node, current_node_index_in_g_swith)

        neighbor_nodes = g_new.neighbors(edge[1])
        # find current node index in g_swith
        # current_node_index_in_g_swith = 0
        for node in g_switch.nodes:
            ori_node_index = g_switch.nodes[node]['ori_node_index']
            if ori_node_index[0] == edge[1] and ori_node_index[1] == edge[1]:
                current_node_index_in_g_swith = node
            elif ori_node_index[1] == edge[1] and ori_node_index[0] == edge[1]:
                current_node_index_in_g_swith = node

        for neighbor_node in neighbor_nodes:
            if neighbor_node != edge[0]:
                for node in g_switch.nodes:
                    ori_node_index = g_switch.nodes[node]['ori_node_index']
                    if ori_node_index[0] == edge[1] and ori_node_index[1] == neighbor_node:
                        g_switch.add_edge(node, current_node_index_in_g_swith)
                    elif ori_node_index[1] == edge[1] and ori_node_index[0] == neighbor_node:
                        g_switch.add_edge(node, current_node_index_in_g_swith)

    print(g_switch)
    for idx, node in enumerate(g_switch.nodes()):
        print(f"node index = {idx}, ori = {g_switch.nodes[node]['ori_node_index']}")

    labeldict = {}
    for node_index in g_switch.nodes():
        # labeldict[node_index] = f"{node_index}|{g_switch.nodes[node_index]['ori_node_index']}\n{g_switch.nodes[node_index]['data'].vessel_class}"
        labeldict[node_index] = f"{g_switch.nodes[node_index]['data'].vessel_class}"
    nx.draw(g_switch, labels=labeldict, with_labels=True)

    plt.savefig(os.path.join(args.save_dir, patient_name, f"{patient_name}_step11_g_switch.png"))
    plt.close()

    pickle.dump(g_switch, open(os.path.join(args.save_dir, patient_name, f"{patient_name}_g_switch.pkl"), 'wb'))

    # step 12 visualize graph
    # def visualize_graph(vessel_infos, original_image, save_path, semantic_mapping, assign_label=False):
    # vessel_infos = [graph.edges[edge[0], edge[1]]['data'],]

    if visualize:
        vessel_infos = [g_switch.nodes[i]['data'] for i in g_switch.nodes]
        visualizer.visualize_graph(
            vessel_infos=vessel_infos,
            original_image=np.zeros_like(original_image),
            save_path=os.path.join(args.save_dir, patient_name, f"{patient_name}_step12_adj.png"),
            semantic_mapping=constants.semantic_mapping,
            assign_label=False)

        visualizer.visualize_graph(
            vessel_infos=vessel_infos,
            original_image=np.zeros_like(original_image),
            save_path=os.path.join(args.save_dir, patient_name, f"{patient_name}_step12_adj2.png"),
            semantic_mapping=constants.semantic_mapping,
            assign_label=True)

    # assign unique labels
    g_switch_unique = assign_unique_label(g_new, args.save_dir, patient_name, original_image)
    pickle.dump(g_switch_unique, open(os.path.join(args.save_dir, patient_name, f"{patient_name}_g_switch_unique.pkl"), 'wb'))
    # quit MATLAB
########################################################################################################################


def data_graph_generation(data_path='/home/miil/Downloads/automated-labelme/labelme/data',
                         save_dir='/home/miil/Downloads/automated-labelme/labelme/data/processed',
                         centerline_length_threshold=8,
                         radius_threshold=0.5,
                         cardiac_pattern='LCA',
                         num_threads=1,
                         patient_name='EEEEEEEE_11'):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default=data_path)
    parser.add_argument('--save_dir', type=str, default=save_dir)
    parser.add_argument('--centerline_length_threshold', type=float, default=centerline_length_threshold)
    parser.add_argument('--radius_threshold', type=float, default=radius_threshold)
    parser.add_argument('--cardiac_pattern', type=str, default=cardiac_pattern)
    parser.add_argument('--num_threads', type=int, default=num_threads)
    parser.add_argument('--patient_name', type=str, default=patient_name)
    parser.add_argument('--visualize', type=bool, default=True)

    global args
    args = parser.parse_args()
    files = sorted(os.listdir(os.path.join(args.data_path, "training")))
    files = [f for f in files if f.endswith(".png") and f.rfind(args.cardiac_pattern) != -1]
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    if args.patient_name == "all":
        f = open(os.path.join(args.save_dir, "config.txt"), "w")
        f.write(json.dumps(args.__dict__, indent=4))
        f.close()
        # process on all subjects
        for file in tqdm(files):
            patient_name = file[:file.rfind('.')]
            # if os.path.isfile(os.path.join(args.save_dir, "{}.pkl".format(patient_name))):
            #     continue
            if not os.path.isdir(os.path.join(args.save_dir, patient_name)):
                os.makedirs(os.path.join(args.save_dir, patient_name))
            config_f = open(os.path.join(args.save_dir, args.patient_name, "config.json"), "w")
            json.dump(args.__dict__, config_f, indent=4)

            process_single_patient(data_dir=args.data_path,
                                   spacing_csv_path=os.path.join(args.data_path, "pixel_spacing.csv"),
                                   patient_name=patient_name,
                                   save_dir=os.path.join(args.save_dir, patient_name),
                                   radius_threshold=args.radius_threshold,
                                   centerline_length_threshold=args.centerline_length_threshold,
                                   visualize=args.visualize)
            print(f"[x] patient {patient_name} done")
    else:
        print("[x] process {}".format(args.patient_name))
        start_time = time.time()
        if not os.path.isdir(os.path.join(args.save_dir, args.patient_name)):
            os.makedirs(os.path.join(args.save_dir, args.patient_name))

        config_f = open(os.path.join(args.save_dir, args.patient_name, "config.json"), "w")
        json.dump(args.__dict__, config_f, indent=4)

        process_single_patient(data_dir=args.data_path,
                               spacing_csv_path=os.path.join(args.data_path, "pixel_spacing.csv"),
                               patient_name=args.patient_name,
                               save_dir=os.path.join(args.save_dir, args.patient_name),
                               radius_threshold=args.radius_threshold,
                               centerline_length_threshold=args.centerline_length_threshold,
                               visualize=args.visualize)
        print(f"[x] processing time : {time.time() - start_time} s")