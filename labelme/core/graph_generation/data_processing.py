"""
CODE to generate artery graphs.
"""
import json

import matlab.engine
import networkx as nx
import sys
sys.path.append("../../")
import pandas as pd
import webcolors
import numpy as np
import os
import math
import cv2
import matplotlib.pyplot as plt
import pickle
import argparse

from scipy.interpolate import interp1d
from tqdm import tqdm
from skimage import measure
from glob import glob

from labelme.core.utils.module import Node, VesselSegment
from labelme.core.graph_generation import visualization as visualizer
from labelme.core.graph_generation import networkx_graph_visualizer as nxg
from PIL import Image

import io
from labelme.core.graph_generation import constants

########################################################################################################################
# code for step 1
def skeleton_by_matlab(file_path, eng):
    """
    calculate the skeleton by matlab function
    :param file_path: file path to binary image_x
    :return:
    """
    # path = 'path/to/some/dir/file.txt'
    # with open(path, 'w') as f:
    #     with contextlib.redirect_stdout(f):
    #         print('Hello, World')

    ret = eng.skeleton_matlab(file_path, stdout=io.StringIO())
    skeleton_image = np.array(ret, dtype=np.int32)
    return skeleton_image


def remove_isolated_segment(skeleton_image):
    labeling = measure.label(skeleton_image)
    regions = measure.regionprops(labeling)
    largest_region = None
    area_max = 0.
    for region in regions:
        if region.area > area_max:
            area_max = region.area
            largest_region = region

    bin_image = np.zeros_like(skeleton_image)
    for coord in largest_region.coords:
        bin_image[coord[0], coord[1]] = 1

    return bin_image


########################################################################################################################
# code for step 2
def calculate_end_and_joint_points_by_matlab(vessel_file_path, eng):
    """
    generate the joint points and end points according to matlab code
    :param vessel_file_path:
    :param eng:
    :param show:
    :return:
        points: each point is a tuple, and the first element is the column, the second is the row
    """
    ret = eng.graph_point_wrapper(vessel_file_path, nargout=5, stdout=io.StringIO())
    rj, cj, re, ce, skeleton_image = np.array(ret[0], dtype=np.int32).flatten() - 1, \
                                     np.array(ret[1], dtype=np.int32).flatten() - 1, \
                                     np.array(ret[2], dtype=np.int32).flatten() - 1, \
                                     np.array(ret[3], dtype=np.int32).flatten() - 1, \
                                     np.array(ret[4], dtype=np.int32)
    image_size = skeleton_image.shape[0]
    end_points = []
    joint_points = []

    nodes = []

    for i in range(len(rj)):
        joint_points.append((rj[i], cj[i]))
    for i in range(len(ce)):
        end_points.append((re[i], ce[i]))

    # for each point, the first element is the column, the second is the row

    joint_nodes = []
    for point in joint_points:
        adj_matrix = skeleton_image[point[0] - 1:point[0] + 2, point[1] - 1:point[1] + 2]
        degree = np.sum(adj_matrix == 1) - skeleton_image[point[0], point[1]]
        nodes.append(Node(degree, point[0], point[1]))
        joint_nodes.append(Node(degree, point[0], point[1]))
        #print("joint point {}, degree = {}".format(point, degree))

    end_nodes = []
    for point in end_points:
        adj_matrix = skeleton_image[point[0] - 1:point[0] + 2, point[1] - 1:point[1] + 2]
        degree = np.sum(adj_matrix == 1) - skeleton_image[point[0], point[1]]
        nodes.append(Node(degree, point[0], point[1]))
        end_nodes.append(Node(degree, point[0], point[1]))
        #print("end point {}, degree = {}".format(point, degree))

    return nodes, joint_nodes, end_nodes



########################################################################################################################
# code for step 3
def diameter_map(binary_file_path, eng):
    ret = eng.matlab_diameter_map(binary_file_path, stdout=io.StringIO())
    dist_map = np.array(ret, dtype=np.float32)
    return dist_map


def filter_vessel(vessel_objs: list, g: nx.Graph, nodes: list, dist_map,
                  patient_name, csv_path, radius_threshold, centerline_length_threshold):
    v_objs = []
    pixel_spacing = get_spacing(csv_path, patient_name)

    for idx, vessel_obj in enumerate(vessel_objs):
        segment_dist_map = vessel_obj.vessel_centerline * dist_map
        # assign distance map to each vessel object
        vessel_obj.vessel_centerline_dist = segment_dist_map
        # average_dist = np.mean(segment_dist_map[segment_dist_map > 0])
        mean_dist = np.mean(segment_dist_map[segment_dist_map > 0])
        min_dist = np.min(segment_dist_map[segment_dist_map > 0])
        max_dist = np.max(segment_dist_map[segment_dist_map > 0])

        if max_dist * pixel_spacing >= radius_threshold and np.sum(vessel_obj.vessel_centerline) > centerline_length_threshold:
            v_objs.append(vessel_obj)
        else:
            # remove vessel_object
            if vessel_obj.node1.degree == 1:
                # remove edge with end points: in implementation, remove node will automatically remove edge
                g.remove_node(nodes.index(vessel_obj.node1))
            elif vessel_obj.node2.degree == 1:
                # remove node with end points: in implementation, remove node will automatically remove edge
                g.remove_node(nodes.index(vessel_obj.node2))
            else:
                # remove edge without end points
                g.remove_edge(nodes.index(vessel_obj.node1), nodes.index(vessel_obj.node2))
                if not nx.is_connected(g):
                    g.add_edge(nodes.index(vessel_obj.node1), nodes.index(vessel_obj.node2))
                    v_objs.append(vessel_obj)

    return v_objs, g


def merge_vessel(vessel_objects, graph, nodes):
    v_objs = []

    def __get_vessel_obj_by_edge(edge_start, edge_end):
        for vessel_object in vessel_objects:
            if nodes.index(edge_points[0]):
                pass

    # 1. remove degree-2 nodes and merge vessel objects
    for node_idx in list(graph.nodes):
        if graph.degree[node_idx] == 2:
            # get edge
            connected_edges = list(graph.edges(node_idx))


            # merge related edge

    # for idx, vessel_obj in enumerate(vessel_objects):
        # segment_dist_map = vessel_obj.vessel_centerline * dist_map
        # assign distance map to each vessel object
        # vessel_obj.vessel_centerline_dist = segment_dist_map
        # # average_dist = np.mean(segment_dist_map[segment_dist_map > 0])
        # mean_dist = np.mean(segment_dist_map[segment_dist_map > 0])
        # min_dist = np.min(segment_dist_map[segment_dist_map > 0])
        # max_dist = np.max(segment_dist_map[segment_dist_map > 0])

        # # remove vessel_object
        # if vessel_obj.node1.degree == 1:
        #     # remove edge with end points: in implementation, remove node will automatically remove edge
        #     g.remove_node(nodes.index(vessel_obj.node1))
        # elif vessel_obj.node2.degree == 1:
        #     # remove node with end points: in implementation, remove node will automatically remove edge
        #     g.remove_node(nodes.index(vessel_obj.node2))
        # else:
        #     # remove edge without end points
        #     g.remove_edge(nodes.index(vessel_obj.node1), nodes.index(vessel_obj.node2))
        #     if not nx.is_connected(g):
        #         g.add_edge(nodes.index(vessel_obj.node1), nodes.index(vessel_obj.node2))
        #         v_objs.append(vessel_obj)

    return vessel_objects, graph

def get_spacing(pixel_spacing_csv_path, patient_name):
    df = pd.read_csv(pixel_spacing_csv_path)
    df = df[df['patient']==patient_name]
    return df['spacing'].values[0]


def graph_generation_by_connectivity(patient_name, nodes, skeleton_img, eng=None):
    """
    creating VesselSegment objects according to the detected nodes
    :param nodes:
    :param skeleton_img:
    :param eng:
    :return:
    """
    g = nx.Graph()
    vessel_objects = []
    image_size = skeleton_img.shape[0]
    for idx, node in enumerate(nodes):
        g.add_nodes_from([(idx, {"x": node.x, "y": node.y})])
        for i in range(node.x-1, node.x+2):
            for j in range(node.y-1, node.y+2):
                if i < skeleton_img.shape[0] and j < skeleton_img.shape[1]:
                    skeleton_img[i, j] = 0

    #labeling = measure.label(skeleton_img, neighbors=8, connectivity=1)
    labeling = measure.label(skeleton_img)

    for idx, label in enumerate(np.unique(labeling)[1:]):
        vessel_segment = np.zeros_like(labeling)
        vessel_segment[labeling == label] = 1

        # find end points by matlab
        if not os.path.isdir(os.path.join("temp", patient_name)):
            os.makedirs(os.path.join("temp", patient_name))
        cv2.imwrite(os.path.join("temp", patient_name, "temp.png"), vessel_segment)
        ret = eng.vessel_segment_point(os.path.join("temp", patient_name, "temp.png"), nargout=4, stdout=io.StringIO())
        rj, cj, re, ce = np.array(ret[0], dtype=np.int32).flatten() - 1, \
                         np.array(ret[1], dtype=np.int32).flatten() - 1, \
                         np.array(ret[2], dtype=np.int32).flatten() - 1, \
                         np.array(ret[3], dtype=np.int32).flatten() - 1

        if re.shape[0] == 2:
            #print("rj {}, cj {}, re {}, ce {}".format(rj, cj, re, ce))
            edge_points = []
            for i in range(2):
                near_dist = np.inf
                near_node = None
                node_idx = 0
                for idx, node in enumerate(nodes):
                    end_point_x, end_point_y = re[i], ce[i]
                    dist = np.sqrt((end_point_x - node.x) ** 2 + (end_point_y - node.y) ** 2)
                    if dist < near_dist:
                        near_dist = dist
                        near_node = node
                        node_idx = idx
                edge_points.append(near_node)

            fig, ax = plt.subplots()
            plt.axis('off')
            fig.set_size_inches(image_size/100, image_size/100)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0.0, wspace=0.0)
            plt.margins(0, 0)
            plt.imshow(vessel_segment, cmap="gray")
            plt.scatter(edge_points[0].y, edge_points[0].x, marker='+')
            plt.scatter(edge_points[1].y, edge_points[1].x, marker='+')
            plt.savefig(f"temp/{patient_name}/edge_{idx}.png")
            plt.close()

            vessel_object = VesselSegment(edge_points[0], edge_points[1], vessel_segment)
            vessel_objects.append(vessel_object)
            # add a new edge and
            g.add_node(nodes.index(edge_points[0]), obj=edge_points[0])
            g.add_node(nodes.index(edge_points[1]), obj=edge_points[1])
            g.add_edge(nodes.index(edge_points[0]), nodes.index(edge_points[1]), obj=vessel_object)

        # else:
        #     print("rj {}, cj {}, re {}, ce {}".format(rj, cj, re, ce))
        #     plt.imshow(vessel_segment, cmap="gray")
        #     plt.savefig("tmpa.png")
        #     plt.close()
        #
        # fig, ax = plt.subplots()
        # plt.axis('off')
        # fig.set_size_inches(image_size/100, image_size/100)
        # ax.set_xticks([])
        # ax.set_yticks([])
        # plt.gca().xaxis.set_major_locator(plt.NullLocator())
        # plt.gca().yaxis.set_major_locator(plt.NullLocator())
        # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0.0, wspace=0.0)
        # plt.margins(0, 0)
        #
        # plt.imshow(skeleton_img, cmap='gray')
        # plt.scatter([n.y for n in nodes if n.degree > 2],
        #             [n.x for n in nodes if n.degree > 2],
        #             c='r', linewidth=1, marker='*')
        # plt.scatter([n.y for n in nodes if n.degree == 1],
        #             [n.x for n in nodes if n.degree == 1],
        #             c='g', linewidth=1, marker='+')
        # plt.scatter([n.y for n in nodes if n.degree == 2],
        #             [n.x for n in nodes if n.degree == 2],
        #             c='b', linewidth=1, marker='o')
        # plt.savefig(save_path)
        # plt.close()

    return vessel_objects, g


# the code below is for point turning
def distance(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

def point_line_distance(point, start, end):
    if start == end:
        return distance(point, start)
    else:
        n = abs((end[0] - start[0]) * (start[1] - point[1]) - (start[0] - point[0]) * (end[1] - start[1]))
        d = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
        return n / d


########################################################################################################################
# code for step 4
def __add_adjacent_points(coord, binary_image, adj_size=3):
    r_points = []
    for i in range(-(adj_size // 2), (adj_size // 2) + 1):
        for j in range(-(adj_size // 2), (adj_size // 2) + 1):
            if i == j == 0:
                continue

            if (coord[0] + i) < binary_image.shape[0] and (coord[0] + i) >= 0:
                if (coord[1] + j) < binary_image.shape[1] and (coord[1] + j) >= 0:
                    if binary_image[coord[0] + i, coord[1] + j] > 0:
                        r_points.append((coord[0] + i, coord[1] + j))
    return r_points


def __find_vessel_pixel(center_point, slope, binary_image, offset=10, interpolation_interval=0.1, adj_size=5):
    assert len(center_point) == 2
    # to right
    vessel_pixel_coords = []
    vessel_pixel_coords.append((int(center_point[0]), int(center_point[1])))
    ## the code is for calculate the `right` part of the vessel
    for delta_x in np.arange(interpolation_interval, offset, interpolation_interval):
        if slope == np.inf:
            current_x = center_point[0]
            current_y = center_point[1] + delta_x
        else:
            current_x = center_point[0] + delta_x
            current_y = slope * delta_x + center_point[1]
        current_y = np.clip(current_y, 2, binary_image.shape[0] - 2)
        current_x = np.clip(current_x, 2, binary_image.shape[0] - 2)
        if distance(center_point, (current_x, current_y)) < offset:
            if binary_image[int(np.round(current_x)), int(np.round(current_y))] > 0:
                vessel_pixel_coords.append((int(np.round(current_x)), int(np.round(current_y))))
                vessel_pixel_coords.extend(__add_adjacent_points((int(np.round(current_x)), int(np.round(current_y))), binary_image, adj_size))
            elif binary_image[int(np.round(current_x)), int(np.ceil(current_y))] > 0:
                vessel_pixel_coords.append((int(np.round(current_x)), int(np.ceil(current_y))))
                vessel_pixel_coords.extend(__add_adjacent_points((int(np.round(current_x)), int(np.ceil(current_y))), binary_image, adj_size))
            elif binary_image[int(np.ceil(current_x)), int(np.ceil(current_y))] > 0:
                vessel_pixel_coords.append((int(np.ceil(current_x)), int(np.ceil(current_y))))
                vessel_pixel_coords.extend(__add_adjacent_points((int(np.ceil(current_x)), int(np.ceil(current_y))), binary_image, adj_size))
            elif binary_image[int(np.ceil(current_x)), int(np.round(current_y))] > 0:
                vessel_pixel_coords.append((int(np.ceil(current_x)), int(np.round(current_y))))
                vessel_pixel_coords.extend(__add_adjacent_points((int(np.ceil(current_x)), int(np.round(current_y))), binary_image, adj_size))
            else:
                break

    ## the code is for calculate the `left` part of the vessel
    for delta_x in np.arange(interpolation_interval, offset, interpolation_interval):
        if slope == np.inf:
            current_x = center_point[0]
            current_y = center_point[1] - delta_x
        else:
            current_x = center_point[0] - delta_x
            current_y = slope * (-delta_x) + center_point[1]
        current_y = np.clip(current_y, 2, binary_image.shape[0] - 2)
        if distance(center_point, (current_x, current_y)) < offset:
            if binary_image[int(np.round(current_x)), int(np.round(current_y))] > 0:
                vessel_pixel_coords.append((int(np.round(current_x)), int(np.round(current_y))))
                vessel_pixel_coords.extend(
                    __add_adjacent_points((int(np.round(current_x)), int(np.round(current_y))), binary_image,
                                          adj_size))
            elif binary_image[int(np.round(current_x)), int(np.ceil(current_y))] > 0:
                vessel_pixel_coords.append((int(np.round(current_x)), int(np.ceil(current_y))))
                vessel_pixel_coords.extend(
                    __add_adjacent_points((int(np.round(current_x)), int(np.ceil(current_y))), binary_image,
                                          adj_size))
            elif binary_image[int(np.ceil(current_x)), int(np.ceil(current_y))] > 0:
                vessel_pixel_coords.append((int(np.ceil(current_x)), int(np.ceil(current_y))))
                vessel_pixel_coords.extend(
                    __add_adjacent_points((int(np.ceil(current_x)), int(np.ceil(current_y))), binary_image,
                                          adj_size))
            elif binary_image[int(np.ceil(current_x)), int(np.round(current_y))] > 0:
                vessel_pixel_coords.append((int(np.ceil(current_x)), int(np.round(current_y))))
                vessel_pixel_coords.extend(
                    __add_adjacent_points((int(np.ceil(current_x)), int(np.round(current_y))), binary_image,
                                          adj_size))
            else:
                break

    return vessel_pixel_coords


def __calculate_perpendicular_line_slope(point, vessel_points):
    min_dist = np.inf
    min_point = None
    for vessel_point in vessel_points:
        dist = distance((point[0], point[1]), (vessel_point[0], vessel_point[1]))
        if dist > 0 and dist < min_dist:
            min_dist = dist
            min_point = vessel_point
    if min_point[0] == point[0]:
        # current line is perpendicular to x-axis, so the perpendicular line is parallel to x-axis with K=0
        # print("x - node = {}, min_point = {}, slope = 0.".format((point[0], point[1]), (min_point[0], min_point[1])))
        return 0.
    elif min_point[1] == point[1]:
        # current line is perpendicular to y-axis, so the perpendicular line is parallel to y-axis with K=np.inf
        # print("y - node = {}, min_point = {}, slope = np.inf".format((point[0], point[1]), (min_point[0], min_point[1])))
        return np.inf
    else:
        slope = (min_point[1] - point[1]) / (min_point[0] - point[0])
        slope = -1 / slope
        # print("node = {}, min_point = {}, slope = {}".format((point[0], point[1]), (min_point[0], min_point[1]), slope))
        return slope


def __interpolate_centerline_points(vessel_info, interpolation_method='cubic', interp_num=5):
    vessel_points = []
    num_points_in_centerline = len(np.where(vessel_info.vessel_centerline == 1)[0])
    for i in range(num_points_in_centerline):
        vessel_points.append((np.where(vessel_info.vessel_centerline == 1)[0][i],
                              np.where(vessel_info.vessel_centerline == 1)[1][i]))
    vessel_points = np.array(vessel_points)
    assert len(vessel_points.shape) == 2
    assert vessel_points.shape[1] == 2

    alpha = np.linspace(0, 1, interp_num * num_points_in_centerline)

    distance = np.cumsum(np.sqrt(np.sum(np.diff(vessel_points, axis=0) ** 2, axis=1)))
    distance = np.insert(distance, 0, 0) / distance[-1]
    interpolator = interp1d(distance, vessel_points, kind=interpolation_method, axis=0)
    interpolated_points = interpolator(alpha)  # shape = (interp_num,2)
    r_points = []
    for i in range(interp_num * num_points_in_centerline):
        r_points.append((interpolated_points[i][0], interpolated_points[i][1]))
    return r_points


def __generate_vessel_binary_map(binary_image, vessel_pixel_coords, vessel_mask):
    for vessel_pixel_coord in vessel_pixel_coords:
        if binary_image[vessel_pixel_coord[0], vessel_pixel_coord[1]] > 0:
            vessel_mask[vessel_pixel_coord[0], vessel_pixel_coord[1]] = 1
    return vessel_mask


def find_vessel_pixel_in_centerline(vessel_infos, binary_image):
    mapped_mask = np.zeros_like(binary_image)
    for vessel_info in vessel_infos:
        vessel_points = __interpolate_centerline_points(vessel_info, 'slinear', interp_num=5)
        vessel_mask = np.zeros_like(binary_image)
        for vessel_point in vessel_points:
            slope = __calculate_perpendicular_line_slope(vessel_point, vessel_points)
            vessel_pixel_coordinates = __find_vessel_pixel(vessel_point, slope, binary_image,
                                                           offset=8,
                                                           interpolation_interval=0.5,
                                                           adj_size=2)
            vessel_mask = __generate_vessel_binary_map(binary_image, vessel_pixel_coordinates, vessel_mask)

        vessel_info.vessel_mask = vessel_mask
        mapped_mask += vessel_mask

        """
        if show:
            fig, ax = plt.subplots()
            ax.imshow(vessel_info.vessel_centerline, cmap='gray')
            ax.imshow(vessel_mask, alpha=0.6, cmap='gray')
            plt.axis('off')
            fig.set_size_inches(5.12, 5.12)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0.0, wspace=0.0)
            plt.margins(0, 0)
            plt.show()
        """

    mapped_mask = np.clip(mapped_mask, 0, 1)
    print(f"# pixels in binary image = {np.sum(binary_image>0)}, # pixels in mapped image = {np.sum(mapped_mask > 0)}, ratio = {np.sum(mapped_mask > 0)/np.sum(binary_image>0)}")

    return vessel_infos, mapped_mask

########################################################################################################################
# code for step 5
def assign_class(vessel_infos: list, semantic_image, semantic_mapping: dict):
    def __get_class(coord):
        for key in semantic_mapping.keys():
            # color to label
            if semantic_image[coord[0], coord[1], 0] == semantic_mapping[key][0]:
                if semantic_image[coord[0], coord[1], 1] == semantic_mapping[key][1]:
                    if semantic_image[coord[0], coord[1], 2] == semantic_mapping[key][2]:
                        return key
        return constants.VESSEL_OTHER_COLOR_KEY

    for vessel_object in vessel_infos:
        num_pixels = np.sum(vessel_object.vessel_centerline>0)
        count_dict = {k: 0 for k in semantic_mapping.keys()}

        for i in range(num_pixels):
            try:
                vessel_pixel_coord = (np.where(vessel_object.vessel_centerline == 1)[0][i], np.where(vessel_object.vessel_centerline == 1)[1][i])
                clazz = __get_class(vessel_pixel_coord)
                count_dict[clazz] = count_dict[clazz]+1
            except:
                vessel_object.vessel_centerline[vessel_pixel_coord[0], vessel_pixel_coord[1]] = 0
                print("coord = {}, color = {}".format(vessel_pixel_coord, semantic_image[vessel_pixel_coord[0], vessel_pixel_coord[1]]))
        #print(count_dict)
        count_dict = sorted(count_dict.items(), key=lambda x: x[1])[::-1]
        vessel_object.vessel_class = count_dict[0][0]

    return vessel_infos

########################################################################################################################
def save_vessel_objs(vessel_objects, file_path):
    pickle.dump(vessel_objects, open(file_path, 'wb'))


def generate_semantic_image(semantic_images_path, image_size, colormap, search_priority):
    semantic_image_paths = glob(os.path.join(semantic_images_path, "*.png"))

    semantic_image = np.zeros([image_size, image_size, 3], dtype=np.uint8)

    for pattern in search_priority:
        for semantic_image_path in semantic_image_paths:
            semantic_label = semantic_image_path[semantic_image_path.rfind("_") + 1: semantic_image_path.rfind(".png")]
            if semantic_label.rfind(pattern) !=-1:
                semantic_image_segment = cv2.imread(semantic_image_path, cv2.IMREAD_GRAYSCALE)
                semantic_image[np.where(semantic_image_segment > 0)[0], np.where(semantic_image_segment > 0)[1], :] = colormap[semantic_label]
    return semantic_image


def process_single_patient(eng, data_dir, spacing_csv_path, patient_name, save_dir, radius_threshold, centerline_length_threshold):
    original_file_path = os.path.join(data_dir, 'training/{}.png'.format(patient_name))
    binary_file_path = os.path.join(data_dir, 'label/{}.png'.format(patient_name))
    semantic_files_path = os.path.join(data_dir, 'image', patient_name)

    original_image = cv2.imread(original_file_path, cv2.IMREAD_GRAYSCALE)
    binary_image = cv2.imread(binary_file_path, cv2.IMREAD_GRAYSCALE)

    # step 1: generate center lines
    skeleton_image = skeleton_by_matlab(binary_file_path, eng)
    skeleton_image = remove_isolated_segment(skeleton_image)
    visualizer.visualize_binary_image([skeleton_image], [1.0], os.path.join(save_dir, f"{patient_name}_step1_skeleton.png"))

    # step 2: calculate end points and joint points
    nodes, joint_nodes, end_nodes = calculate_end_and_joint_points_by_matlab(os.path.join(save_dir, f"{patient_name}_step1_skeleton.png"), eng)
    visualizer.visualize_artery_tree_with_points(skeleton_image, end_nodes, joint_nodes,
                                                 os.path.join(save_dir, f"{patient_name}_step2_skeleton_with_points.png"))

    # step3:
    # generate vessel objects: generate objects according to extracted artery tree
    # filter vessel objects: remove the vessel segment with the average diameter less than threshold
    original_vessel_objects, graph = graph_generation_by_connectivity(patient_name, nodes, skeleton_image, eng)
    nxg.visualize_networkx_graph(graph, skeleton_image,
                                        save_path=os.path.join(save_dir, f"{patient_name}_step3-1_generated_graph_networkx.png"))
    dist_map = diameter_map(os.path.join(data_dir, f'label/{patient_name}.png'), eng)
    filtered_vessel_objects, graph = filter_vessel(original_vessel_objects, graph, nodes, dist_map, patient_name,
                                                   csv_path=spacing_csv_path,
                                                   radius_threshold=radius_threshold,
                                                   centerline_length_threshold=centerline_length_threshold)

    visualizer.visualize_graph(original_vessel_objects, original_image,
                               save_path=os.path.join(save_dir, f"{patient_name}_step3-1_generated_graph.png"),
                               semantic_mapping=constants.semantic_mapping, assign_label=False)

    visualizer.visualize_graph(filtered_vessel_objects, original_image,
                               save_path=os.path.join(save_dir, f"{patient_name}_step3-1_filtered_graph.png"),
                               semantic_mapping=constants.semantic_mapping, assign_label=False)

    filtered_vessel_objects, graph = merge_vessel(filtered_vessel_objects, graph, nodes)

    # step 4: map each centerline to vessel pixels
    vessel_objects, mapped_artery_mask = find_vessel_pixel_in_centerline(filtered_vessel_objects, binary_image)
    visualizer.visualize_binary_image([skeleton_image, mapped_artery_mask], [1.0, 0.5],
                               os.path.join(save_dir, f"{patient_name}_step4_vessel_search.png"))

    # step 5: assign class information to each vessel segments
    semantic_image = generate_semantic_image(semantic_files_path, mapped_artery_mask.shape[0], constants.color_map)
    im = Image.fromarray(semantic_image)
    im.save(os.path.join(save_dir, f"{patient_name}_semantic_image.png"))

    vessel_objects = assign_class(vessel_objects, semantic_image, constants.semantic_mapping)

    # visualizer.visualize_semantic_image(vessel_objects, original_image, constants.semantic_mapping,
    #                                     os.path.join(save_dir, f"{patient_name}_step5_mapped_semantic_label.png"), centerline=False)
    # visualizer.visualize_semantic_image(vessel_objects, original_image, constants.semantic_mapping,
    #                                     os.path.join(save_dir, f"{patient_name}_step5_mapped_semantic_centerline.png"), centerline=True)

    visualizer.visualize_semantic_image2(vessel_objects, original_image, constants.semantic_mapping,
                                        os.path.join(save_dir, f"{patient_name}_step5_mapped_semantic_centerline.png"), centerline=True)
    # visualizer.visualize_semantic_image2(vessel_objects, original_image, constants.semantic_mapping,
                                        # os.path.join(save_dir, f"{patient_name}_step5_mapped_semantic_label.png"), centerline=False)

    visualizer.visualize_graph(vessel_objects,
                               original_image,
                               save_path=os.path.join(save_dir, "{}_step5_generated_graph.png".format(patient_name)),
                               semantic_mapping=constants.semantic_mapping,
                               assign_label=True)

    # save objects
    save_vessel_objs(vessel_objects, file_path=os.path.join(save_dir, "{}.pkl".format(patient_name)))

    # original_vessel_objects = assign_class(original_vessel_objects, semantic_image, constants.semantic_mapping)
    # save_vessel_objs(original_vessel_objects, file_path=os.path.join(save_dir, "{}_ori.pkl".format(patient_name)))
    # save_vessel_objs(graph, file_path=os.path.join(save_dir, "{}_graph.pkl".format(patient_name)))
    # save_vessel_objs(nodes, file_path=os.path.join(save_dir, "{}_nodes.pkl".format(patient_name)))


########################################################################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/media/zhaochen/data/fluoro_gnn/data/data_no_semantic')
    parser.add_argument('--save_dir', type=str, default='/media/zhaochen/data/fluoro_gnn/data/data_no_semantic/processed')
    parser.add_argument('--centerline_length_threshold', type=float, default=10)
    parser.add_argument('--radius_threshold', type=float, default=1.8)
    parser.add_argument('--cardiac_pattern', type=str, default='LCA')
    parser.add_argument('--patient_name', type=str, default='7_LCA_RAO2')

    args = parser.parse_args()

    eng = matlab.engine.start_matlab()
    files = sorted(os.listdir(os.path.join(args.data_path, "label")))

    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)


    if args.patient_name == "all":
        f = open(os.path.join(args.save_dir, "config.txt"), "w")
        f.write(json.dumps(args.__dict__, indent=4))
        f.close()

        # process on all subjects
        for file in tqdm(files):
            if file.endswith(".png"):
                patient_name = file[:file.rfind('.')]
                # if os.path.isfile(os.path.join(args.save_dir, "{}.pkl".format(patient_name))):
                #     continue
                if patient_name.rfind(args.cardiac_pattern) != -1:
                    print("[x] process {}".format(file))
                    #def process_single_patient(eng, data_dir, spacing_csv_path, patient_name, save_dir, mean_dist_threshold):
                    process_single_patient(eng=eng,
                                           data_dir=args.data_path,
                                           spacing_csv_path=os.path.join    (args.data_path, "pixel_spacing.csv"),
                                           patient_name=patient_name,
                                           save_dir=args.save_dir,
                                           radius_threshold=args.radius_threshold,
                                           centerline_length_threshold=args.centerline_length_threshold)
                print(f"[x] patient {patient_name} done")
                # time.sleep(10)
    else:
        print("[x] process {}".format(args.patient_name))
        # def process_single_patient(eng, data_dir, spacing_csv_path, patient_name, save_dir, mean_dist_threshold):
        process_single_patient(eng=eng,
                               data_dir=args.data_path,
                               spacing_csv_path=os.path.join(args.data_path, "pixel_spacing.csv"),
                               patient_name=args.patient_name,
                               save_dir=args.save_dir,
                               radius_threshold=args.radius_threshold,
                               centerline_length_threshold=args.centerline_length_threshold)
    # quit MATLAB
    eng.quit()
    eng.exit()
