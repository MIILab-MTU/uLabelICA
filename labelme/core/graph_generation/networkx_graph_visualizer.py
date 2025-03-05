import numpy as np
import webcolors
import networkx as nx
import matplotlib.pyplot as plt
from core.graph_generation.constants import DEBUG


def visualize_semantic_image_unique(graph: nx.Graph, original_image, semantic_mapping, save_path):
    """
    visualize pre-processed graph with unique artery labels
    """
    fig, ax = plt.subplots()
    ax.imshow(original_image, cmap="gray")
    assigned_nodes = []
    
    print(len(graph.nodes))
    print(graph.nodes[0])
    for node in list(graph.nodes):
        vessel_obj = graph.nodes[node]['data']
        # plot points
        # visualize artery centerlines
        vessel_class = ''.join([i for i in vessel_obj.vessel_class if not i.isdigit()])
        plt.scatter(np.where(vessel_obj.vessel_centerline==1)[1], np.where(vessel_obj.vessel_centerline==1)[0],
                    color=webcolors.rgb_to_hex(semantic_mapping[vessel_class]), s=1)
        label_x = np.where(vessel_obj.vessel_centerline==1)[1][len(np.where(vessel_obj.vessel_centerline==1)[1])//2]
        label_y = np.where(vessel_obj.vessel_centerline==1)[0][len(np.where(vessel_obj.vessel_centerline==1)[0])//2]
        print(f"{label_x},{label_y}")
        ax.annotate(vessel_obj.vessel_class, (label_x, label_y), fontSize=15, color="w")

    plt.axis('off')
    fig.set_size_inches(original_image.shape[0] / 100, original_image.shape[0] / 100)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0.0, wspace=0.0)
    plt.margins(0, 0)

    plt.savefig(save_path)
    plt.close()


def visualize_networkx_graph(graph: nx.Graph, skeleton_image, save_path, debug=DEBUG):
    fig, ax = plt.subplots()
    plt.axis('off')
    fig.set_size_inches(skeleton_image.shape[0] / 100, skeleton_image.shape[1] / 100)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0.0, wspace=0.0)
    plt.margins(0, 0)
    plt.imshow(skeleton_image, alpha=0.6, cmap='gray')

    joint_nodes = []
    degree2_nodes = []
    end_nodes = []
    print(graph.degree)
    for node_index in list(graph.nodes):
        if graph.degree[node_index] == 1:
            end_nodes.append(graph.nodes[node_index]['data'])
        elif graph.degree[node_index] == 2:
            degree2_nodes.append(graph.nodes[node_index]['data'])
        else:
            joint_nodes.append(graph.nodes[node_index]['data'])

    plt.scatter([node.y for node in joint_nodes], [node.x for node in joint_nodes], c='r', linewidth=1, marker='*')
    plt.scatter([node.y for node in end_nodes], [node.x for node in end_nodes], c='g', linewidth=1, marker='+')
    plt.scatter([node.y for node in degree2_nodes], [node.x for node in degree2_nodes], c='b', linewidths=1, marker='o')
    if debug:
        plt.show()
    else:
        plt.savefig(save_path)
    plt.close()


def visualize_semantic_image(graph: nx.Graph, original_image, semantic_mapping, save_path, centerline=False, point=True, debug=DEBUG):
    fig, ax = plt.subplots()
    ax.imshow(original_image, cmap="gray")
    assigned_labels = []
    assigned_nodes = []

    for edge in list(graph.edges):
        if len(graph.edges[edge[0], edge[1]].keys()) > 0:
            vessel_obj = graph.edges[edge[0], edge[1]]['data']
            # plot points
            if point:
                if graph.degree[edge[0]] == 1:
                    plt.scatter(graph.nodes[edge[0]]['data'].y, graph.nodes[edge[0]]['data'].x, c='g', linewidth=1, marker='+')
                elif graph.degree[edge[0]] == 2:
                    plt.scatter(graph.nodes[edge[0]]['data'].y, graph.nodes[edge[0]]['data'].x, c='b', linewidth=1, marker='o')
                else:
                    plt.scatter(graph.nodes[edge[0]]['data'].y, graph.nodes[edge[0]]['data'].x, c='r', linewidth=1, marker='*')

                if graph.degree[edge[1]] == 1:
                    plt.scatter(graph.nodes[edge[1]]['data'].y, graph.nodes[edge[1]]['data'].x, c='g', linewidth=1, marker='+')
                elif graph.degree[edge[1]] == 2:
                    plt.scatter(graph.nodes[edge[1]]['data'].y, graph.nodes[edge[1]]['data'].x, c='b', linewidth=1, marker='o')
                else:
                    plt.scatter(graph.nodes[edge[1]]['data'].y, graph.nodes[edge[1]]['data'].x, c='r', linewidth=1, marker='*')

                # annotate point index
                if edge[0] not in assigned_nodes:
                    ax.annotate(edge[0], (graph.nodes[edge[0]]['data'].y, graph.nodes[edge[0]]['data'].x), fontSize=15, color='w')
                    assigned_nodes.append(edge[0])

                if edge[1] not in assigned_nodes:
                    ax.annotate(edge[1], (graph.nodes[edge[1]]['data'].y, graph.nodes[edge[1]]['data'].x), fontSize=15, color='w')
                    assigned_nodes.append(edge[1])

            if centerline:
                # visualize artery centerlines
                if vessel_obj.vessel_class not in assigned_labels:
                    plt.scatter(np.where(vessel_obj.vessel_centerline==1)[1], np.where(vessel_obj.vessel_centerline==1)[0],
                                color=webcolors.rgb_to_hex(semantic_mapping[vessel_obj.vessel_class]),
                                label=vessel_obj.vessel_class, s=1)
                    assigned_labels.append(vessel_obj.vessel_class)
                else:
                    plt.scatter(np.where(vessel_obj.vessel_centerline==1)[1], np.where(vessel_obj.vessel_centerline==1)[0],
                                color=webcolors.rgb_to_hex(semantic_mapping[vessel_obj.vessel_class]), s=1)
            else:
                # visualize artery segments
                if vessel_obj.vessel_class not in assigned_labels:
                    plt.scatter(np.where(vessel_obj.vessel_mask==1)[1], np.where(vessel_obj.vessel_mask==1)[0],
                                color=webcolors.rgb_to_hex(semantic_mapping[vessel_obj.vessel_class]),
                                label=vessel_obj.vessel_class)
                    assigned_labels.append(vessel_obj.vessel_class)
                else:
                    plt.scatter(np.where(vessel_obj.vessel_mask==1)[1], np.where(vessel_obj.vessel_mask==1)[0],
                                color=webcolors.rgb_to_hex(semantic_mapping[vessel_obj.vessel_class]))

    plt.legend()
    plt.axis('off')
    fig.set_size_inches(original_image.shape[0] / 100, original_image.shape[0] / 100)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0.0, wspace=0.0)
    plt.margins(0, 0)

    if debug:
        plt.show()
    else:
        plt.savefig(save_path)
    plt.close()


def draw_simple_graph(graph: nx.Graph, save_path, debug=DEBUG):
    labeldict = {}
    for node_index in graph.nodes():
        labeldict[node_index] = f"{node_index}|{graph.degree[node_index]}"
    nx.draw(graph, labels=labeldict, with_labels=True)
    if debug:
        plt.show()
    else:
        plt.savefig(save_path)
    plt.close()


def draw_simple_graph_with_coord(graph: nx.Graph, save_path, debug=DEBUG):
    pos = {}
    for node_index in graph.nodes():
        pos[node_index] = (graph.nodes[node_index]['data'].y, graph.nodes[node_index]['data'].x)

    # degree one nodes GREEN
    node_list = [node_index for node_index in graph.nodes() if graph.degree[node_index] == 1]
    nx.draw_networkx_nodes(graph, pos, nodelist=node_list, node_size=20, node_color="tab:green", label=node_list)

    # degree two nodes BLUE
    node_list = [node_index for node_index in graph.nodes() if graph.degree[node_index] == 2]
    nx.draw_networkx_nodes(graph, pos, nodelist=node_list, node_size=20, node_color="yellow", label=node_list)

    # degree greater than 3 nodes RED
    node_list = [node_index for node_index in graph.nodes() if graph.degree[node_index] > 2]
    nx.draw_networkx_nodes(graph, pos, nodelist=node_list, node_size=20, node_color="tab:red", label=node_list)

    nx.draw_networkx_edges(graph, pos, alpha=0.5, width=1)

    fig, ax = plt.subplots()
    plt.axis('off')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()

    labeldict = {}
    for node_index in graph.nodes():
        # labeldict[node_index] = f"{node_index}|{graph.degree[node_index]}"
        labeldict[node_index] = f"{node_index}"
    nx.draw(graph, labels=labeldict, with_labels=True, node_color="#3498DB", width=2.0, font_size=15, font_color="w", node_size=600)
    if debug:
        plt.show()
    else:
        plt.savefig(save_path)

    plt.close()

