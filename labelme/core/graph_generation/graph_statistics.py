import networkx as nx
import os
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt

import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)

sys.path.insert(0, parentdir)
sys.path.insert(0, os.path.dirname(parentdir))



from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='/media/z/data2/gmn_vessel/data/data_tw_semantic/processed2')
    parser.add_argument('--node_threshold', type=int, default=5)
    parser.add_argument('--edge_threshold', type=int, default=5)
    args = parser.parse_args()

    subjects = []
    for subject in os.listdir(args.save_dir):
        if os.path.isdir(os.path.join(args.save_dir, subject)):
            print(subject)
            subjects.append(subject)

    print(f"# {len(subjects)} subjects detected in the {args.save_dir}")
    num_edges = []
    num_nodes = []

    for subject in subjects:
        g = pickle.load(open(os.path.join(args.save_dir, subject, f"{subject}_g_switch.pkl"), "rb"))
        num_nodes.append(len(g.nodes))
        num_edges.append(len(g.edges))
        print(f"[x] process subject = {subject}, # nodes = {len(g.nodes)}, # edges = {len(g.edges)}")

    print(f"edges, min = {np.min(num_edges)}, max = {np.max(num_edges)}, mean = {np.mean(num_edges)}, std = {np.std(num_edges)}")
    print(f"nodes, min = {np.min(num_nodes)}, max = {np.max(num_nodes)}, mean = {np.mean(num_nodes)}, std = {np.std(num_nodes)}")


    fig, axs = plt.subplots(nrows=1, ncols=2)

    axs[0].hist(num_nodes, bins=20)
    axs[0].set_title("node histogram")
    axs[1].hist(num_edges, bins=20)
    axs[1].set_title("edge histogram")

    plt.show()

    f = open("patient_list.txt", "w")
    selected_subjects = []
    for subject in subjects:
        g = pickle.load(open(os.path.join(args.save_dir, subject, f"{subject}.pkl"), "rb"))
        num_nodes.append(len(g.nodes))
        num_edges.append(len(g.edges))
        if len(g.nodes) >= args.node_threshold and len(g.edges) >= args.edge_threshold:
            selected_subjects.append(subject)
        f.write(f"{subject}\n")
    f.flush()
    f.close()

    print(f"# selected subjects = {len(selected_subjects)}")


