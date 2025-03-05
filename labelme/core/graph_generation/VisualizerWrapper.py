import matplotlib.pyplot as plt
import numpy as np
import webcolors

class Visualizer(object):
    @staticmethod
    def visualize_semantic_image(vessel_infos, original_image, semantic_mapping, centerline=True):
        processed_semantic_vessel = np.zeros([original_image.shape[0], original_image.shape[1], 3])
        for vessel_object in vessel_infos:
            if centerline:
                processed_semantic_vessel[:, :, 0][vessel_object.vessel_centerline == 1] = \
                semantic_mapping[vessel_object.vessel_class][0]
                processed_semantic_vessel[:, :, 1][vessel_object.vessel_centerline == 1] = \
                semantic_mapping[vessel_object.vessel_class][1]
                processed_semantic_vessel[:, :, 2][vessel_object.vessel_centerline == 1] = \
                semantic_mapping[vessel_object.vessel_class][2]
            else:
                processed_semantic_vessel[:, :, 0][vessel_object.vessel_mask == 1] = \
                semantic_mapping[vessel_object.vessel_class][0]
                processed_semantic_vessel[:, :, 1][vessel_object.vessel_mask == 1] = \
                semantic_mapping[vessel_object.vessel_class][1]
                processed_semantic_vessel[:, :, 2][vessel_object.vessel_mask == 1] = \
                semantic_mapping[vessel_object.vessel_class][2]

        fig, ax = plt.subplots()
        ax.imshow(processed_semantic_vessel)
        plt.axis('off')
        fig.set_size_inches(original_image.shape[0] / 100, original_image.shape[1] / 100)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0.0, wspace=0.0)
        plt.margins(0, 0)
        plt.show()
        plt.close()

    @staticmethod
    def visualize_semantic_image2(vessel_infos, original_image, semantic_mapping, centerline=False):
        image_size = original_image.shape[0]
        fig, ax = plt.subplots()
        ax.imshow(original_image, cmap="gray")
        assigned_labels = []
        for vessel_info in vessel_infos:
            if vessel_info.node1.degree == 1:
                plt.scatter(vessel_info.node1.y, vessel_info.node1.x, c='g', linewidth=1, marker='+')
            elif vessel_info.node1.degree == 2:
                plt.scatter(vessel_info.node1.y, vessel_info.node1.x, c='b', linewidth=1, marker='o')
            else:
                plt.scatter(vessel_info.node1.y, vessel_info.node1.x, c='r', linewidth=1, marker='*')

            if vessel_info.node2.degree == 1:
                plt.scatter(vessel_info.node2.y, vessel_info.node2.x, c='g', linewidth=1, marker='+')
            elif vessel_info.node2.degree == 2:
                plt.scatter(vessel_info.node2.y, vessel_info.node2.x, c='b', linewidth=1, marker='o')
            else:
                plt.scatter(vessel_info.node2.y, vessel_info.node2.x, c='r', linewidth=1, marker='*')

            if centerline:
                if vessel_info.vessel_class not in assigned_labels:
                    plt.scatter(np.where(vessel_info.vessel_centerline == 1)[1],
                                np.where(vessel_info.vessel_centerline == 1)[0],
                                color=webcolors.rgb_to_hex(semantic_mapping[vessel_info.vessel_class]),
                                label=vessel_info.vessel_class, s=1)
                    assigned_labels.append(vessel_info.vessel_class)
                else:
                    plt.scatter(np.where(vessel_info.vessel_centerline == 1)[1],
                                np.where(vessel_info.vessel_centerline == 1)[0],
                                color=webcolors.rgb_to_hex(semantic_mapping[vessel_info.vessel_class]), s=1)
            else:
                if vessel_info.vessel_class not in assigned_labels:
                    plt.scatter(np.where(vessel_info.vessel_mask == 1)[1], np.where(vessel_info.vessel_mask == 1)[0],
                                color=webcolors.rgb_to_hex(semantic_mapping[vessel_info.vessel_class]),
                                label=vessel_info.vessel_class)
                    assigned_labels.append(vessel_info.vessel_class)
                else:
                    plt.scatter(np.where(vessel_info.vessel_mask == 1)[1], np.where(vessel_info.vessel_mask == 1)[0],
                                color=webcolors.rgb_to_hex(semantic_mapping[vessel_info.vessel_class]))

        plt.legend()
        plt.axis('off')
        fig.set_size_inches(image_size / 100, image_size / 100)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0.0, wspace=0.0)
        plt.margins(0, 0)
        plt.show()
        plt.close()

    @staticmethod
    def visualize_graph(vessel_infos, original_image, semantic_mapping, assign_label=False):
        image_size = original_image.shape[0]
        fig, ax = plt.subplots()
        ax.imshow(original_image, cmap="gray")
        assigned_labels = []
        for vessel_info in vessel_infos:
            if vessel_info.node1.degree == 1:
                plt.scatter(vessel_info.node1.y, vessel_info.node1.x, c='g', linewidth=1, marker='+')
            elif vessel_info.node1.degree == 2:
                plt.scatter(vessel_info.node1.y, vessel_info.node1.x, c='b', linewidth=1, marker='o')
            else:
                plt.scatter(vessel_info.node1.y, vessel_info.node1.x, c='r', linewidth=1, marker='*')

            if vessel_info.node2.degree == 1:
                plt.scatter(vessel_info.node2.y, vessel_info.node2.x, c='g', linewidth=1, marker='+')
            elif vessel_info.node2.degree == 2:
                plt.scatter(vessel_info.node2.y, vessel_info.node2.x, c='b', linewidth=1, marker='o')
            else:
                plt.scatter(vessel_info.node2.y, vessel_info.node2.x, c='r', linewidth=1, marker='*')

            if assign_label:
                if vessel_info.vessel_class not in assigned_labels:
                    plt.plot([vessel_info.node1.y, vessel_info.node2.y], [vessel_info.node1.x, vessel_info.node2.x],
                             '-',
                             color=webcolors.rgb_to_hex(semantic_mapping[vessel_info.vessel_class]),
                             label=vessel_info.vessel_class)
                    assigned_labels.append(vessel_info.vessel_class)
                else:
                    plt.plot([vessel_info.node1.y, vessel_info.node2.y], [vessel_info.node1.x, vessel_info.node2.x],
                             '-',
                             color=webcolors.rgb_to_hex(semantic_mapping[vessel_info.vessel_class]))
            else:
                # assign white
                plt.plot([vessel_info.node1.y, vessel_info.node2.y], [vessel_info.node1.x, vessel_info.node2.x], '-',
                         color="#ffffff")

        if assign_label:
            plt.legend()

        plt.axis('off')
        fig.set_size_inches(image_size / 100, image_size / 100)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0.0, wspace=0.0)
        plt.margins(0, 0)
        plt.show()
        plt.close()

    @staticmethod
    def visualize_centerline(vessel_infos, original_image, with_ori=True, with_joint=True):
        image_size = original_image.shape[0]
        fig, ax = plt.subplots()
        # ax.imshow(original_image, cmap="gray")
        if with_ori:
            plt.imshow(original_image, cmap='gray')

        plt.axis('off')
        fig.set_size_inches(image_size / 100, image_size / 100)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0.0, wspace=0.0)
        plt.margins(0, 0)

        centerline_img = np.zeros_like(original_image)

        for vessel_info in vessel_infos:
            if vessel_info.node1.degree == 1:
                if with_joint:
                    # plt.scatter(vessel_info.node1.y, vessel_info.node1.x, c='g', linewidth=2, marker='+', s=100)
                    plt.scatter(vessel_info.node1.y, vessel_info.node1.x, c='limegreen', linewidth=2, marker='*', s=100)
            elif vessel_info.node1.degree == 2:
                plt.scatter(vessel_info.node1.y, vessel_info.node1.x, c='b', linewidth=1, marker='o')
            else:
                if with_joint:
                    plt.scatter(vessel_info.node1.y, vessel_info.node1.x, c='r', linewidth=2, marker='*', s=100)

            if vessel_info.node2.degree == 1:
                if with_joint:
                    # plt.scatter(vessel_info.node1.y, vessel_info.node1.x, c='g', linewidth=2, marker='+', s=100)
                    plt.scatter(vessel_info.node2.y, vessel_info.node2.x, c='limegreen', linewidth=2, marker='*', s=100)
            elif vessel_info.node2.degree == 2:
                plt.scatter(vessel_info.node2.y, vessel_info.node2.x, c='b', linewidth=1, marker='o')
            else:
                if with_joint:
                    plt.scatter(vessel_info.node2.y, vessel_info.node2.x, c='r', linewidth=2, marker='*', s=100)

            centerline_img[vessel_info.vessel_centerline > 0] = 1.

        if with_ori:
            plt.imshow(centerline_img, alpha=0.6, cmap='gray')
        else:
            plt.imshow(centerline_img, cmap='gray')
        plt.show()
        plt.close()

    @staticmethod
    def visualize_binary_image(binary_images, alphas):
        fig, ax = plt.subplots()
        for idx, binary_image in enumerate(binary_images):
            ax.imshow(binary_image, alpha=alphas[idx], cmap=plt.cm.gray)
        plt.axis('off')
        fig.set_size_inches(binary_images[0].shape[0] / 100, binary_images[0].shape[1] / 100)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0.0, wspace=0.0)
        plt.margins(0, 0)
        plt.show()
        plt.close()

    @staticmethod
    def visualize_artery_tree_with_points(skeleton_image, end_nodes, joint_nodes):
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
        plt.scatter([node.y for node in joint_nodes], [node.x for node in joint_nodes], c='r', linewidth=1, marker='*')
        plt.scatter([node.y for node in end_nodes], [node.x for node in end_nodes], c='g', linewidth=1, marker='+')
        plt.show()
        plt.close()