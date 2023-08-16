"""
Code taken from https://github.com/NVlabs/6dof-graspnet

Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto.  Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

import numpy as np

def farthest_points(
        data,
        nclusters,
        dist_func,
        return_center_indexes=False,
        return_distances=False,
        verbose=False):
    """
      Code taken from https://github.com/NVlabs/6dof-graspnet

      Performs farthest point sampling on data points.
      Args:
        data: numpy array of the data points.
        nclusters: int, number of clusters.
        dist_dunc: distance function that is used to compare two data points.
        return_center_indexes: bool, If True, returns the indexes of the center of
          clusters.
        return_distances: bool, If True, return distances of each point from centers.

      Returns clusters, [centers, distances]:
        clusters: numpy array containing the cluster index for each element in
          data.
        centers: numpy array containing the integer index of each center.
        distances: numpy array of [npoints] that contains the closest distance of
          each point to any of the cluster centers.
    """
    if nclusters >= data.shape[0]:
        if return_center_indexes:
            return np.arange(
                data.shape[0], dtype=np.int32), np.arange(
                data.shape[0], dtype=np.int32)

        return np.arange(data.shape[0], dtype=np.int32)

    clusters = np.ones((data.shape[0],), dtype=np.int32) * -1
    distances = np.ones((data.shape[0],), dtype=np.float32) * 1e7
    centers = []
    for iter in range(nclusters):
        index = np.argmax(distances)
        centers.append(index)
        shape = list(data.shape)
        for i in range(1, len(shape)):
            shape[i] = 1

        broadcasted_data = np.tile(np.expand_dims(data[index], 0), shape)
        new_distances = dist_func(broadcasted_data, data)
        distances = np.minimum(distances, new_distances)
        clusters[distances == new_distances] = iter
        if verbose:
            print(
                'farthest points max distance : {}'.format(
                    np.max(distances)))

    if return_center_indexes:
        if return_distances:
            return clusters, np.asarray(centers, dtype=np.int32), distances
        return clusters, np.asarray(centers, dtype=np.int32)

    return clusters


def distance_by_translation_grasp(p1, p2):
    """
      Gets two nx4x4 numpy arrays and computes the translation of all the
      grasps.
    """
    t1 = p1[:, :3, 3]
    t2 = p2[:, :3, 3]
    return np.sqrt(np.sum(np.square(t1 - t2), axis=-1))


def cluster_grasps(grasps, num_clusters=32):
    ratio_of_grasps_to_be_used = 1.0
    cluster_indexes = np.asarray(
        farthest_points(
            grasps,
            num_clusters,
            distance_by_translation_grasp))
    output_grasps = []

    for i in range(num_clusters):
        indexes = np.where(cluster_indexes == i)[0]
        if ratio_of_grasps_to_be_used < 1:
            num_grasps_to_choose = max(
                1, int(ratio_of_grasps_to_be_used * float(len(indexes))))
            if len(indexes) == 0:
                raise ValueError('Error in clustering grasps')
            indexes = np.random.choice(
                indexes, size=num_grasps_to_choose, replace=False)

        output_grasps.append(grasps[indexes, :, :])

    output_grasps = np.asarray(output_grasps)

    return output_grasps


def sample_grasp_indexes(n, grasps):
    """
        Stratified sampling of the graps.
    """
    nonzero_rows = [i for i in range(len(grasps)) if len(grasps[i]) > 0]
    num_clusters = len(nonzero_rows)
    replace = n > num_clusters
    assert num_clusters != 0

    grasp_rows = np.random.choice(
        range(num_clusters),
        size=n,
        replace=replace).astype(
        np.int32)
    grasp_rows = [nonzero_rows[i] for i in grasp_rows]
    grasp_cols = []
    for grasp_row in grasp_rows:
        if len(grasps[grasp_rows]) == 0:
            raise ValueError('grasps cannot be empty')

        grasp_cols.append(np.random.randint(len(grasps[grasp_row])))

    grasp_cols = np.asarray(grasp_cols, dtype=np.int32)

    return np.vstack((grasp_rows, grasp_cols)).T


def distance_by_translation_point(p1, p2):
    """
      Gets two nx3 points and computes the disntace between point p1 and p2.
    """
    return np.sqrt(np.sum(np.square(p1 - p2), axis=-1))


def farthest_points(
        data,
        nclusters,
        dist_func,
        return_center_indexes=False,
        return_distances=False,
        verbose=False):
    """
      Performs farthest point sampling on data points.
      Args:
        data: numpy array of the data points.
        nclusters: int, number of clusters.
        dist_dunc: distance function that is used to compare two data points.
        return_center_indexes: bool, If True, returns the indexes of the center of
          clusters.
        return_distances: bool, If True, return distances of each point from centers.

      Returns clusters, [centers, distances]:
        clusters: numpy array containing the cluster index for each element in
          data.
        centers: numpy array containing the integer index of each center.
        distances: numpy array of [npoints] that contains the closest distance of
          each point to any of the cluster centers.
    """
    if nclusters >= data.shape[0]:
        if return_center_indexes:
            return np.arange(
                data.shape[0], dtype=np.int32), np.arange(
                data.shape[0], dtype=np.int32)

        return np.arange(data.shape[0], dtype=np.int32)

    clusters = np.ones((data.shape[0],), dtype=np.int32) * -1
    distances = np.ones((data.shape[0],), dtype=np.float32) * 1e7
    centers = []
    for iter in range(nclusters):
        index = np.argmax(distances)
        centers.append(index)
        shape = list(data.shape)
        for i in range(1, len(shape)):
            shape[i] = 1

        broadcasted_data = np.tile(np.expand_dims(data[index], 0), shape)
        new_distances = dist_func(broadcasted_data, data)
        distances = np.minimum(distances, new_distances)
        clusters[distances == new_distances] = iter
        if verbose:
            print(
                'farthest points max distance : {}'.format(
                    np.max(distances)))

    if return_center_indexes:
        if return_distances:
            return clusters, np.asarray(centers, dtype=np.int32), distances
        return clusters, np.asarray(centers, dtype=np.int32)

    return clusters


def regularize_pc_point_count(pc, npoints, use_farthest_point=False):
    """
      If point cloud pc has less points than npoints, it oversamples.
      Otherwise, it downsample the input pc to have npoint points.
      use_farthest_point: indicates whether to use farthest point sampling
      to downsample the points. Farthest point sampling version runs slower.
    """
    if pc.shape[0] > npoints:
        if use_farthest_point:
            _, center_indexes = farthest_points(
                pc, npoints, distance_by_translation_point, return_center_indexes=True)
        else:
            center_indexes = np.random.choice(
                range(pc.shape[0]), size=npoints, replace=False)
        pc = pc[center_indexes, :]
    else:
        required = npoints - pc.shape[0]
        if required > 0:
            index = np.random.choice(range(pc.shape[0]), size=required)
            pc = np.concatenate((pc, pc[index, :]), axis=0)
    return pc


def farthest_grasps(grasps, num_clusters=32, num_grasps=64):
    """ Returns grasps sampled with farthest point sampling """
    grasps_fps = cluster_grasps(grasps, num_clusters=num_clusters)
    clusters_fps = sample_grasp_indexes(num_grasps, grasps_fps)
    grasps = np.array([grasps_fps[cluster[0]][cluster[1]]
                       for cluster in clusters_fps])
    return grasps
