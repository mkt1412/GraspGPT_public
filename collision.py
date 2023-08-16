import numpy as np
import trimesh
import open3d
from visualize_data import draw_scene, get_gripper_collision_geometry
from copy import deepcopy as copy


def get_gripper_finger_sweep_volume_mayavi(grasp):
    """
    This is just for the sawyer gripper
    """
    align = tra.euler_matrix(np.pi / 2, 0, 0)
    extents = [0.06, 0.02, 0.14]
    transform = np.eye(4)
    transform = np.matmul(align, transform)
    transform = np.matmul(grasp, transform)
    finger_sweep_volume = trimesh.primitives.Box(
        extents=extents, transform=transform)

    return finger_sweep_volume, extents, transform


def get_gripper_finger_sweep_volume(grasp):
    """
    This is just for the sawyer gripper
    """
    align = tra.euler_matrix(np.pi / 2, 0, 0)
    extents = [0.06, 0.02, 0.14]
    transform = np.eye(4)
    transform[0, 3] = -extents[0] / 2
    transform[1, 3] = -extents[1] / 2
    transform[2, 3] = -extents[2] / 2
    transform = np.matmul(align, transform)
    transform = np.matmul(grasp, transform)
    finger_sweep_volume = trimesh.primitives.Box(
        extents=extents, transform=transform)

    return finger_sweep_volume, extents, transform


def get_gripper_collision_geometry(grasp):
    """
    This is an approximation of the sawyer mesh
    """
    meshes = []

    align = tra.euler_matrix(np.pi / 2, 0, 0)
    extents = [0.02, 0.02, 0.14]
    transform = np.eye(4)
    transform[0, 3] = -extents[0]
    transform = np.matmul(align, transform)
    transform = np.matmul(grasp, transform)
    part1 = trimesh.primitives.Box(extents=extents, transform=transform)
    meshes.append((part1, extents, transform))

    extents = [0.06, 0.02, 0.02]
    transform = np.eye(4)
    transform[2, 3] = 0.07
    transform = np.matmul(align, transform)
    transform = np.matmul(grasp, transform)
    part2 = trimesh.primitives.Box(extents=extents, transform=transform)
    meshes.append((part2, extents, transform))

    extents = [0.06, 0.02, 0.02]
    transform = np.eye(4)
    transform[2, 3] = -0.07
    transform = np.matmul(align, transform)
    transform = np.matmul(grasp, transform)
    part3 = trimesh.primitives.Box(extents=extents, transform=transform)
    meshes.append((part3, extents, transform))

    extents = [0.07, 0.02, 0.02]
    transform = np.eye(4)
    transform[0, 3] = -extents[0]
    transform = np.matmul(align, transform)
    transform = np.matmul(grasp, transform)
    part4 = trimesh.primitives.Box(extents=extents, transform=transform)
    meshes.append((part4, extents, transform))

    return meshes


class CollisionManager(object):
    def __init__(self, voxel_size=0.005):
        self._manager = trimesh.collision.CollisionManager()
        self._collision_objects = []
        self._voxel_size = voxel_size
        self._voxel_extents = [
            self._voxel_size,
            self._voxel_size,
            self._voxel_size]
        self._pc = None

    def construct_occupancy_grid(self, pc, max_points=1000):
        """
        pc: (N, 3) array
        Assume point cloud is already mean centered
        """

        if len(self._collision_objects) > 0:
            raise ValueError('Occupancy grid already constucted')

        self._pc = copy(pc)
        n_points = self._pc.shape[0]

        if n_points > max_points:
            # TODO Do farthest point sampling instead of random sampling
            chosen_idx = np.random.choice(
                list(range(n_points)), max_points, replace=False)
            self._pc = self._pc[chosen_idx, :]
            n_points = self._pc.shape[0]

        # Construct collision objects
        for i in range(n_points):
            extents = self._voxel_extents
            transform = np.eye(4)
            transform[:3, 3] = self._pc[i, :3]
            voxel = trimesh.primitives.Box(
                extents=extents, transform=transform)
            self._collision_objects.append((voxel, extents, transform))

        # Add to collision manager
        for i, (voxel, _, _) in enumerate(self._collision_objects):
            self._manager.add_object("voxel_{}".format(i), voxel)

    def check_collisions(self, grasps):
        """
        grasps: (N, 4, 4)
        returns:
            (N, 1) True if colliding, False otherwise
        """
        result = []
        for grasp in grasps:
            gripper_mesh = get_gripper_collision_geometry(grasp)
            is_collision = np.array([self.check_collision_manager(
                elem[0]) for elem in gripper_mesh]).sum() > 0
            result.append(is_collision)
        return np.array(result)

    def check_free_space_grasp(self, grasps):
        """
        grasps: (N, 4, 4)
        returns:
            (N, 1) True if grasping free space, false otherwise
        """
        result = []
        for grasp in grasps:
            finger_sweep_volume, _, _ = get_gripper_finger_sweep_volume_mayavi(
                grasp)
            is_collision = self.check_collision_manager(finger_sweep_volume)
            result.append(is_collision)
        return np.array(result)

    def check_collision_manager(self, mesh):
        return self._manager.in_collision_single(mesh)

    def visualize_occupancy_grid(
            self,
            grasp=None,
            debug_mode=False):
        """
        grasp: (4,4)
        Assume grasp is pc mean centered
        Only one grasp at a time
        """

        meshes = copy(self._collision_objects)
        grasps = []
        if grasp is not None:
            grasps.append(grasp)
            finger_sweep_volume, extents, transform = get_gripper_finger_sweep_volume(
                grasp)
            meshes.append((finger_sweep_volume, extents, transform))

        draw_scene(
            pc=self._pc,
            grasps=grasps,
            meshes=meshes,
            subtract_pc_mean=False,
            debug_mode=debug_mode)
