import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import linear_sum_assignment

def find_shortest_path(dist_matrix):
    """
    Finds the shortest path that visits all views based on the distance matrix.

    Args:
    - dist_matrix: A square matrix of distances between each pair of views.

    Returns:
    - path: The order of views that gives the shortest path.
    """
    n = len(dist_matrix)
    row_ind, col_ind = linear_sum_assignment(dist_matrix)
    path = col_ind  # The optimal path order based on distances
    return path

def generate_random_poses(views, num_poses_max=100):
    """
    Generates interpolated poses along the shortest path between views in world coordinates, 
    then transforms to the pseudo camera's own coordinate system.

    Args:
    - views: A list of view objects, where each view has a rotation matrix (R) and translation vector (T).
    - num_poses: Total number of interpolated poses to generate.

    Returns:
    - interpolated_poses: A numpy array of shape (num_poses, 4, 4) containing the generated poses.
    """

    # Step 1: Convert each view's position to world coordinates and compute the distance matrix
    positions = []
    for view in views:
        tmp_view = np.eye(4)
        tmp_view[:3] = np.concatenate([view.R.T, view.T[:, None]], axis=1)
        camera_to_world = np.linalg.inv(tmp_view)
        positions.append(camera_to_world[:3, 3])
    
    positions = np.array(positions)
    dist_matrix = squareform(pdist(positions))

    # Step 2: Find the shortest path visiting all views
    path = find_shortest_path(dist_matrix)

    # Step 3: Sort views according to the shortest path
    sorted_views = [views[i] for i in path]
    sorted_positions = [positions[i] for i in path]

    # Step 4: Generate interpolated poses in world coordinates
    interpolated_poses = []

    for i in range(len(sorted_views) - 1):
        start_view = sorted_views[i]
        end_view = sorted_views[i + 1]

        start_R = start_view.R
        start_T = sorted_positions[i]

        end_R = end_view.R
        end_T = sorted_positions[i + 1]

        # Linear space from 0 to 1
        t_linear = np.linspace(0, 1, num=num_poses_max, endpoint=False)

        # Cosine adjustment to make the ends denser and middle sparser
        t_cosine = 0.5 * (1 - np.cos(np.pi * t_linear))

        for t in t_cosine:
            # Linear interpolation for position in world coordinates
            interp_T_world = (1 - t) * start_T + t * end_T

            # Spherical linear interpolation (slerp) for rotation
            omega = np.arccos(np.clip(np.dot(start_R.ravel(), end_R.ravel()), -1, 1))
            sin_omega = np.sin(omega)
            if sin_omega == 0:
                interp_R_world = start_R
            else:
                interp_R_world = (np.sin((1 - t) * omega) / sin_omega) * start_R + (np.sin(t * omega) / sin_omega) * end_R

            # Construct the interpolated pose matrix in world coordinates
            interp_pose_world = np.eye(4)
            interp_pose_world[:3, :3] = interp_R_world
            interp_pose_world[:3, 3] = interp_T_world

            # Convert to the pseudo camera's own coordinate system
            interp_pose_camera = np.linalg.inv(interp_pose_world)

            interpolated_poses.append(interp_pose_camera)

    return np.stack(interpolated_poses, axis=0)
