import numpy as np
from sklearn.linear_model import RANSACRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.cluster import DBSCAN
from typing import Tuple, Dict, List, Optional


# --- Helper: fit line using RANSAC ---
class LineModel(BaseEstimator, RegressorMixin):
    def fit(self, X, y):
        # X is x, y is y → fit y = ax + b
        A = np.hstack([X, np.ones((X.shape[0], 1))])
        self.params_, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        return self

    def predict(self, X):
        return X * self.params_[0] + self.params_[1]


def fit_line_ransac(points: np.ndarray, residual_threshold: float = 0.1) -> float:
    """
    Fit a line to 2D points using RANSAC.
    
    Args:
        points: Nx2 numpy array of 2D points
        residual_threshold: RANSAC residual threshold
        
    Returns:
        theta: orientation angle (radians)
    """
    if points.shape[0] < 2:
        return 0.0
        
    X = points[:, 0].reshape(-1, 1)
    y = points[:, 1]

    try:
        ransac = RANSACRegressor(LineModel(), residual_threshold=residual_threshold)
        ransac.fit(X, y)
        a = ransac.estimator_.params_[0]
        theta = np.arctan(a)
    except:
        # Fallback: fit with least squares
        A = np.hstack([X, np.ones((X.shape[0], 1))])
        params, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        theta = np.arctan(params[0])

    return theta


def project_3d_to_2d(points_3d: np.ndarray, plane: str = "xz") -> np.ndarray:
    """
    Project 3D points to 2D for bird's-eye view.
    
    Args:
        points_3d: Nx3 numpy array of 3D points [X, Y, Z]
        plane: which plane to project to ('xz' for bird's eye, 'xy' for top-down)
        
    Returns:
        Nx2 numpy array of 2D points
    """
    if plane == "xz":
        # Bird's eye view: X-Z plane
        return points_3d[:, [0, 2]]
    elif plane == "xy":
        # Top-down: X-Y plane
        return points_3d[:, [0, 1]]
    else:
        raise ValueError(f"Unknown plane: {plane}")


def cluster_3d_points(points_3d: np.ndarray, eps: float = 0.05, min_samples: int = 10) -> Dict[int, np.ndarray]:
    """
    Cluster 3D points using DBSCAN.
    
    Args:
        points_3d: Nx3 numpy array of 3D points
        eps: DBSCAN epsilon parameter
        min_samples: DBSCAN min_samples parameter
        
    Returns:
        Dictionary mapping cluster_id -> cluster_points
    """
    if points_3d.shape[0] < min_samples:
        return {0: points_3d}
    
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points_3d)
    labels = clustering.labels_
    
    clusters = {}
    for label in set(labels):
        if label >= 0:  # Ignore noise points (label == -1)
            clusters[label] = points_3d[labels == label]
    
    return clusters


def estimate_orientation_2d(cluster_points_2d: np.ndarray, delta_thresh: float = np.deg2rad(20)) -> Dict[str, float]:
    """
    Apply L-shaped algorithm to estimate orientation from 2D points.
    
    Algorithm:
    1. Find closest point D to origin
    2. Compute angles relative to D
    3. Split points into two groups based on median angle
    4. Fit lines to each group
    5. Detect L-shape if angle difference exceeds threshold
    6. Return dominant orientation
    
    Args:
        cluster_points_2d: Nx2 numpy array of 2D points
        delta_thresh: angle threshold for L-shape detection (radians)
        
    Returns:
        Dictionary with keys:
            - theta: primary orientation angle (radians)
            - theta_deg: primary orientation (degrees)
            - is_l_shaped: whether L-shape was detected
            - theta_A: orientation of group A
            - theta_B: orientation of group B
            - delta: angle difference between groups
            - num_points: total number of points processed
    """
    if cluster_points_2d.shape[0] < 5:
        return {
            "theta": 0.0,
            "theta_deg": 0.0,
            "is_l_shaped": False,
            "theta_A": 0.0,
            "theta_B": 0.0,
            "delta": 0.0,
            "num_points": cluster_points_2d.shape[0]
        }

    # Step 1: find closest point D to origin
    dists = np.linalg.norm(cluster_points_2d, axis=1)
    D = cluster_points_2d[np.argmin(dists)]

    # Step 2: compute angles relative to D
    rel = cluster_points_2d - D
    angles = np.arctan2(rel[:, 1], rel[:, 0])

    # Step 3: split into A and B based on median angle
    median_angle = np.median(angles)
    A = cluster_points_2d[angles <= median_angle]
    B = cluster_points_2d[angles > median_angle]

    # Edge case: small splits
    if len(A) < 5 or len(B) < 5:
        theta = fit_line_ransac(cluster_points_2d)
        return {
            "theta": theta,
            "theta_deg": np.rad2deg(theta),
            "is_l_shaped": False,
            "theta_A": theta,
            "theta_B": theta,
            "delta": 0.0,
            "num_points": cluster_points_2d.shape[0]
        }

    # Step 4: fit lines to each group
    theta_A = fit_line_ransac(A)
    theta_B = fit_line_ransac(B)

    # Step 5: compare angle difference
    delta = np.abs(theta_A - theta_B)
    delta = min(delta, np.pi - delta)  # normalize to [0, pi/2]

    is_l_shaped = delta > delta_thresh
    
    if is_l_shaped:
        # L-shape detected → choose dominant line (more points)
        theta = theta_A if len(A) > len(B) else theta_B
    else:
        # Single edge → fit all points
        theta = fit_line_ransac(cluster_points_2d)

    return {
        "theta": theta,
        "theta_deg": np.rad2deg(theta),
        "is_l_shaped": is_l_shaped,
        "theta_A": theta_A,
        "theta_B": theta_B,
        "delta": delta,
        "delta_deg": np.rad2deg(delta),
        "num_points": cluster_points_2d.shape[0],
        "len_A": len(A),
        "len_B": len(B)
    }


def process_chair_points(points_3d: np.ndarray, 
                         dbscan_eps: float = 0.05,
                         dbscan_min_samples: int = 10,
                         delta_thresh: float = np.deg2rad(20),
                         plane: str = "xz") -> Dict[int, Dict]:
    """
    Full pipeline to process 3D chair point cloud and estimate orientations.
    
    Args:
        points_3d: Nx3 numpy array of 3D points from render_video.py
        dbscan_eps: DBSCAN epsilon for clustering
        dbscan_min_samples: DBSCAN min_samples
        delta_thresh: L-shape detection threshold (radians)
        plane: projection plane ('xz' for bird's eye, 'xy' for top-down)
        
    Returns:
        Dictionary mapping cluster_id -> orientation results
    """
    if points_3d.shape[0] == 0:
        return {}
    
    # Cluster the 3D points
    clusters = cluster_3d_points(points_3d, eps=dbscan_eps, min_samples=dbscan_min_samples)
    
    results = {}
    for cluster_id, cluster_3d in clusters.items():
        # Project to 2D
        cluster_2d = project_3d_to_2d(cluster_3d, plane=plane)
        
        # Estimate orientation
        orientation = estimate_orientation_2d(cluster_2d, delta_thresh=delta_thresh)
        
        # Add cluster info
        orientation["cluster_id"] = cluster_id
        orientation["centroid_3d"] = np.mean(cluster_3d, axis=0)
        orientation["centroid_2d"] = np.mean(cluster_2d, axis=0)
        
        results[cluster_id] = orientation
    
    return results


# --- Backward compatibility ---
def estimate_orientation(cluster_points: np.ndarray, delta_thresh: float = np.deg2rad(20)) -> float:
    """
    Legacy function for backward compatibility.
    
    Args:
        cluster_points: Nx2 numpy array of 2D points
        delta_thresh: angle threshold for L-shape detection
        
    Returns:
        theta: primary orientation angle (radians)
    """
    result = estimate_orientation_2d(cluster_points, delta_thresh=delta_thresh)
    return result["theta"]