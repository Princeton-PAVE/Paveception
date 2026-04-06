from orientationfit import process_chair_points

chair_orientations = process_chair_points(
    points_3d=clustered_chair_points,
    dbscan_eps=0.05,
    dbscan_min_samples=10,
    delta_thresh=np.deg2rad(20)
)

for cluster_id, orientation_data in chair_orientations.items():
    print(f"Chair {cluster_id}: {orientation_data['theta_deg']:.1f}°")
    print(f"  L-shaped: {orientation_data['is_l_shaped']}")
    print(f"  Centroid: {orientation_data['centroid_3d']}")