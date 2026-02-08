import os
import numpy as np
from math import cos, radians
from sklearn.neighbors import KDTree

# =========================
# Parameters
# =========================

STEP = 10                          # Process every N frames
ANGLE_THRESHOLD_DEG = 30           # Direction change threshold
ANGLE_THRESHOLD_COS = cos(radians(ANGLE_THRESHOLD_DEG))
MIN_DISPLACEMENT = 1e-3            # Ignore tiny movements
GROUND_AXES = (0, 2)               # x, z

# =========================
# Utilities
# =========================

def load_root_positions_from_csv(csv_path):
    """
    Expected CSV format (per frame):
    frame, root_x, root_y, root_z, ...
    """
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    return data[:, 1:4]  # (x, y, z)


def cosine_between(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


# =========================
# KD-tree Data Construction
# =========================

def build_motion_dataset(csv_files):
    """
    Returns:
        X        : (N, 3) array of KD-tree points (dx, dz, time)
        metadata : list of (file_id, start_frame)
    """
    entries = []

    for file_id, csv_path in enumerate(csv_files):

        global_positions = load_root_positions_from_csv(csv_path)
        num_frames = len(global_positions)

        root_pos = global_positions[0]
        prev_disp = None
        offset = np.array([0.0, 0.0])
        current_start_frame = 0

        for f in range(STEP, num_frames, STEP):

            new_root_pos = global_positions[f]
            disp_3d = new_root_pos - root_pos
            root_pos = new_root_pos

            # Project to ground plane (x, z)
            disp = np.array([disp_3d[GROUND_AXES[0]], disp_3d[GROUND_AXES[1]]])
            disp_len = np.linalg.norm(disp)

            # Reset if movement is too small
            if disp_len < MIN_DISPLACEMENT:
                current_start_frame = f
                offset[:] = 0.0
                prev_disp = None
                continue

            # Reset if direction changes too much
            if prev_disp is not None:
                cos_angle = cosine_between(disp, prev_disp)
                if cos_angle < ANGLE_THRESHOLD_COS:
                    current_start_frame = f
                    offset[:] = 0.0
                    prev_disp = None
                    continue

            # Accumulate displacement
            offset += disp
            time = f - current_start_frame

            # Store KD-tree entry
            entries.append([
                offset[0],              # displacement_x
                offset[1],              # displacement_z
                time,                   # elapsed time
                file_id,                # reference to CSV/BVH
                current_start_frame     # start frame
            ])

            prev_disp = disp

    # Split KD-tree data and metadata
    X = np.array([[e[0], e[1], e[2]] for e in entries])
    metadata = [(e[3], e[4]) for e in entries]

    return X, metadata


# =========================
# KD-tree Wrapper
# =========================

class MotionKDTree:
    def __init__(self, X, metadata):
        self.tree = KDTree(X)
        self.metadata = metadata

    def query(self, displacement_x, displacement_z, time, k=5):
        query_point = np.array([[displacement_x, displacement_z, time]])
        dist, idx = self.tree.query(query_point, k=k)

        results = []
        for i, d in zip(idx[0], dist[0]):
            file_id, start_frame = self.metadata[i]
            results.append({
                "file_id": file_id,
                "start_frame": start_frame,
                "distance": d
            })

        return results


# =========================
# Example Usage
# =========================

if __name__ == "__main__":

    # CSV files generated from pymotion (one per BVH)
    csv_dir = "motion_csv"
    csv_files = [
        os.path.join(csv_dir, f)
        for f in os.listdir(csv_dir)
        if f.endswith(".csv")
    ]

    # Build dataset
    X, metadata = build_motion_dataset(csv_files)

    print(f"KD-tree dataset size: {len(X)} points")

    # Build KD-tree
    motion_tree = MotionKDTree(X, metadata)

    # Example query
    dx, dz, t = 1.2, 0.5, 40
    results = motion_tree.query(dx, dz, t, k=3)

    for r in results:
        print(r)
