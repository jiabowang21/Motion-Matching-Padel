import os
import numpy as np
from math import cos, radians
from sklearn.neighbors import KDTree
from pymotion.io.bvh import BVH
from pymotion.ops.skeleton import fk
import matplotlib.pyplot as plt  # para la gráfica 2D

# =========================
# Parameters
# =========================
STEP = 10
ANGLE_THRESHOLD_DEG = 30
ANGLE_THRESHOLD_COS = cos(radians(ANGLE_THRESHOLD_DEG))
MIN_DISPLACEMENT = 1e-3
SEGMENT_LENGTH = 50  # frames para comprobar
FPS = 20  # no usado, pero conservado

# =========================
# Utilities
# =========================
def cosine_between(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def load_root_positions_from_bvh(bvh_path):
    bvh = BVH()
    bvh.load(bvh_path)
    _, local_positions, _, _, _, _ = bvh.get_data()
    return np.array(local_positions[:, 0, :], dtype=float)  # root joint

# =========================
# Build KD-tree dataset
# =========================
def build_motion_dataset_from_bvhs(bvh_files):
    entries = []
    for file_id, bvh_file in enumerate(bvh_files):
        global_positions = load_root_positions_from_bvh(bvh_file)
        num_frames = len(global_positions)
        if num_frames < 2:
            continue

        root_pos = global_positions[0]
        prev_disp = None
        offset = np.array([0.0, 0.0])
        current_start_frame = 0

        for f in range(STEP, num_frames, STEP):
            new_root_pos = global_positions[f]
            disp = np.array([new_root_pos[0] - root_pos[0], new_root_pos[2] - root_pos[2]])
            root_pos = new_root_pos
            disp_len = np.linalg.norm(disp)

            if disp_len < MIN_DISPLACEMENT:
                current_start_frame = f
                offset[:] = 0.0
                prev_disp = None
                continue

            if prev_disp is not None:
                cos_angle = cosine_between(disp, prev_disp)
                if cos_angle < ANGLE_THRESHOLD_COS:
                    current_start_frame = f
                    offset[:] = 0.0
                    prev_disp = None
                    continue

            offset += disp
            time = f - current_start_frame
            entries.append([offset[0], offset[1], time, file_id, current_start_frame])
            prev_disp = disp

    X = np.array([[e[0], e[1], e[2]] for e in entries])
    metadata = [(e[3], e[4]) for e in entries]
    return X, metadata

# =========================
# KD-tree wrapper
# =========================
class MotionKDTree:
    def __init__(self, X, metadata):
        self.tree = KDTree(X)
        self.metadata = metadata

    def query(self, dx, dz, t, k=1):
        query_point = np.array([[dx, dz, t]])
        dist, idx = self.tree.query(query_point, k=k)
        results = []
        for i, d in zip(idx[0], dist[0]):
            file_id, start_frame = self.metadata[i]
            results.append({"file_id": file_id, "start_frame": start_frame, "distance": d})
        return results

# =========================
# Prepare segment data
# =========================
def get_segment_data(bvh_path, start_frame, segment_length=SEGMENT_LENGTH):
    bvh_obj = BVH()
    bvh_obj.load(bvh_path)
    local_rot, local_pos, parents, offsets, _, _ = bvh_obj.get_data()
    end_frame = min(start_frame + segment_length, local_pos.shape[0])
    segment_rot = local_rot[start_frame:end_frame]
    segment_pos = local_pos[start_frame:end_frame]
    global_pos, _ = fk(segment_rot, segment_pos[:, 0, :], offsets, parents)
    return global_pos, parents

# =========================
# Main execution
# =========================
if __name__ == "__main__":
    # Cargar BVH
    bvh_folder = r"C:\Users\jiabo\Desktop\TFM\BBDD-professional\BVH processed"
    bvh_files = [os.path.join(bvh_folder, f) for f in os.listdir(bvh_folder) if f.endswith(".bvh")]
    print(f"Found {len(bvh_files)} BVH files")

    # Construir KD-tree
    X, metadata = build_motion_dataset_from_bvhs(bvh_files)
    motion_tree = MotionKDTree(X, metadata)
    print(f"KD-tree built with {len(X)} points")

    # Ejemplo de query
    dx, dz, t = 60, 70, 40  # Ajusta según escala de BVH
    results = motion_tree.query(dx, dz, t, k=1)
    best = results[0]
    file_name = bvh_files[best['file_id']]
    start_frame = best['start_frame']
    print(f"Best match: {os.path.basename(file_name)}, start_frame: {start_frame}, distance: {best['distance']:.4f}")

    # Preparar segmento
    global_pos, parents = get_segment_data(file_name, start_frame, segment_length=SEGMENT_LENGTH)

    # =========================
    # Comprobación de desplazamiento
    # =========================
    root_start = global_pos[0, 0, :]
    frame_index = min(t, global_pos.shape[0]-1)
    root_t = global_pos[frame_index, 0, :]
    dx_real = root_t[0] - root_start[0]
    dz_real = root_t[2] - root_start[2]

    print(f"Root position at start_frame ({start_frame}): {root_start}")
    print(f"Root position at start_frame + t ({start_frame + frame_index}): {root_t}")
    print(f"Real displacement after t frames: dx = {dx_real:.3f}, dz = {dz_real:.3f}")

    # =========================
    # Gráfica 2D X-Z
    # =========================
    x_traj = global_pos[:, 0, 0]
    z_traj = global_pos[:, 0, 2]

    plt.figure(figsize=(6,6))
    plt.plot(x_traj, z_traj, '-o', markersize=3, label="Trajectory")
    plt.scatter(root_start[0], root_start[2], color='green', s=80, label="Start")
    plt.scatter(root_t[0], root_t[2], color='red', s=80, label=f"t={t} frames")
    plt.xlabel("X position")
    plt.ylabel("Z position")
    plt.title(f"Root trajectory segment: {os.path.basename(file_name)}")
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()
