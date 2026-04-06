import os
import numpy as np
from math import cos, radians
from sklearn.neighbors import KDTree
from pymotion.io.bvh import BVH
from pymotion.ops.skeleton import fk
import matplotlib.pyplot as plt  # Para la gráfica 2D


# ==========================================================
# PARÁMETROS GLOBALES
# ==========================================================

STEP = 10  
# Cada cuántos frames se calcula el desplazamiento.
# Reduce ruido y tamaño del dataset.

ANGLE_THRESHOLD_DEG = 30  
# Umbral máximo de cambio de dirección permitido (en grados)

ANGLE_THRESHOLD_COS = cos(radians(ANGLE_THRESHOLD_DEG))  
# Convertimos el umbral angular a coseno para comparación directa

MIN_DISPLACEMENT = 1e-3  
# Umbral mínimo para considerar que existe movimiento

SEGMENT_LENGTH = 50  
# Longitud máxima del segmento que se extraerá para validación

FPS = 20  
# Frames por segundo (no usado directamente aquí)


# ==========================================================
# FUNCIONES AUXILIARES
# ==========================================================

def cosine_between(v1, v2):
    """
    Calcula el coseno del ángulo entre dos vectores 2D.

    Parámetros:
        v1, v2: vectores numpy

    Retorna:
        float: coseno del ángulo entre ambos vectores

    Se utiliza para detectar cambios bruscos de dirección.
    """
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def load_root_positions_from_bvh(bvh_path):
    """
    Carga un archivo BVH y devuelve las posiciones del root joint
    para todos los frames.

    Parámetros:
        bvh_path (str): ruta al archivo BVH

    Retorna:
        np.array de tamaño [num_frames, 3]
        Contiene posiciones (x,y,z) del root.
    """
    bvh = BVH()
    bvh.load(bvh_path)

    # get_data devuelve:
    # local_rot, local_pos, parents, offsets, _, _
    _, local_positions, _, _, _, _ = bvh.get_data()

    # Tomamos el joint 0 (root)
    return np.array(local_positions[:, 0, :], dtype=float)


# ==========================================================
# CONSTRUCCIÓN DEL DATASET PARA KD-TREE
# ==========================================================

def build_motion_dataset_from_bvhs(bvh_files):
    """
    Construye el dataset de motion matching a partir de múltiples BVHs.

    Para cada BVH:
        - Recorre los frames cada STEP
        - Calcula desplazamientos acumulados en el plano XZ
        - Descarta segmentos con poco movimiento
        - Descarta cambios bruscos de dirección
        - Guarda entradas tipo (dx, dz, tiempo)

    Parámetros:
        bvh_files (list): lista de rutas BVH

    Retorna:
        X (np.array): matriz de características [dx, dz, tiempo]
        metadata (list): [(file_id, start_frame), ...]
    """

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

        # Recorremos cada STEP frames
        for f in range(STEP, num_frames, STEP):

            new_root_pos = global_positions[f]

            # Desplazamiento en plano XZ
            disp = np.array([
                new_root_pos[0] - root_pos[0],
                new_root_pos[2] - root_pos[2]
            ])

            root_pos = new_root_pos
            disp_len = np.linalg.norm(disp)

            # Si no hay movimiento suficiente → reiniciar segmento
            if disp_len < MIN_DISPLACEMENT:
                current_start_frame = f
                offset[:] = 0.0
                prev_disp = None
                continue

            # Si hay cambio brusco de dirección → cortar segmento
            if prev_disp is not None:
                cos_angle = cosine_between(disp, prev_disp)
                if cos_angle < ANGLE_THRESHOLD_COS:
                    current_start_frame = f
                    offset[:] = 0.0
                    prev_disp = None
                    continue

            # Acumular desplazamiento
            offset += disp
            time = f - current_start_frame

            entries.append([
                offset[0],
                offset[1],
                time,
                file_id,
                current_start_frame
            ])

            prev_disp = disp

    # Construcción del dataset numérico
    X = np.array([[e[0], e[1], e[2]] for e in entries])

    # Información auxiliar
    metadata = [(e[3], e[4]) for e in entries]

    return X, metadata


# ==========================================================
# CLASE KD-TREE PARA MOTION MATCHING
# ==========================================================

class MotionKDTree:
    """
    Encapsula la estructura KD-tree para búsqueda eficiente
    en el espacio (dx, dz, tiempo).
    """

    def __init__(self, X, metadata):
        """
        Construye el árbol KD.

        Parámetros:
            X: matriz de características
            metadata: lista con información adicional
        """
        self.tree = KDTree(X)
        self.metadata = metadata

    def query(self, dx, dz, t, k=1):
        """
        Realiza una búsqueda del segmento más similar
        al desplazamiento deseado.

        Parámetros:
            dx, dz: desplazamiento deseado
            t: tiempo deseado
            k: número de vecinos más cercanos

        Retorna:
            Lista de resultados con file_id, start_frame y distancia.
        """
        query_point = np.array([[dx, dz, t]])

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


# ==========================================================
# EXTRACCIÓN DEL SEGMENTO SELECCIONADO
# ==========================================================

def get_segment_data(bvh_path, start_frame, segment_length=SEGMENT_LENGTH):
    """
    Extrae un segmento de animación y calcula posiciones globales.

    Parámetros:
        bvh_path (str): ruta BVH
        start_frame (int): frame inicial
        segment_length (int): longitud máxima del segmento

    Retorna:
        global_pos: posiciones globales [frames, joints, 3]
        parents: jerarquía del esqueleto
    """

    bvh_obj = BVH()
    bvh_obj.load(bvh_path)

    local_rot, local_pos, parents, offsets, _, _ = bvh_obj.get_data()

    end_frame = min(start_frame + segment_length, local_pos.shape[0])

    segment_rot = local_rot[start_frame:end_frame]
    segment_pos = local_pos[start_frame:end_frame]

    # Forward Kinematics → convierte posiciones locales a globales
    global_pos, _ = fk(
        segment_rot,
        segment_pos[:, 0, :],
        offsets,
        parents
    )

    return global_pos, parents


# ==========================================================
# BLOQUE PRINCIPAL
# ==========================================================

if __name__ == "__main__":

    # Ruta donde están los BVHs
    bvh_folder = r"C:\Users\jiabo\Desktop\TFM\BBDD-professional\BVH processed"

    bvh_files = [
        os.path.join(bvh_folder, f)
        for f in os.listdir(bvh_folder)
        if f.endswith(".bvh")
    ]

    print(f"Found {len(bvh_files)} BVH files")

    # Construcción del dataset
    X, metadata = build_motion_dataset_from_bvhs(bvh_files)

    motion_tree = MotionKDTree(X, metadata)

    print(f"KD-tree built with {len(X)} points")

    # ------------------------------------------------------
    # QUERY DE EJEMPLO
    # ------------------------------------------------------

    dx, dz, t = 60, 70, 40

    results = motion_tree.query(dx, dz, t, k=1)

    best = results[0]

    file_name = bvh_files[best['file_id']]
    start_frame = best['start_frame']

    print(f"Best match: {os.path.basename(file_name)}")
    print(f"Start frame: {start_frame}")
    print(f"Distance in feature space: {best['distance']:.4f}")

    # ------------------------------------------------------
    # VALIDACIÓN DEL DESPLAZAMIENTO REAL
    # ------------------------------------------------------

    global_pos, parents = get_segment_data(
        file_name,
        start_frame,
        segment_length=SEGMENT_LENGTH
    )

    root_start = global_pos[0, 0, :]
    frame_index = min(t, global_pos.shape[0] - 1)
    root_t = global_pos[frame_index, 0, :]

    dx_real = root_t[0] - root_start[0]
    dz_real = root_t[2] - root_start[2]

    print(f"Root position at start_frame: {root_start}")
    print(f"Root position at start_frame + t: {root_t}")
    print(f"Real displacement: dx = {dx_real:.3f}, dz = {dz_real:.3f}")

    # ------------------------------------------------------
    # GRÁFICA 2D X-Z
    # ------------------------------------------------------

    x_traj = global_pos[:, 0, 0]
    z_traj = global_pos[:, 0, 2]

    plt.figure(figsize=(6, 6))

    plt.plot(x_traj, z_traj, '-o', markersize=3, label="Trajectory")
    plt.scatter(root_start[0], root_start[2], s=80, label="Start")
    plt.scatter(root_t[0], root_t[2], s=80, label=f"t={t} frames")

    plt.xlabel("X position")
    plt.ylabel("Z position")
    plt.title(f"Root trajectory segment: {os.path.basename(file_name)}")

    plt.legend()
    plt.grid(True)
    plt.axis('equal')

    plt.show()
