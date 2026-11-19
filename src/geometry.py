"""
Bendra geometrija: automobilio koordinates, kamera, veidrodeliai, atspindzio desnis.

Automobilio koordinates (m), atskaitos taskas = vidutinio vairuotojo akiu vidurys:
  x - i prieki, y - i kaire, z - i virsu.
"""
import numpy as np

# ---------------- Veidrodeliai ----------------
# Veidrodelio centro padetis ir norima matymo kryptis (atgal, siek tiek i isore - BGE metodas)
MIRRORS = {
    "L": {"pos": np.array([0.80, 0.50, -0.10]), "view": np.array([-1.0, 0.12, 0.0])},
    "R": {"pos": np.array([0.85, -1.30, -0.10]), "view": np.array([-1.0, -0.12, 0.0])},
}
TARGETS = ["L_yaw", "L_pitch", "R_yaw", "R_pitch"]

# ---------------- Salono kamera (vairuotojo stebejimo kamera ant vairo kolonelės) ----------------
CAM_POS = np.array([0.65, -0.02, -0.12])
IMG_W, IMG_H = 1280, 800
FOCAL_PX = 560.0          # ~ 98 laipsniu horizontalus matymo laukas
CX, CY = IMG_W / 2, IMG_H / 2
K1, K2 = -0.28, 0.07      # plataus kampo objektyvo radialiniai iskraipymai


def _camera_rotation():
    """Kameros asys automobilio koordinatemis: kamera nukreipta i vidutinio vairuotojo akis."""
    z_c = -CAM_POS / np.linalg.norm(CAM_POS)            # optine asis
    x_c = np.cross(z_c, np.array([0.0, 0.0, 1.0]))      # vaizdo "desine"
    x_c /= np.linalg.norm(x_c)
    y_c = np.cross(z_c, x_c)                            # vaizdo "zemyn"
    return np.vstack([x_c, y_c, z_c])


R_CAM = _camera_rotation()


def project(points, distort=True):
    """3D taskai automobilio koordinatemis -> pikseliai (u, v). points: (..., 3)."""
    p = (points - CAM_POS) @ R_CAM.T
    xn, yn = p[..., 0] / p[..., 2], p[..., 1] / p[..., 2]
    if distort:
        r2 = xn ** 2 + yn ** 2
        k = 1 + K1 * r2 + K2 * r2 ** 2
        xn, yn = xn * k, yn * k
    return np.stack([FOCAL_PX * xn + CX, FOCAL_PX * yn + CY], axis=-1)


def backproject(u, v, depth):
    """Pikseliai + gylis kameros asimi -> 3D taskas automobilio koordinatemis (be iskraipymu korekcijos)."""
    xn, yn = (u - CX) / FOCAL_PX, (v - CY) / FOCAL_PX
    p_cam = np.stack([xn * depth, yn * depth, depth], axis=-1)
    return p_cam @ R_CAM + CAM_POS


def mirror_angles(eye):
    """Atspindzio desnis. eye: (N, 3). Grazina (N, 4): L_yaw, L_pitch, R_yaw, R_pitch laipsniais."""
    eye = np.atleast_2d(eye)
    out = []
    for key in ("L", "R"):
        m = MIRRORS[key]
        e = eye - m["pos"]
        e /= np.linalg.norm(e, axis=1, keepdims=True)
        t = m["view"] / np.linalg.norm(m["view"])
        n = e + t
        n /= np.linalg.norm(n, axis=1, keepdims=True)
        out.append(np.degrees(np.arctan2(n[:, 1], -n[:, 0])))
        out.append(np.degrees(np.arcsin(n[:, 2])))
    return np.stack(out, axis=1)


def rot_matrix(yaw, pitch, roll):
    """Galvos pasukimo matrica (laipsniai). yaw - apie z, pitch - apie y, roll - apie x."""
    y, p, r = np.radians(yaw), np.radians(pitch), np.radians(roll)
    cy, sy, cp, sp, cr, sr = np.cos(y), np.sin(y), np.cos(p), np.sin(p), np.cos(r), np.sin(r)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx
