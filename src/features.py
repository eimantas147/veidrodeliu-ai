"""
Duomenu valymas ir pozymiai: naivus ir fizikiskai pataisytas geometrinis ivertis.
Sie moduliai naudojami ir treniruojant, ir demonstracineje programoje.
"""
import numpy as np
import pandas as pd

from geometry import CAM_POS, K1, K2, R_CAM, TARGETS, mirror_angles

FEATURES = ["u_n", "v_n", "ipd_n", "face_w_n", "face_h_n", "yaw_est", "pitch_est", "roll_est"]
GEO_NAIVE = ["gn_" + t for t in TARGETS]
GEO_PHYS = ["gp_" + t for t in TARGETS]
MEAN_IPD = 0.063
NECK_OFFSET = np.array([-0.09, 0.0, -0.07])     # kaklo sukimosi taskas akiu atzvilgiu (vidutinis)


def clean(df, log=None):
    """Pasalina kadrus be veido ir fiziskai negalimus aptikimus (IPD / veido plotis)."""
    log = {} if log is None else log
    n0 = len(df)
    df = df.dropna(subset=FEATURES)
    log["nerasti_veidai"] = n0 - len(df)
    ok = (df["ipd_n"] / df["face_w_n"]).between(0.33, 0.65)
    log["klaidingi_aptikimai"] = int((~ok).sum())
    df = df[ok].reset_index(drop=True)
    log["liko_kadru"] = len(df)
    return df


def _cam_to_car(xn, yn, depth):
    p = np.stack([xn * depth, yn * depth, depth], axis=-1)
    return p @ R_CAM + CAM_POS


def _undistort(xd, yd, iters=10):
    x, y = xd.copy(), yd.copy()
    for _ in range(iters):
        r2 = x * x + y * y
        k = 1 + K1 * r2 + K2 * r2 * r2
        x, y = xd / k, yd / k
    return x, y


def _rot(yaw, pitch, roll):
    """Vektorizuota galvos pasukimo matrica (N, 3, 3)."""
    y, p, r = (np.radians(a) for a in (yaw, pitch, roll))
    cy, sy, cp, sp, cr, sr = np.cos(y), np.sin(y), np.cos(p), np.sin(p), np.cos(r), np.sin(r)
    o, z = np.ones_like(y), np.zeros_like(y)
    Rz = np.stack([np.stack([cy, -sy, z], -1), np.stack([sy, cy, z], -1), np.stack([z, z, o], -1)], -2)
    Ry = np.stack([np.stack([cp, z, sp], -1), np.stack([z, o, z], -1), np.stack([-sp, z, cp], -1)], -2)
    Rx = np.stack([np.stack([o, z, z], -1), np.stack([z, cr, -sr], -1), np.stack([z, sr, cr], -1)], -2)
    return Rz @ Ry @ Rx


def eye_naive(df):
    """Klasikinis metodas: kameros modelis + vidutinis IPD (be iskraipymu ir galvos pasukimo korekcijos)."""
    depth = MEAN_IPD / df["ipd_n"].to_numpy()
    return _cam_to_car(df["u_n"].to_numpy(), df["v_n"].to_numpy(), depth)


def eye_physics(df):
    """Fizikiskai pataisytas: objektyvo iskraipymu pasalinimas, IPD sutrumpejimas pasukus galva,
    akiu poslinkis sukantis apie kakla."""
    yaw, pitch, roll = (df[c].to_numpy() for c in ("yaw_est", "pitch_est", "roll_est"))
    xn, yn = _undistort(df["u_n"].to_numpy(), df["v_n"].to_numpy())
    depth = MEAN_IPD * np.cos(np.radians(yaw)) / df["ipd_n"].to_numpy()
    eye_obs = _cam_to_car(xn, yn, depth)
    R = _rot(yaw, pitch, roll)
    return eye_obs - NECK_OFFSET + R @ NECK_OFFSET


def build_features(df):
    df = df.copy()
    df[GEO_NAIVE] = mirror_angles(eye_naive(df))
    df[GEO_PHYS] = mirror_angles(eye_physics(df))
    return df


def as_frame(feat_dict):
    """Vieno kadro pozymiai (dict) -> DataFrame su visais pozymiais (demonstracijai)."""
    return build_features(pd.DataFrame([feat_dict]))
