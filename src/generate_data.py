"""
Sintetinio duomenu rinkinio generatorius.

Kiekvienas virtualus vairuotojas turi: ugi, tarpvyzdini atstuma (IPD), veido ploti,
sedejimo padeti ir asmeninius veidrodeliu pageidavimus. Kiekvienam vairuotojui
sugeneruojama N kadru su laikysenos svyravimais, galvos pasukimais ir kameros triuksmu.

Etalonas (teisingas atsakymas) - veidrodeliu kampai, apskaiciuoti pagal atspindzio
desni is TIKROSIOS akiu padeties (kai galva nepasukta).

Paleidimas:  python src/generate_data.py  ->  data/dataset.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from geometry import FOCAL_PX, CX, CY, TARGETS, mirror_angles, project, rot_matrix

from features import FEATURES  # noqa: E402  (vienas saltinis)


def make_drivers(n, rng):
    sex = rng.integers(0, 2, n)                                       # 0 - moteris, 1 - vyras
    height = np.where(sex == 1, rng.normal(1.80, 0.07, n), rng.normal(1.67, 0.065, n))
    ipd = np.clip(np.where(sex == 1, rng.normal(0.0645, 0.0035, n), rng.normal(0.0617, 0.0035, n)),
                  0.054, 0.074)
    face_w = ipd * 2.15 + rng.normal(0, 0.006, n)                     # skruostikauliu plotis
    face_h = rng.normal(0.19, 0.012, n)                               # kakta - smakras
    dh = height - 1.74
    seat = np.stack([-0.30 * dh + rng.normal(0, 0.03, n),             # aukstesni sedi toliau
                     rng.normal(0, 0.015, n),
                     0.47 * dh + rng.normal(0, 0.025, n)], axis=1)    # sedimasis ugis
    pref = np.column_stack([rng.normal(0, 0.8, n), rng.normal(0, 0.5, n),
                            rng.normal(0, 0.8, n), rng.normal(0, 0.5, n)])
    return pd.DataFrame({"driver_id": np.arange(n), "sex": sex, "height_m": height, "ipd_m": ipd,
                         "face_w_m": face_w, "face_h_m": face_h,
                         "seat_x": seat[:, 0], "seat_y": seat[:, 1], "seat_z": seat[:, 2],
                         **{f"pref_{t}": pref[:, i] for i, t in enumerate(TARGETS)}})


def make_frames(drv, n_frames, rng):
    rows = []
    for d in drv.itertuples():
        for _ in range(n_frames):
            eye = np.array([d.seat_x, d.seat_y, d.seat_z]) + rng.normal(0, 0.012, 3)   # laikysena
            if rng.random() < 0.3:                                    # zvilgsnis i veidrodeli / saliai
                yaw = rng.uniform(-35, 35)
            else:
                yaw = rng.normal(0, 6)
            pitch, roll = rng.normal(0, 5), rng.normal(0, 3)
            R = rot_matrix(yaw, pitch, roll)
            pivot = eye + np.array([-0.09, 0, -0.07])                  # kaklo sukimosi taskas
            local = np.array([[0, d.ipd_m / 2, 0], [0, -d.ipd_m / 2, 0],          # akys
                              [-0.03, d.face_w_m / 2, -0.04], [-0.03, -d.face_w_m / 2, -0.04],  # skruostai
                              [0.01, 0, 0.07], [0.02, 0, 0.07 - d.face_h_m]])      # kakta, smakras
            pts = pivot + (eye + local - pivot) @ R.T
            px = project(pts)
            sigma = 5.0 if rng.random() < 0.04 else 1.2               # 4 % kadru - akiniu atspindziai
            px += rng.normal(0, sigma, px.shape)
            mid = (px[0] + px[1]) / 2
            f = {
                "u_n": (mid[0] - CX) / FOCAL_PX, "v_n": (mid[1] - CY) / FOCAL_PX,
                "ipd_n": np.linalg.norm(px[0] - px[1]) / FOCAL_PX,
                "face_w_n": np.linalg.norm(px[2] - px[3]) / FOCAL_PX,
                "face_h_n": np.linalg.norm(px[4] - px[5]) / FOCAL_PX,
                "yaw_est": yaw + rng.normal(0, 2.5), "pitch_est": pitch + rng.normal(0, 2.5),
                "roll_est": roll + rng.normal(0, 2.5),
            }
            r = rng.random()
            if r < 0.03:                                              # veidas nerastas
                f = {k: np.nan for k in f}
            elif r < 0.04:                                            # klaidingas aptikimas
                f["ipd_n"] *= rng.choice([0.45, 1.8])
            ang = mirror_angles(eye)[0]
            rows.append({"driver_id": d.driver_id, **f,
                         "eye_x": eye[0], "eye_y": eye[1], "eye_z": eye[2],
                         "yaw_true": yaw, "pitch_true": pitch,
                         **{t: ang[i] for i, t in enumerate(TARGETS)},
                         **{f"{t}_pref": ang[i] + getattr(d, f"pref_{t}") for i, t in enumerate(TARGETS)}})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drivers", type=int, default=80)
    ap.add_argument("--frames", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    drv = make_drivers(a.drivers, rng)
    df = make_frames(drv, a.frames, rng)
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    drv.to_csv(out / "drivers.csv", index=False)
    df.to_csv(out / "dataset.csv", index=False)
    print(f"Sugeneruota: {len(drv)} vairuotoju, {len(df)} kadru -> {out}/dataset.csv")


if __name__ == "__main__":
    main()
