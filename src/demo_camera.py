"""
Gyva demonstracija su internetine kamera.

Kamera -> veido orientyrai (MediaPipe; jei nera - OpenCV) -> pozymiai -> hibridinis modelis
-> asmeninis profilis -> PASIULYMAS. Veidrodeliai (ekrane) pasisuka TIK vairuotojui patvirtinus
(ENTER) ir tik kai "automobilis stovi" - taip laikomasi taisykles, kad AI nevaldo irenginio
be zmogaus patvirtinimo.

Klavisai:
  C        - kalibruoti (atsisedus iprastai ~60 cm nuo kameros, ziurint tiesiai)
  1 / 2 / 3 - pasirinkti vairuotojo profili (kaip atminties mygtukai)
  ENTER    - patvirtinti pasiulyma (veidrodeliai pradeda suktis)
  W/S A/D  - rankinis kairiojo veidrodelio pataisymas (vert. / horiz.)
  I/K J/L  - rankinis desiniojo veidrodelio pataisymas
  P        - issaugoti rankini pataisyma i profili
  V        - imituoti "automobilis vaziuoja / stovi"
  Q / ESC  - iseiti

Paleidimas (is projekto aplanko):  python src/demo_camera.py
"""
import json
import math
import sys
import time
import urllib.request
from collections import deque
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import FEATURES, build_features  # noqa: E402
from geometry import IMG_W, mirror_angles, project  # noqa: E402
from models import DriverProfile  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CAMERA_INDEX = 0
WEBCAM_FOV_DEG = 60          # tipines nesiojamo kompiuterio kameros horizontalus matymo laukas
YAW_SIGN, PITCH_SIGN = 1, 1  # jei pasukus galva i kaire "yaw" rodomas neigiamas - pakeiskite i -1
TOL, MOTOR_SPEED, WINDOW = 0.5, 3.0, 10
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/1/face_landmarker.task")
PROFILES_FILE = ROOT / "profiles.json"


# ---------------- Veido orientyru detektoriai ----------------
class MediaPipeDetector:
    name = "MediaPipe Face Landmarker"

    def __init__(self):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
        path = ROOT / "models" / "face_landmarker.task"
        if not path.exists():
            print("Atsisiunciamas MediaPipe modelis (~4 MB)...")
            path.parent.mkdir(exist_ok=True)
            urllib.request.urlretrieve(MODEL_URL, path)
        opts = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(path)),
            running_mode=vision.RunningMode.VIDEO, num_faces=1,
            output_facial_transformation_matrixes=True)
        self.lm = vision.FaceLandmarker.create_from_options(opts)
        self.mp = mp
        self.t0 = time.time()

    def detect(self, frame):
        h, w = frame.shape[:2]
        img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        res = self.lm.detect_for_video(img, int((time.time() - self.t0) * 1000))
        if not res.face_landmarks:
            return None
        L = res.face_landmarks[0]
        P = lambda i: np.array([L[i].x * w, L[i].y * h])  # noqa: E731
        R = np.array(res.facial_transformation_matrixes[0])[:3, :3]
        pose = (YAW_SIGN * math.degrees(math.atan2(R[0, 2], R[2, 2])),
                PITCH_SIGN * math.degrees(math.atan2(-R[1, 2], R[2, 2])),
                math.degrees(math.atan2(R[1, 0], R[0, 0])))
        return {"eyes": (P(468), P(473)), "cheeks": (P(234), P(454)), "forehead": P(10), "chin": P(152),
                "pose": pose}


class HaarDetector:
    """Atsarginis variantas be MediaPipe: tik akys ir veido staciakampis, galvos pasukimas = 0."""
    name = "OpenCV Haar (atsarginis)"

    def __init__(self):
        import os
        dirs = [getattr(getattr(cv2, "data", None), "haarcascades", ""), "/usr/share/opencv4/haarcascades/",
                "/usr/share/opencv/haarcascades/"]
        d = next(d for d in dirs if d and os.path.exists(d + "haarcascade_frontalface_default.xml"))
        self.face = cv2.CascadeClassifier(d + "haarcascade_frontalface_default.xml")
        self.eye = cv2.CascadeClassifier(d + "haarcascade_eye_tree_eyeglasses.xml")

    def detect(self, frame):
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face.detectMultiScale(g, 1.15, 6, minSize=(90, 90))
        if len(faces) == 0:
            return None
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        eyes = sorted(self.eye.detectMultiScale(g[y:y + h * 5 // 8, x:x + w], 1.1, 6), key=lambda e: -e[2])[:2]
        if len(eyes) == 2:
            c = sorted([np.array([x + ex + ew / 2, y + ey + eh / 2]) for ex, ey, ew, eh in eyes], key=lambda p: p[0])
        else:
            c = [np.array([x + 0.3 * w, y + 0.4 * h]), np.array([x + 0.7 * w, y + 0.4 * h])]
        return {"eyes": (c[0], c[1]), "cheeks": (np.array([x, y + 0.5 * h]), np.array([x + w, y + 0.5 * h])),
                "forehead": np.array([x + w / 2, y + 0.1 * h]), "chin": np.array([x + w / 2, y + h]),
                "pose": (0.0, 0.0, 0.0)}


def make_detector():
    try:
        return MediaPipeDetector()
    except Exception as e:  # noqa: BLE001
        print(f"MediaPipe nepasiekiamas ({e}). Naudojamas OpenCV.")
        return HaarDetector()


# ---------------- Pozymiai ----------------
def raw_features(det, f_px, w, h):
    (l, r), (c1, c2) = det["eyes"], det["cheeks"]
    mid = (l + r) / 2
    yaw, pitch, roll = det["pose"]
    return {"u_n": (mid[0] - w / 2) / f_px, "v_n": (mid[1] - h / 2) / f_px,
            "ipd_n": np.linalg.norm(l - r) / f_px, "face_w_n": np.linalg.norm(c1 - c2) / f_px,
            "face_h_n": np.linalg.norm(det["forehead"] - det["chin"]) / f_px,
            "yaw_est": yaw, "pitch_est": pitch, "roll_est": roll}


def nominal_features():
    """Kokius pozymius automobilio kamera matytu vidutiniam vairuotojui (kalibravimo tikslas)."""
    pts = np.array([[0, 0.0315, 0], [0, -0.0315, 0], [-0.03, 0.0677, -0.04], [-0.03, -0.0677, -0.04],
                    [0.01, 0, 0.07], [0.02, 0, -0.12]])
    px = project(pts)
    from geometry import CX, CY, FOCAL_PX
    mid = (px[0] + px[1]) / 2
    return {"u_n": (mid[0] - CX) / FOCAL_PX, "v_n": (mid[1] - CY) / FOCAL_PX,
            "ipd_n": np.linalg.norm(px[0] - px[1]) / FOCAL_PX, "face_w_n": np.linalg.norm(px[2] - px[3]) / FOCAL_PX,
            "face_h_n": np.linalg.norm(px[4] - px[5]) / FOCAL_PX}


class Calibration:
    """Perkelia internetines kameros pozymius i automobilio kameros 'erdve'."""
    def __init__(self):
        self.ref = None
        self.nom = nominal_features()

    def set(self, f):
        self.ref = dict(f)

    def apply(self, f):
        g = dict(f)
        s = self.nom["ipd_n"] / self.ref["ipd_n"]          # kameru atstumo / zidinio santykis
        g["u_n"] = self.nom["u_n"] + (f["u_n"] - self.ref["u_n"]) * s
        g["v_n"] = self.nom["v_n"] + (f["v_n"] - self.ref["v_n"]) * s
        for k in ("ipd_n", "face_w_n", "face_h_n"):
            g[k] = f[k] * self.nom[k] / self.ref[k]
        return g


# ---------------- Veidrodeliai ir profiliai ----------------
class Mirrors:
    def __init__(self):
        self.angle = mirror_angles(np.zeros(3))[0].copy()
        self.target = self.angle.copy()
        self.moving = False

    def step(self, dt):
        if not self.moving:
            return
        err = self.target - self.angle
        if np.all(np.abs(err) < 0.05):
            self.moving = False
            return
        self.angle += np.clip(2.0 * err, -1, 1) * MOTOR_SPEED * dt


def load_profiles():
    try:
        data = json.load(open(PROFILES_FILE))
    except Exception:  # noqa: BLE001
        data = {}
    profs = {}
    for k in ("1", "2", "3"):
        p = DriverProfile()
        if k in data:
            p.sum, p.n = np.array(data[k]["sum"]), data[k]["n"]
        profs[k] = p
    return profs


def save_profiles(profs):
    json.dump({k: {"sum": p.sum.tolist(), "n": p.n} for k, p in profs.items()}, open(PROFILES_FILE, "w"), indent=1)


# ---------------- Piesimas ----------------
def put(img, text, y, color=(255, 255, 255), scale=0.5, x=10):
    """Tekstas ant tamsaus fono (aiskiai matosi bet kokiame vaizde)."""
    (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    roi = img[max(0, y - h - 4):y + base + 2, max(0, x - 4):x + w + 4]
    if roi.size:
        roi[:] = (roi * 0.35).astype(roi.dtype)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


# Automobilio vaizdas is virsaus: skydelio mastelis ir padetis
PANEL_W, PANEL_H, SCALE = 250, 278, 44          # px, px, px/m
ZOOM = 5                                        # kampu pokyciai ekrane padidinami 5 kartus, kad matytusi
CAR_CENTER_Y = -0.40                            # automobilio asis y koordinate (m), akys - 0
CAR_FRONT, CAR_REAR, CAR_HALF_W = 2.0, -2.5, 0.85


def _to_px(x, y, ox, oy):
    """Automobilio koordinates (x - i prieki, y - i kaire) -> skydelio pikseliai (prieki - virsus)."""
    return int(ox - (y - CAR_CENTER_Y) * SCALE), int(oy - x * SCALE)


NOMINAL = mirror_angles(np.zeros(3))[0]


def _mirror_geometry(key, angles):
    """Veidrodelio stiklo kryptis ir kryptis, kuria vairuotojas mato veidrodelyje (is virsaus)."""
    from geometry import MIRRORS
    i = 0 if key == "L" else 2
    base = NOMINAL[i]
    yaw = math.radians(base + ZOOM * (angles[i] - base))
    n = np.array([-math.cos(yaw), math.sin(yaw)])           # normale (x, y)
    m = MIRRORS[key]["pos"][:2]
    d = m / np.linalg.norm(m)                               # spindulys nuo akiu iki veidrodelio
    r = d - 2 * np.dot(d, n) * n                            # atspindetas spindulys
    g = np.array([n[1], -n[0]])                             # stiklo kryptis (statmena normalei)
    return m, g, r


def draw_panel(img, mirrors, proposal):
    H, W = img.shape[:2]
    x0, y0 = W - PANEL_W - 10, H - PANEL_H - 10
    roi = img[y0:y0 + PANEL_H, x0:x0 + PANEL_W]
    roi[:] = (roi * 0.25).astype(roi.dtype)
    put(img, "Veidrodeliai (vaizdas is virsaus)", y0 + 16, (230, 230, 230), 0.42, x0 + 8)
    ox, oy = x0 + PANEL_W // 2, y0 + 30 + int(CAR_FRONT * SCALE)
    # kebulas
    p1 = _to_px(CAR_FRONT, CAR_CENTER_Y + CAR_HALF_W, ox, oy)
    p2 = _to_px(CAR_REAR, CAR_CENTER_Y - CAR_HALF_W, ox, oy)
    for wx in (1.35, -1.6):                                                          # ratai
        for wy in (CAR_CENTER_Y + CAR_HALF_W, CAR_CENTER_Y - CAR_HALF_W):
            c = _to_px(wx, wy, ox, oy)
            cv2.rectangle(img, (c[0] - 4, c[1] - 11), (c[0] + 4, c[1] + 11), (60, 60, 60), -1)
    cv2.rectangle(img, p1, p2, (150, 150, 150), 2, cv2.LINE_AA)
    wsx1, wsy = _to_px(0.95, CAR_CENTER_Y + CAR_HALF_W - 0.08, ox, oy)
    wsx2, _ = _to_px(0.95, CAR_CENTER_Y - CAR_HALF_W + 0.08, ox, oy)
    cv2.line(img, (wsx1, wsy), (wsx2, wsy), (150, 150, 150), 2, cv2.LINE_AA)        # priekinis stiklas
    cv2.circle(img, _to_px(0, 0, ox, oy), 7, (190, 200, 230), -1, cv2.LINE_AA)       # vairuotojo galva
    cv2.putText(img, "prieki", (ox - 18, y0 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (170, 170, 170), 1, cv2.LINE_AA)
    for key in ("L", "R"):
        m, g, r = _mirror_geometry(key, mirrors.angle)
        mp = np.array(_to_px(m[0], m[1], ox, oy))
        # matymo laukas (~ +-8 laipsniai aplink atspindeta spinduli)
        ang = math.atan2(r[1], r[0])
        pts = [mp]
        for da in (-8, 8):
            a = ang + math.radians(da)
            end = (m[0] + 2.6 * math.cos(a), m[1] + 2.6 * math.sin(a))
            pts.append(np.array(_to_px(*end, ox, oy)))
        overlay = img.copy()
        cv2.fillPoly(overlay, [np.array(pts, np.int32)], (0, 200, 255))
        cv2.addWeighted(overlay, 0.45, img, 0.55, 0, img)
        # zvilgsnio linija nuo galvos iki veidrodelio
        cv2.line(img, _to_px(0, 0, ox, oy), tuple(int(v) for v in mp), (110, 110, 110), 1, cv2.LINE_AA)
        # pasiulyta padetis (plona oranzine) ir tikra padetis (stora balta)
        if proposal is not None:
            _, gp, _ = _mirror_geometry(key, proposal)
            a, b = m + gp * 0.3, m - gp * 0.3
            cv2.line(img, _to_px(*a, ox, oy), _to_px(*b, ox, oy), (0, 165, 255), 1, cv2.LINE_AA)
        a, b = m + g * 0.3, m - g * 0.3
        cv2.line(img, _to_px(*a, ox, oy), _to_px(*b, ox, oy), (255, 255, 255), 3, cv2.LINE_AA)
        # korpuso jungtis prie kebulo
        side = CAR_CENTER_Y + (CAR_HALF_W if key == "L" else -CAR_HALF_W)
        cv2.line(img, tuple(int(v) for v in mp), _to_px(m[0], side, ox, oy), (90, 90, 90), 3, cv2.LINE_AA)
        lab = "Kairysis" if key == "L" else "Desinysis"
        tx = int(mp[0]) - (62 if key == "L" else -8)
        cv2.putText(img, lab, (tx, int(mp[1]) + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.putText(img, "oranzine - ka matote veidrodelyje", (x0 + 8, y0 + PANEL_H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (170, 170, 170), 1, cv2.LINE_AA)
    cv2.putText(img, f"(kampu pokyciai padidinti {ZOOM}x)", (x0 + 8, y0 + PANEL_H - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (170, 170, 170), 1, cv2.LINE_AA)


def load_model():
    """Ikelia apmokyta modeli; jei nepavyksta (pvz., kita scikit-learn versija) - greitai apmoko is naujo."""
    try:
        return joblib.load(ROOT / "models" / "hybrid_model.joblib")["model"]
    except Exception as e:  # noqa: BLE001
        print(f"Modelio failas netinka ({e}). Apmokoma is naujo...")
        from features import clean
        from geometry import TARGETS
        from models import HybridMirrorModel
        df = build_features(clean(pd.read_csv(ROOT / "data" / "dataset.csv")))
        return HybridMirrorModel(alpha=0.1).fit(df, df[[t + "_pref" for t in TARGETS]].to_numpy())


def main():
    model = load_model()
    det = make_detector()
    cap = cv2.VideoCapture(CAMERA_INDEX)
    ok, frame = cap.read()
    if not ok:
        print("Kamera nerasta. Pakeiskite CAMERA_INDEX faile demo_camera.py.")
        return
    h, w = frame.shape[:2]
    f_px = (w / 2) / math.tan(math.radians(WEBCAM_FOV_DEG / 2))
    calib, mirrors, profiles = Calibration(), Mirrors(), load_profiles()
    active, stopped, manual = "1", True, np.zeros(4)
    hist, raw_hist, times = deque(maxlen=WINDOW), deque(maxlen=WINDOW), deque(maxlen=30)
    last, msg, msg_t = time.time(), "", 0
    print(f"Detektorius: {det.name}. Atsisedkite ~60 cm nuo kameros ir paspauskite C.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        now = time.time()
        dt, last = now - last, now
        t0 = time.perf_counter()
        d = det.detect(frame)
        proposal = None
        view = cv2.flip(frame, 1)
        if d is None:
            hist.clear()
            put(view, "Veidas nerastas", 22, (0, 0, 255))
        else:
            f = raw_features(d, f_px, w, h)
            raw_hist.append(f)
            for p in list(d["eyes"]) + [d["forehead"], d["chin"]]:
                cv2.circle(view, (int(w - p[0]), int(p[1])), 4, (0, 255, 255), -1)
            if calib.ref is None:
                put(view, "Atsisedkite ~60 cm nuo kameros, ziurekite tiesiai ir spauskite C", 22, (0, 255, 255))
            else:
                X = build_features(pd.DataFrame([calib.apply(f)]))
                hist.append(model.predict(X)[0])
                times.append((time.perf_counter() - t0) * 1000)
                if len(hist) == WINDOW:
                    proposal = profiles[active].apply(np.median(np.array(hist), axis=0))
                yaw, pitch, _ = d["pose"]
                put(view, f"Galva: yaw={yaw:+5.1f}  pitch={pitch:+5.1f}   apdorojimas {np.mean(times):.0f} ms/kadra",
                    22, (200, 200, 200), 0.45)
        put(view, f"Profilis {active} (pataisymu: {profiles[active].n})   "
                  f"Automobilis: {'STOVI' if stopped else 'VAZIUOJA'}   [{det.name}]", 44, (200, 200, 200), 0.45)
        if proposal is not None:
            tgt = proposal + manual
            diff = np.abs(tgt - mirrors.angle).max()
            put(view, f"Kair.: {tgt[0]:6.2f} / {tgt[1]:5.2f}   Des.: {tgt[2]:6.2f} / {tgt[3]:5.2f}   "
                      f"(dabar {mirrors.angle[0]:6.2f} / {mirrors.angle[1]:5.2f}  {mirrors.angle[2]:6.2f} / "
                      f"{mirrors.angle[3]:5.2f})", 68, scale=0.45)
            if mirrors.moving:
                put(view, "Veidrodeliai reguliuojami...", 94, (0, 255, 255), 0.6)
            elif diff > TOL and stopped:
                put(view, f"SIULOMA pakeisti padeti ({diff:.1f} laipsn.). ENTER - patvirtinti", 94, (0, 200, 255), 0.6)
            elif diff > TOL:
                put(view, "Vaziuojant veidrodeliai nereguliuojami", 94, (0, 0, 255), 0.6)
            else:
                put(view, "Veidrodeliai nustatyti", 94, (0, 255, 0), 0.6)
        if now - msg_t < 2.5:
            put(view, msg, h - 20, (0, 255, 0), 0.6)
        mirrors.step(dt)
        draw_panel(view, mirrors, None if proposal is None else proposal + manual)
        cv2.imshow("Automatinis veidrodeliu reguliavimas (AI)", view)

        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27):
            break
        if k == ord("c") and len(raw_hist) >= 5:
            calib.set({c: float(np.median([r[c] for r in raw_hist])) for c in FEATURES})
            hist.clear()
            msg, msg_t = "Kalibruota", now
        if k in (ord("1"), ord("2"), ord("3")):
            active, manual = chr(k), np.zeros(4)
            msg, msg_t = f"Pasirinktas profilis {active}", now
        if k == ord("v"):
            stopped = not stopped
            if not stopped:
                mirrors.moving = False
        if k == 13 and proposal is not None and stopped:            # ENTER - zmogaus patvirtinimas
            mirrors.target, mirrors.moving = proposal + manual, True
        step = {ord("w"): (1, 0.2), ord("s"): (1, -0.2), ord("a"): (0, 0.2), ord("d"): (0, -0.2),
                ord("i"): (3, 0.2), ord("k"): (3, -0.2), ord("j"): (2, 0.2), ord("l"): (2, -0.2)}
        if k in step and stopped:
            i, v = step[k]
            manual[i] += v
            mirrors.angle[i] += v
            mirrors.target = mirrors.angle.copy()
        if k == ord("p") and proposal is not None:
            profiles[active].add_correction(proposal + manual, proposal - profiles[active].offset)
            save_profiles(profiles)
            manual = np.zeros(4)
            msg, msg_t = f"Pataisymas issaugotas profilyje {active}", now
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
