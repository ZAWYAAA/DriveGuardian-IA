# ============================================================
# DriveGuardian IA - video_detector.py (version sans DEMO)
# ============================================================

import os
import sys
import re
import csv
import cv2
import numpy as np
import collections
import textwrap
import unicodedata

# winsound = Windows-only
try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False

# Matplotlib (graphes export PNG)
try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[GRAPHS] Matplotlib non installe. Faites `pip install matplotlib` pour activer les graphes.")

from risk_analysis import analyze_risk
from recommendations import generate_report


# -------------------------------------------------
# 0) Debug: imprimer les paramètres
# -------------------------------------------------
def print_params_from_globals(g, title="PARAMS"):
    """
    Affiche toutes les variables simples (int/float/str/dict/list/tuple/bool)
    visibles dans un namespace.
    """
    allowed_types = (int, float, str, dict, tuple, list, bool)
    keys = [k for k in g.keys() if not k.startswith("__")]

    print(f"\n=== {title} ===")
    print(f"[DEBUG] nb variables visibles = {len(keys)}")

    printed = 0
    for k in sorted(keys):
        v = g[k]
        if isinstance(v, allowed_types):
            print(f"{k} = {v}")
            printed += 1

    if printed == 0:
        print("[INFO] Aucune variable simple detectee ici.")
    print("=" * (8 + len(title)) + "\n")


# -------------------------------------------------
# 1) Paramètres globaux
# -------------------------------------------------

# Calibration distance approximative (heuristique)
DISTANCE_K = 5.0
MIN_DISTANCE_M = 3.0
MAX_DISTANCE_M = 80.0

# Mode de conduite : "city" ou "highway"
MODE = "city"

# Vidéo / performance
SPEED = 1.5
TARGET_WIDTH, TARGET_HEIGHT = 854, 480
DETECT_VEHICLE_EVERY_N = 3

# Audio
ENABLE_WARNING_AUDIO = True
ENABLE_DANGER_AUDIO = True

WARNING_MIN_GAP = 3.5
DANGER_MIN_GAP = 2.5
DANGER_OVERRIDE_GAP = 0.3

# Panneau dashboard (partie data)
PANEL_W = 640
PANEL_H = 360

# Stabilité audio (lissage)
RISK_STABLE_RATIO = 0.6

# Graphes
SHOW_GRAPHS = False  

# Chemins
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
VIDEO_PATH = os.path.join(BASE_DIR, "data", "raw_videos", "trajet_01_jour.mp4")

AUDIO_DIR = os.path.join(BASE_DIR, "data", "audio")
DANGER_AUDIO = os.path.join(AUDIO_DIR, "danger.wav")
WARNING_AUDIO = os.path.join(AUDIO_DIR, "warning.wav")

CAR_CASCADE_PATH = os.path.join(BASE_DIR, "data", "models", "cars.xml")


# -------------------------------------------------
# 2) Utilitaires affichage (Unicode -> ASCII pour OpenCV)
# -------------------------------------------------
def normalize_for_opencv(text: str) -> str:
    """
    Supprime les accents et caractères spéciaux pour cv2.putText
    (OpenCV gère mal l'Unicode selon les polices).
    """
    if not isinstance(text, str):
        return str(text)
    text_norm = unicodedata.normalize("NFKD", text)
    return text_norm.encode("ascii", "ignore").decode("ascii")


# -------------------------------------------------
# 3) Audio
# -------------------------------------------------
def speak(kind: str):
    """
    kind : 'danger' ou 'warning'
    Joue le .wav correspondant (si activé).
    """
    global ENABLE_WARNING_AUDIO, ENABLE_DANGER_AUDIO

    if kind == "danger":
        if not ENABLE_DANGER_AUDIO:
            return
        path = DANGER_AUDIO
    elif kind == "warning":
        if not ENABLE_WARNING_AUDIO:
            return
        path = WARNING_AUDIO
    else:
        print(f"[AUDIO] Type inconnu: {kind}")
        return

    if not os.path.exists(path):
        print(f"[AUDIO] Fichier introuvable : {path}")
        return

    if HAS_WINSOUND:
        winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC)
    else:
        print(f"[AUDIO] winsound indisponible sur cet OS. (son non joue) -> {path}")


# -------------------------------------------------
# 4) Vision : voies
# -------------------------------------------------
def preprocess_lane(frame_bgr):
    """
    Filtre les marquages blancs/jaunes (HLS) puis Canny.
    """
    hls = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HLS)
    _, l, _ = cv2.split(hls)

    white_mask = cv2.inRange(l, 200, 255)

    lower_yellow = np.array([15, 30, 80])
    upper_yellow = np.array([35, 204, 255])
    yellow_mask = cv2.inRange(hls, lower_yellow, upper_yellow)

    mask = cv2.bitwise_or(white_mask, yellow_mask)
    blurred = cv2.GaussianBlur(mask, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    return edges


def region_of_interest(image_gray):
    h, w = image_gray.shape[:2]
    polygon = np.array([[
        (int(w * 0.05), h),
        (int(w * 0.30), int(h * 0.60)),
        (int(w * 0.70), int(h * 0.60)),
        (int(w * 0.95), h)
    ]], dtype=np.int32)

    mask = np.zeros_like(image_gray)
    cv2.fillPoly(mask, polygon, 255)
    return cv2.bitwise_and(image_gray, mask)


def detect_lane_lines(edges_roi):
    lines = cv2.HoughLinesP(
        edges_roi,
        rho=1,
        theta=np.pi / 180,
        threshold=40,
        minLineLength=40,
        maxLineGap=120
    )

    line_image = np.zeros((edges_roi.shape[0], edges_roi.shape[1], 3), dtype=np.uint8)
    if lines is not None:
        for (x1, y1, x2, y2) in lines[:, 0]:
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 5)
    return line_image, lines


def combine_images(frame_bgr, line_image):
    return cv2.addWeighted(frame_bgr, 0.75, line_image, 1.0, 1.0)


def estimate_lane_offset_from_lines(lines, frame_shape):
    """
    Estime offset normalisé (offset_px / lane_width) et une confiance
    """
    h, w = frame_shape[:2]
    if lines is None or len(lines) == 0:
        return 0.0, 0.0

    left_lines, right_lines = [], []

    for (x1, y1, x2, y2) in lines[:, 0]:
        if x2 == x1:
            continue
        slope = (y2 - y1) / (x2 - x1)
        if abs(slope) < 0.2:
            continue
        if max(y1, y2) < h * 0.55:
            continue

        if slope < 0 and x1 < w * 0.5 and x2 < w * 0.5:
            left_lines.append((x1, y1, x2, y2))
        elif slope > 0 and x1 > w * 0.5 and x2 > w * 0.5:
            right_lines.append((x1, y1, x2, y2))

    if not left_lines or not right_lines:
        return 0.0, 0.0

    y_eval = int(h * 0.8)

    def avg_x_at_y(lines_list):
        xs = []
        for x1, y1, x2, y2 in lines_list:
            if y2 == y1:
                continue
            if not (min(y1, y2) <= y_eval <= max(y1, y2)):
                continue
            x_at = x1 + (y_eval - y1) * (x2 - x1) / (y2 - y1)
            xs.append(x_at)
        if not xs:
            return None
        return sum(xs) / len(xs)

    x_left = avg_x_at_y(left_lines)
    x_right = avg_x_at_y(right_lines)
    if x_left is None or x_right is None or x_right <= x_left:
        return 0.0, 0.0

    lane_center = (x_left + x_right) / 2.0
    vehicle_center = w / 2.0
    lane_width = x_right - x_left

    offset_px = vehicle_center - lane_center
    offset_norm = offset_px / lane_width
    offset_norm = max(-2.0, min(2.0, offset_norm))

    used_count = len(left_lines) + len(right_lines)
    total_count = len(lines)
    confidence = min(1.0, used_count / max(1, total_count))

    return float(offset_norm), float(confidence)


def lane_status_from_offset(offset_norm, confidence, mode="city"):
    """
    City = plus tolérant. Highway = plus strict.
    """
    if confidence < 0.2:
        return "center"

    abs_off = abs(offset_norm)

    if mode == "highway":
        if abs_off < 0.12:
            return "center"
        elif abs_off < 0.35:
            return "near_line"
        else:
            return "out_of_lane"
    else:
        if abs_off < 0.20:
            return "center"
        elif abs_off < 0.50:
            return "near_line"
        else:
            return "out_of_lane"


# -------------------------------------------------
# 5) Véhicules : Haar + pseudo radar + distance + clignotant
# -------------------------------------------------
def vehicle_relative_position(box, frame_width):
    x, y, w, h = box
    cx = x + w / 2
    if cx < frame_width * 0.4:
        return "left"
    elif cx > frame_width * 0.6:
        return "right"
    else:
        return "center"


def detect_turn_signal_for_vehicle(frame, box):
    """
    Heuristique V1 : HSV + pixels orangés/jaunes dans la ROI du véhicule.
    Retour : left/right/both/none
    """
    x, y, w, h = box
    roi = frame[y:y + h, x:x + w]
    if roi.size == 0:
        return "none"

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower_orange = np.array([10, 120, 180])
    upper_orange = np.array([40, 255, 255])
    mask = cv2.inRange(hsv, lower_orange, upper_orange)

    total_pixels = w * h
    if total_pixels <= 0:
        return "none"

    pix_thresh = int(0.003 * total_pixels)
    if np.count_nonzero(mask) < pix_thresh:
        return "none"

    mid = w // 2
    left_count = np.count_nonzero(mask[:, :mid])
    right_count = np.count_nonzero(mask[:, mid:])

    if left_count > pix_thresh and right_count < pix_thresh:
        return "left"
    if right_count > pix_thresh and left_count < pix_thresh:
        return "right"
    if left_count > pix_thresh and right_count > pix_thresh:
        return "both"
    return "none"


def estimate_distance_and_zone(vehicle_box, frame_height, mode="city"):
    """
    Distance heuristique basée sur la hauteur bbox :
    distance_est ≈ K / (h_ratio).
    """
    if vehicle_box is None:
        return None, "safe"

    _, _, _, h = vehicle_box
    h_ratio = h / float(frame_height)
    if h_ratio <= 0:
        return None, "safe"

    distance_est = DISTANCE_K / h_ratio
    distance_est = max(MIN_DISTANCE_M, min(MAX_DISTANCE_M, distance_est))

    if mode == "highway":
        if distance_est > 70:
            zone = "safe"
        elif distance_est > 45:
            zone = "close"
        else:
            zone = "very_close"
    else:
        if distance_est > 40:
            zone = "safe"
        elif distance_est > 22:
            zone = "close"
        else:
            zone = "very_close"

    return distance_est, zone


def detect_vehicles(frame, car_cascade, mode="city", max_vehicles=3):
    if car_cascade is None:
        return []

    h, w = frame.shape[:2]
    y_start = int(h * 0.55)
    y_end = h
    x_start = int(w * 0.10)
    x_end = int(w * 0.90)

    roi = frame[y_start:y_end, x_start:x_end]
    if roi.size == 0:
        return []

    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray_roi = cv2.equalizeHist(gray_roi)

    cars = car_cascade.detectMultiScale(
        gray_roi,
        scaleFactor=1.05,
        minNeighbors=3,
        minSize=(35, 35)
    )

    vehicles = []
    for (x, y, w_box, h_box) in cars:
        x_full = x_start + x
        y_full = y_start + y
        box = (x_full, y_full, w_box, h_box)

        distance_est, zone = estimate_distance_and_zone(box, h, mode)
        rel_pos = vehicle_relative_position(box, w)
        signal_dir = detect_turn_signal_for_vehicle(frame, box)

        vehicles.append({
            "box": box,
            "distance": distance_est,
            "zone": zone,
            "rel_pos": rel_pos,
            "signal": signal_dir,
        })

    vehicles = [v for v in vehicles if v["distance"] is not None]
    vehicles.sort(key=lambda v: v["distance"])
    return vehicles[:max_vehicles]


def select_primary_vehicle(vehicles):
    if not vehicles:
        return None
    for v in vehicles:
        if v["rel_pos"] == "center":
            return v
    return vehicles[0]


# -------------------------------------------------
# 6) Risque + score + texte contexte
# -------------------------------------------------
def classify_instant_risk(lane_status, distance_zone):
    if distance_zone == "very_close" or lane_status == "out_of_lane":
        return "DANGER"
    if distance_zone == "close":
        return "WARNING"
    return "SAFE"


def compute_risk_score(lane_status, distance_zone, lane_offset, primary_vehicle, vehicles, mode="city"):
    """
    Score 0–100 (heuristique pondérée).
    """
    score = 0

    if mode == "highway":
        lane_near_w = 20
        lane_out_w = 40
        offset_scale = 60
        close_w = 30
        very_close_w = 45
        multi_car_w = 7
        signal_w = 18
        close_dist_thresh = 40
    else:
        lane_near_w = 10
        lane_out_w = 25
        offset_scale = 35
        close_w = 20
        very_close_w = 35
        multi_car_w = 4
        signal_w = 12
        close_dist_thresh = 30

    if lane_status == "near_line":
        score += lane_near_w
    elif lane_status == "out_of_lane":
        score += lane_out_w

    score += min(25, abs(lane_offset) * offset_scale)

    if distance_zone == "close":
        score += close_w
    elif distance_zone == "very_close":
        score += very_close_w

    close_veh = [v for v in vehicles if v.get("distance") is not None and v["distance"] < close_dist_thresh]
    score += multi_car_w * min(len(close_veh), 3)

    for v in close_veh:
        if v.get("signal") != "none":
            score += signal_w
            break

    return max(0, min(100, int(score)))


def describe_current_risk(lane_status, distance_zone, lane_offset, primary_vehicle, vehicles, mode="city"):
    """
    Message court + conseil, adapté au mode.
    (Texte sans accents -> plus robuste avec OpenCV)
    """
    if mode == "highway":
        prefix_crit = "Sur autoroute, "
        prefix_info = "A cette vitesse, "
    else:
        prefix_crit = "En ville, "
        prefix_info = "En circulation urbaine, "

    direction = None
    if lane_status in ("near_line", "out_of_lane"):
        if lane_offset > 0.08:
            direction = "droite"
        elif lane_offset < -0.08:
            direction = "gauche"

    dist_txt = None
    if primary_vehicle and primary_vehicle.get("distance") is not None:
        dist_txt = f"~{int(primary_vehicle['distance'])} m"

    close_veh = [v for v in vehicles if v.get("distance") is not None and v["distance"] < (40 if mode == "highway" else 30)]

    signal_info = None
    for v in vehicles:
        if v.get("distance") is not None and v["distance"] < 40 and v.get("signal") != "none":
            side = v["rel_pos"]
            side_txt = "gauche" if side == "left" else ("droite" if side == "right" else "devant")
            signal_info = (side_txt, int(v["distance"]))
            break

    if lane_status == "out_of_lane" and distance_zone == "very_close":
        if direction and dist_txt:
            return (prefix_crit +
                    f"vehicule tres proche devant ({dist_txt}) et trajectoire hors voie sur la {direction}. "
                    "Ralentissez et revenez progressivement au centre.")
        if direction:
            return (prefix_crit +
                    f"trajectoire hors voie sur la {direction}. "
                    "Revenez au centre et recreez une distance de securite.")
        return (prefix_crit +
                "trajectoire instable et distance tres reduite. "
                "Ralentissez immediatement et stabilisez le vehicule.")

    if distance_zone == "very_close":
        if dist_txt:
            return (prefix_info +
                    f"distance tres courte avec le vehicule devant ({dist_txt}). "
                    "Ralentissez et augmentez l'ecart.")
        return (prefix_info +
                "distance tres courte avec le vehicule devant. "
                "Relachez l'accelerateur pour recreer un espace de securite.")

    if lane_status == "near_line" and distance_zone == "safe":
        if direction:
            return (prefix_info +
                    f"vous vous rapprochez du marquage de voie a {direction}. "
                    "Corrigez legerement pour rester au centre.")
        return (prefix_info +
                "vous vous rapprochez du marquage de voie. "
                "Stabilisez la trajectoire au centre.")

    if distance_zone == "close":
        if dist_txt:
            msg = (prefix_info +
                   f"vehicule relativement proche devant ({dist_txt}). "
                   "Anticipez et evitez de rester colle.")
        else:
            msg = (prefix_info +
                   "vehicule relativement proche devant. "
                   "Gardez une marge de securite.")
        if len(close_veh) >= 2:
            msg += " Trafic dense, evitez les changements de voie brusques."
        return msg

    if lane_status == "out_of_lane":
        if direction:
            return (prefix_info +
                    f"trajectoire hors voie sur la {direction}. "
                    "Ramenez progressivement le vehicule vers le centre.")
        return (prefix_info +
                "trajectoire hors voie. Corrigez doucement pour revenir au centre.")

    if signal_info:
        side_txt, d = signal_info
        return (prefix_info +
                f"vehicule a {side_txt} avec clignotant actif (~{d} m). "
                "Anticipez son changement de file et adaptez votre vitesse.")

    if len(close_veh) >= 2:
        return (prefix_info +
                "plusieurs vehicules proches autour de vous. "
                "Restez previsible et evitez les manoeuvres brusques.")

    if lane_status == "center" and distance_zone == "safe":
        if mode == "highway":
            return ("Conduite stable sur autoroute : voie centree et distance correcte. "
                    "Maintenez ce niveau de prudence.")
        return ("Conduite stable en ville : voie centree et distance suffisante. "
                "Restez attentif aux changements de trafic.")

    return prefix_info + "situation globalement stable, restez attentif a l'environnement."


# -------------------------------------------------
# 7) Fenêtre finale OpenCV "rapport"
# -------------------------------------------------
def show_report_window(report_text: str, window_name: str = "DriveGuardian - Rapport"):
    report_text = normalize_for_opencv(report_text)

    h, w = 700, 900
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = (15, 15, 15)

    cv2.rectangle(img, (0, 0), (w, 70), (40, 40, 40), -1)
    cv2.putText(img, "DriveGuardian IA - Bilan du trajet", (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    raw_lines = [ln.rstrip() for ln in report_text.splitlines()]
    body_lines = [ln for ln in raw_lines if ln.strip() and not ln.lstrip().startswith(("-", "•"))]
    bullet_lines = [ln.lstrip(" -•").strip() for ln in raw_lines if ln.lstrip().startswith(("-", "•"))]

    y = 100
    cv2.putText(img, "Synthese :", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    y += 30

    for ln in body_lines:
        wrapped = textwrap.wrap(ln, width=85)
        for sub in wrapped:
            if y >= h - 180:
                break
            cv2.putText(img, sub, (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
            y += 20
        if y >= h - 180:
            break

    y = h - 160
    cv2.putText(img, "Recommandations clefs :", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
    y += 30

    if bullet_lines:
        for b in bullet_lines[:4]:
            wrapped = textwrap.wrap(b, width=70)
            for sub in wrapped:
                if y >= h - 30:
                    break
                cv2.putText(img, "- " + sub, (35, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
                y += 20
            if y >= h - 30:
                break
    else:
        cv2.putText(img, "(Voir le rapport detaille dans le fichier texte.)", (35, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 150), 1)

    cv2.putText(img, "Appuyez sur une touche pour fermer", (20, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)

    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    cv2.destroyWindow(window_name)


# -------------------------------------------------
# 8) Main
# -------------------------------------------------
def main():
    global MODE, ENABLE_WARNING_AUDIO, ENABLE_DANGER_AUDIO

    # --- Vérifs fichiers ---
    if not os.path.exists(VIDEO_PATH):
        raise FileNotFoundError(f"Video introuvable : {VIDEO_PATH}")

    print(f"Utilisation de la video : {VIDEO_PATH}")

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError("Impossible d'ouvrir la video. Verifie le chemin ou le format.")

    car_cascade = None
    if os.path.exists(CAR_CASCADE_PATH):
        car_cascade = cv2.CascadeClassifier(CAR_CASCADE_PATH)
        if car_cascade.empty():
            print("⚠️  Impossible de charger le cascade de voitures.")
            car_cascade = None
        else:
            print(f"✅ Cascade voiture charge depuis : {CAR_CASCADE_PATH}")
    else:
        print("⚠️  Fichier cars.xml introuvable dans data/models/.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0:
        fps = 25.0

    base_delay_ms = 1000.0 / fps
    delay_ms = int(base_delay_ms / SPEED)

    print(f"FPS video : {fps}")
    print(f"Nombre de frames : {frame_count}")
    print(f"Duree approx : {frame_count / fps / 60:.1f} minutes")
    print(f"Delay utilise : {delay_ms} ms (SPEED={SPEED})")

    # --- Variables run ---
    frame_index = 0
    paused = False

    frames_metrics = []

    total_time_elapsed = 0.0
    time_lane_center = 0.0
    time_lane_near_line = 0.0
    time_lane_out = 0.0

    lane_offset_history = collections.deque(maxlen=12)
    risk_score_history = collections.deque(maxlen=20)
    risk_history = collections.deque(maxlen=15)

    last_spoken_risk = None
    last_speech_time = -999.0

    print("\nControles clavier :")
    print("  Espace  : pause / reprise")
    print("  d       : avancer de ~5 secondes")
    print("  a       : reculer de ~5 secondes")
    print("  1       : activer / couper audio WARNING")
    print("  2       : activer / couper audio DANGER")
    print("  h       : mode autoroute")
    print("  v       : mode ville")
    print("  q       : quitter\n")

    # ---- DEBUG PARAMS ----
    if "--print-params" in sys.argv:
        print_params_from_globals(globals(), title="video_detector.py (globals)")
        try:
            import risk_analysis
            print_params_from_globals(risk_analysis.__dict__, title="risk_analysis.py")
        except Exception as e:
            print("[WARN] risk_analysis non lu:", e)
        try:
            import recommendations
            print_params_from_globals(recommendations.__dict__, title="recommendations.py")
        except Exception as e:
            print("[WARN] recommendations non lu:", e)

        cap.release()
        cv2.destroyAllWindows()
        sys.exit(0)

    # --- boucle ---
    frame = None
    roi_edges = None
    combo = None

    distance_zone_display = "safe"
    distance_est_display = None
    vehicles_display = []

    risk_level_display = "SAFE"
    risk_score_display = 0.0
    risk_context_text = "Analyse en cours..."

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("Fin de la video ou erreur de lecture.")
                break

            frame_index += 1
            frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))

            lane_edges = preprocess_lane(frame)
            roi_edges = region_of_interest(lane_edges)
            line_image, lines = detect_lane_lines(roi_edges)
            combo = combine_images(frame.copy(), line_image)

            dt = 1.0 / fps
            total_time_elapsed += dt

            lane_offset_raw, lane_conf = estimate_lane_offset_from_lines(lines, frame.shape)
            if lane_conf > 0.2 or not lane_offset_history:
                lane_offset_history.append(lane_offset_raw)
            else:
                lane_offset_history.append(lane_offset_history[-1])

            lane_offset_smooth = sum(lane_offset_history) / len(lane_offset_history)
            lane_status_current = lane_status_from_offset(lane_offset_smooth, lane_conf, MODE)

            if lane_status_current == "center":
                time_lane_center += dt
            elif lane_status_current == "near_line":
                time_lane_near_line += dt
            elif lane_status_current == "out_of_lane":
                time_lane_out += dt

            if frame_index % DETECT_VEHICLE_EVERY_N == 0:
                vehicles_display = detect_vehicles(frame, car_cascade, MODE, max_vehicles=3)

            if vehicles_display:
                primary_vehicle = select_primary_vehicle(vehicles_display)
                distance_est = primary_vehicle["distance"]
                distance_zone = primary_vehicle["zone"]

                # petit "relax" si véhicule latéral sans clignotant
                if primary_vehicle["rel_pos"] != "center" and primary_vehicle["signal"] == "none":
                    if distance_zone == "very_close":
                        distance_zone = "close"
            else:
                primary_vehicle = None
                distance_est = None
                distance_zone = "safe"

            distance_zone_display = distance_zone
            distance_est_display = distance_est

            # Risque brut + lissage pour audio
            risk_level_raw = classify_instant_risk(lane_status_current, distance_zone)
            risk_history.append(risk_level_raw)

            if len(risk_history) >= 5:
                danger_ratio = risk_history.count("DANGER") / len(risk_history)
                warning_ratio = risk_history.count("WARNING") / len(risk_history)
                if danger_ratio >= RISK_STABLE_RATIO:
                    risk_for_audio = "DANGER"
                elif warning_ratio >= RISK_STABLE_RATIO:
                    risk_for_audio = "WARNING"
                else:
                    risk_for_audio = "SAFE"
            else:
                risk_for_audio = risk_level_raw

            risk_level_display = risk_for_audio

            # Score + contexte
            risk_score = compute_risk_score(
                lane_status_current,
                distance_zone,
                lane_offset_smooth,
                primary_vehicle,
                vehicles_display,
                MODE
            )
            risk_score_history.append(risk_score)
            risk_score_display = sum(risk_score_history) / len(risk_score_history)

            risk_context_text = describe_current_risk(
                lane_status_current,
                distance_zone,
                lane_offset_smooth,
                primary_vehicle,
                vehicles_display,
                MODE
            )
            risk_context_text = normalize_for_opencv(risk_context_text)

            # Audio anti-spam
            time_since_last = total_time_elapsed - last_speech_time

            if risk_for_audio == "DANGER":
                if last_spoken_risk != "DANGER":
                    if time_since_last > DANGER_OVERRIDE_GAP:
                        speak("danger")
                        last_spoken_risk = "DANGER"
                        last_speech_time = total_time_elapsed
                else:
                    if time_since_last > DANGER_MIN_GAP:
                        speak("danger")
                        last_speech_time = total_time_elapsed

            elif risk_for_audio == "WARNING":
                if last_spoken_risk == "DANGER" and time_since_last < DANGER_MIN_GAP:
                    pass
                else:
                    if last_spoken_risk != "WARNING" and time_since_last > WARNING_MIN_GAP:
                        speak("warning")
                        last_spoken_risk = "WARNING"
                        last_speech_time = total_time_elapsed
            else:
                last_spoken_risk = "SAFE"

            # Log métriques
            frames_metrics.append({
                "dt": dt,
                "t": total_time_elapsed,
                "lane_status": lane_status_current,
                "distance_zone": distance_zone,
                "lane_offset": lane_offset_smooth,
                "distance_est": distance_est,
                "risk_level": risk_level_display,
                "risk_score": risk_score,
                "mode": MODE,
            })

        # ---------- AFFICHAGE ----------
        if frame is None or roi_edges is None or combo is None:
            key = cv2.waitKey(delay_ms) & 0xFF
            if key == ord("q"):
                break
            continue

        # Temps (mm:ss)
        current_time_sec = frame_index / fps
        minutes = int(current_time_sec // 60)
        seconds = int(current_time_sec % 60)

        frame_display = frame.copy()
        combo_display = combo.copy()
        h, w = frame.shape[:2]

        # HUD (bandeau haut)
        hud_height = 50
        hud = np.zeros((hud_height, w, 3), dtype=np.uint8)
        hud[:] = (25, 25, 25)

        title_text = "DriveGuardian IA"
        mode_text = "Mode: Autoroute" if MODE == "highway" else "Mode: Ville"
        cv2.putText(hud, title_text, (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(hud, mode_text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 2)

        if risk_level_display == "SAFE":
            risk_color = (0, 200, 0)
        elif risk_level_display == "WARNING":
            risk_color = (0, 215, 255)
        else:
            risk_color = (0, 0, 255)

        time_text = f"{minutes:02d}:{seconds:02d}"
        cv2.putText(hud, time_text, (w - 100, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)

        badge_text = f"{risk_level_display}"
        (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        badge_w = tw + 20
        badge_h = th + 12
        bx1 = w // 2 - badge_w // 2
        by1 = 10
        bx2 = bx1 + badge_w
        by2 = by1 + badge_h
        cv2.rectangle(hud, (bx1, by1), (bx2, by2), risk_color, -1)
        cv2.putText(hud, badge_text, (bx1 + 10, by1 + badge_h - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        combo_display_with_hud = np.vstack((hud, combo_display))

        # Zone ROI véhicules (rectangle)
        y_start = int(h * 0.55)
        y_end = h
        x_start = int(w * 0.10)
        x_end = int(w * 0.90)
        cv2.rectangle(combo_display_with_hud, (x_start, y_start + hud_height),
                      (x_end, y_end + hud_height), (255, 0, 0), 1)

        # Dessin véhicules + labels
        if vehicles_display and car_cascade is not None:
            for i, v in enumerate(vehicles_display):
                x, y, w_box, h_box = v["box"]
                zone = v["zone"]
                rel = v["rel_pos"]
                dist = v["distance"]
                sig = v["signal"]

                color = (0, 200, 0) if zone == "safe" else (0, 0, 255)
                cv2.rectangle(combo_display_with_hud, (x, y + hud_height), (x + w_box, y + h_box + hud_height), color, 2)

                label = f"#{i+1} {rel}"
                if dist is not None:
                    label += f" ~{int(dist)}m"
                label += f" [{zone}]"
                if sig == "left":
                    label += " <-"
                elif sig == "right":
                    label += " ->"
                elif sig == "both":
                    label += " <->"

                cv2.putText(combo_display_with_hud, label,
                            (x, max(y + hud_height - 5, hud_height + 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Data panel
        data_panel = np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)
        data_panel[:] = (15, 15, 15)

        cv2.rectangle(data_panel, (0, 0), (PANEL_W, 50), (35, 35, 35), -1)
        cv2.putText(data_panel, "Analyse temps reel", (20, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        block_y = 70
        cv2.putText(data_panel, "Trajectoire & voie", (20, block_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)

        if total_time_elapsed > 0:
            ratio_center = time_lane_center / total_time_elapsed
            ratio_near = time_lane_near_line / total_time_elapsed
            ratio_out = time_lane_out / total_time_elapsed
        else:
            ratio_center = ratio_near = ratio_out = 0.0

        cv2.putText(data_panel, f"Temps d'analyse : {int(total_time_elapsed)} s", (20, block_y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        cv2.putText(data_panel, f"Centre       : {int(ratio_center * 100)} %", (20, block_y + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(data_panel, f"Proche ligne : {int(ratio_near * 100)} %", (20, block_y + 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(data_panel, f"Hors voie    : {int(ratio_out * 100)} %", (20, block_y + 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 128, 255), 2)

        block2_y = block_y + 200
        cv2.putText(data_panel, "Distance et risque", (20, block2_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)

        cv2.putText(data_panel, f"Zone : {distance_zone_display}", (20, block2_y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

        if distance_est_display is not None:
            cv2.putText(data_panel, f"Distance estimee : ~{distance_est_display:.0f} m", (20, block2_y + 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        else:
            cv2.putText(data_panel, "Distance estimee : N/A", (20, block2_y + 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 120, 120), 2)

        cv2.putText(data_panel, f"Decalage lateral : {lane_offset_smooth:+.2f}", (20, block2_y + 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        cv2.putText(data_panel, f"Niveau de risque : {risk_level_display}", (20, block2_y + 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, risk_color, 2)
        cv2.putText(data_panel, f"Score de risque  : {int(risk_score_display):3d}/100", (20, block2_y + 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        # contexte sur 2 lignes
        cv2.putText(data_panel, risk_context_text[:45], (20, block2_y + 175),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 2)
        if len(risk_context_text) > 45:
            cv2.putText(data_panel, risk_context_text[45:90], (20, block2_y + 195),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 2)

        audio_status = f"Audio W:{'ON' if ENABLE_WARNING_AUDIO else 'OFF'}  D:{'ON' if ENABLE_DANGER_AUDIO else 'OFF'}"
        cv2.putText(data_panel, audio_status, (20, block2_y + 220),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)

        # dashboard layout (2x2)
        roi_edges_bgr = cv2.cvtColor(roi_edges, cv2.COLOR_GRAY2BGR)
        bottom_left = cv2.resize(roi_edges_bgr, (PANEL_W, PANEL_H))
        top_left = cv2.resize(frame_display, (PANEL_W, PANEL_H))
        top_right = cv2.resize(combo_display_with_hud, (PANEL_W, PANEL_H))
        bottom_right = data_panel

        top_row = np.hstack((top_left, top_right))
        bottom_row = np.hstack((bottom_left, bottom_right))
        dashboard = np.vstack((top_row, bottom_row))

        cv2.imshow("DriveGuardian - Dashboard", dashboard)

        wait = delay_ms if not paused else 50
        key = cv2.waitKey(wait) & 0xFF

        # Si l'utilisateur ferme la fenêtre (selon OS), on sort
        if cv2.getWindowProperty("DriveGuardian - Dashboard", cv2.WND_PROP_VISIBLE) < 1:
            print("Fenetre fermee par l'utilisateur.")
            break

        if key == ord("q"):
            print("Sortie demandee par l'utilisateur.")
            break
        elif key == ord(" "):
            paused = not paused
        elif key == ord("d"):
            jump_frames = int(fps * 5)
            new_index = min(int(cap.get(cv2.CAP_PROP_POS_FRAMES)) + jump_frames, frame_count - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, new_index)
        elif key == ord("a"):
            jump_frames = int(fps * 5)
            new_index = max(int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - jump_frames, 0)
            cap.set(cv2.CAP_PROP_POS_FRAMES, new_index)
        elif key == ord("1"):
            ENABLE_WARNING_AUDIO = not ENABLE_WARNING_AUDIO
            print(f"Audio WARNING : {'ON' if ENABLE_WARNING_AUDIO else 'OFF'}")
        elif key == ord("2"):
            ENABLE_DANGER_AUDIO = not ENABLE_DANGER_AUDIO
            print(f"Audio DANGER : {'ON' if ENABLE_DANGER_AUDIO else 'OFF'}")
        elif key == ord("h"):
            MODE = "highway"
            print("Mode de conduite : AUTOROUTE")
        elif key == ord("v"):
            MODE = "city"
            print("Mode de conduite : VILLE")

    # fin boucle
    cap.release()
    cv2.destroyAllWindows()

    # -------------------------------------------------
    # 9) Post-analyse + rapport + exports
    # -------------------------------------------------
    print("\n=== Analyse de risque en cours... ===")
    risk_summary = analyze_risk(frames_metrics)

    context = {
        "trajet_nom": "Trajet video 1",
        "type_route": "inconnu (video)",
        "conditions": "non renseignees"
    }

    report = generate_report(risk_summary, context)

    print("\n\n===== BILAN DU TRAJET =====\n")
    print(report)
    print("\n===== FIN DU BILAN =====\n")

    # 1) rapport texte
    reports_dir = os.path.join(BASE_DIR, "data", "reports")
    os.makedirs(reports_dir, exist_ok=True)
    report_path = os.path.join(reports_dir, "report_trajet_01.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Rapport texte sauvegarde dans : {report_path}")

    # 2) export CSV
    if frames_metrics:
        metrics_dir = os.path.join(BASE_DIR, "data", "metrics")
        os.makedirs(metrics_dir, exist_ok=True)
        csv_path = os.path.join(metrics_dir, "metrics_trajet_01.csv")

        fieldnames = ["t", "dt", "lane_status", "distance_zone", "lane_offset", "distance_est", "risk_level", "risk_score", "mode"]

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in frames_metrics:
                writer.writerow({k: row.get(k) for k in fieldnames})

        print(f"Metriques par frame sauvegardees dans : {csv_path}")
    else:
        print("[CSV] Aucune metrique a exporter (frames_metrics vide).")

    # 3) fenêtre jolie
    try:
        show_report_window(report)
    except Exception as e:
        print(f"[REPORT] Impossible d'afficher la fenetre de rapport : {e}")

    # 4) graphes PNG
    if HAS_MPL and frames_metrics:
        print("[GRAPHS] Generation des graphes...")

        t_list = [m["t"] for m in frames_metrics]
        risk_list = [m.get("risk_score", 0) for m in frames_metrics]
        dist_list = [m.get("distance_est") for m in frames_metrics]

        figures_dir = os.path.join(BASE_DIR, "data", "figures")
        os.makedirs(figures_dir, exist_ok=True)

        # 1) risk score vs temps
        fig1 = plt.figure(figsize=(9, 4))
        plt.plot(t_list, risk_list, linewidth=2)
        plt.xlabel("Temps (s)")
        plt.ylabel("Risk score (/100)")
        plt.title("Evolution du score de risque")
        plt.grid(True, alpha=0.4)
        plt.tight_layout()
        fig1_path = os.path.join(figures_dir, "risk_score_vs_time.png")
        fig1.savefig(fig1_path, dpi=150, bbox_inches="tight")
        print(f"[GRAPHS] Sauvegarde : {fig1_path}")

        # 2) distance vs temps (valeurs non None)
        t_dist, d_dist = [], []
        for t_val, d_val in zip(t_list, dist_list):
            if d_val is not None:
                t_dist.append(t_val)
                d_dist.append(d_val)

        if t_dist:
            fig2 = plt.figure(figsize=(9, 4))
            plt.plot(t_dist, d_dist, linewidth=2)
            plt.xlabel("Temps (s)")
            plt.ylabel("Distance estimee (m)")
            plt.title("Distance estimee au vehicule principal")
            plt.grid(True, alpha=0.4)
            plt.tight_layout()
            fig2_path = os.path.join(figures_dir, "distance_vs_time.png")
            fig2.savefig(fig2_path, dpi=150, bbox_inches="tight")
            print(f"[GRAPHS] Sauvegarde : {fig2_path}")

        # 3) répartition statuts voie
        lane_counts = {"center": 0, "near_line": 0, "out_of_lane": 0}
        for m in frames_metrics:
            st = m.get("lane_status")
            if st in lane_counts:
                lane_counts[st] += 1

        labels = list(lane_counts.keys())
        values = [lane_counts[k] for k in labels]

        fig3 = plt.figure(figsize=(6, 4))
        plt.bar(labels, values)
        plt.title("Repartition du temps par statut de voie")
        plt.xlabel("Statut de voie")
        plt.ylabel("Nombre de frames")
        plt.tight_layout()
        fig3_path = os.path.join(figures_dir, "lane_status_distribution.png")
        fig3.savefig(fig3_path, dpi=150, bbox_inches="tight")
        print(f"[GRAPHS] Sauvegarde : {fig3_path}")

        if SHOW_GRAPHS:
            plt.show()
        else:
            plt.close("all")

    else:
        if not HAS_MPL:
            print("[GRAPHS] Matplotlib non installe => pas de graphes.")
        else:
            print("[GRAPHS] frames_metrics vide => pas de graphes.")


if __name__ == "__main__":
    main()
