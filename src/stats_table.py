import pandas as pd

csv_path = r"C:\Users\marou\Desktop\DriveGuardianIA\data\metrics\metrics_trajet_01.csv"

  # <-- mets ton fichier
df = pd.read_csv(csv_path)


COL_TIME = "time_s"        
COL_LANE = "lane_status"   # valeurs: centre/proche/hors (ou équivalent)
COL_RISK = "risk_level"    # SAFE/WARNING/DANGER
COL_NVEH = "num_vehicles"  # 0..3
COL_DIST = "distance_est"  # distance heuristique
COL_WAVW = "audio_warning" # 0/1 si dispo
COL_WAVD = "audio_danger"  # 0/1 si dispo

total_frames = len(df)

# durée
if COL_TIME in df.columns:
    duration = df[COL_TIME].iloc[-1] - df[COL_TIME].iloc[0]
else:
    duration = None

# répartition voie
lane_stats = df[COL_LANE].value_counts(dropna=False) if COL_LANE in df.columns else None

# épisodes WARNING/DANGER
def count_episodes(series, target):
    prev = None
    episodes = 0
    for v in series:
        if v == target and prev != target:
            episodes += 1
        prev = v
    return episodes

warning_ep = count_episodes(df[COL_RISK], "WARNING") if COL_RISK in df.columns else None
danger_ep  = count_episodes(df[COL_RISK], "DANGER")  if COL_RISK in df.columns else None

# distribution véhicules
veh_dist = df[COL_NVEH].value_counts().sort_index() if COL_NVEH in df.columns else None

# stats distance
dist_min = df[COL_DIST].min() if COL_DIST in df.columns else None
dist_mean = df[COL_DIST].mean() if COL_DIST in df.columns else None
dist_max = df[COL_DIST].max() if COL_DIST in df.columns else None

# alertes audio
warn_audio = int(df[COL_WAVW].sum()) if COL_WAVW in df.columns else None
dang_audio = int(df[COL_WAVD].sum()) if COL_WAVD in df.columns else None

print("Durée analysée:", duration if duration is not None else "a completer apres execution")
print("Nombre de frames:", total_frames)
print("WARNING episodes:", warning_ep)
print("DANGER episodes:", danger_ep)
print("Vehicules 0/1/2/3:", veh_dist.to_dict() if veh_dist is not None else "a completer")
print("Distance min/moy/max:", dist_min, dist_mean, dist_max)
print("Alertes warning/danger:", warn_audio, dang_audio)
print("Repartition voie:", (lane_stats/total_frames*100).to_dict() if lane_stats is not None else "a completer")
