import cv2
import os

# 1) Construire le chemin absolu vers la vidéo
# __file__ = chemin du fichier Python actuel (test_video_week1.py)
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # -> dossier DriveGuardianIA

video_path = os.path.join(BASE_DIR, "data", "raw_videos", "trajet_01_jour.mp4")

# Vérifier que le fichier existe
if not os.path.exists(video_path):
    raise FileNotFoundError(f"Vidéo introuvable : {video_path}")

print(f"Utilisation de la vidéo : {video_path}")

# 2) Ouvrir la vidéo avec OpenCV
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise RuntimeError("Impossible d'ouvrir la vidéo. Vérifie le chemin ou le format.")

# 3) Récupérer infos
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("=== Informations sur la vidéo ===")
print(f"Résolution : {width} x {height}")
print(f"FPS : {fps}")
print(f"Nombre de frames : {frame_count}")
if fps > 0:
    print(f"Durée approximative : {frame_count / fps:.1f} secondes")

# 4) Lire et afficher la vidéo frame par frame
while True:
    ret, frame = cap.read()  # ret = True si lecture OK, frame = image

    if not ret:
        print("Fin de la vidéo ou erreur de lecture.")
        break

    # Afficher la frame dans une fenêtre
    cv2.imshow("DriveGuardian - Test video", frame)

    # Attendre 1 ms et écouter la touche 'q' pour quitter
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("Sortie demandée par l'utilisateur (touche 'q').")
        break

# 5) Libérer les ressources
cap.release()
cv2.destroyAllWindows()
