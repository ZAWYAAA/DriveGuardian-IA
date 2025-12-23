=======
# DriveGuardian IA — Assistant d’aide à la conduite (Computer Vision / ADAS)

**DriveGuardian IA** est un prototype d’assistant d’aide à la conduite basé sur une dashcam.  
Il analyse une vidéo de trajet et fournit en temps réel des indicateurs de trajectoire, de distance et de risque, puis génère un bilan en fin de trajet.

> ⚠️ Projet pédagogique (V1) : il **assiste** le conducteur et **ne remplace pas** une conduite responsable ni un système ADAS certifié.

---

## 🎥 Vidéo démo du projet (résultat)
- Démonstration (mode DEMO) :  
  - YouTube : https://youtu.be/HJbUsmukgjA

---

## ✅ Fonctionnalités

- **Détection de voie** (Canny + ROI + Transformée de Hough)
  - Statuts : `center` / `near_line` / `out_of_lane`
  - Décalage latéral (offset) + lissage temporel
- **Détection multi-véhicules** (jusqu’à 3) + pseudo-radar `left/center/right`
- **Estimation heuristique de distance** + zones : `safe` / `close` / `very_close`
- **Détection approximative des clignotants** sur véhicules proches
- **Analyse de risque** : `SAFE` / `WARNING` / `DANGER` + score `0–100`
- **Alertes audio intelligentes** (`warning.wav`, `danger.wav`)
  - anti-spam (cooldown), priorité danger, options ON/OFF séparées WARNING & DANGER
- **Dashboard temps réel** (OpenCV) + texte de contexte “smart”
- **Fin de trajet** : export **CSV**, graphes **PNG**, rapport texte + fenêtre “bilan + recommandations”

---

## 🧠 Architecture (résumé)

Pipeline principal :
1. Lecture vidéo (OpenCV)
2. Détection voie → offset + statut
3. Détection véhicules → position + distance (heuristique) + zone
4. Fusion (règles + pondérations) → niveau de risque + score + contexte
5. Alertes audio (priorité danger + anti-spam)
6. Fin de trajet → exports (CSV + figures + rapport)

---

## 🧰 Prérequis

- Python 3.11.1
- Windows recommandé (audio via `winsound` / dépendances Windows)
- Dépendances (installées via `requirements.txt`) :
  - `opencv-python`, `numpy`
  - `matplotlib` (pour générer les graphes PNG)
  - dépendances audio Windows (ex : `pyttsx3`, `pywin32`)

### Fichiers requis
Assurez-vous que ces éléments existent :
- `data/models/cars.xml` (cascade Haar véhicules)
- `data/audio/warning.wav` et `data/audio/danger.wav`
- une vidéo dans `data/raw_videos/` (ou adapter `video_path` dans le script)

---

## 🚀 Installation (Windows / PowerShell)

```powershell
git clone https://github.com/M-Zaouia/DriveGuardian-IA
cd DriveGuardian-IA
python -m venv venv
.\venv\Scripts\Activate
pip install -r requirements.txt
python -c "import cv2, numpy; import matplotlib.pyplot as plt; print('OK: OpenCV / NumPy / Matplotlib')"


>>>>>>> 12f24a9 (Update project files)

