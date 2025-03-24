import os
import bz2
import urllib.request

url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
compressed_path = "shape_predictor_68_face_landmarks.dat.bz2"
output_path = "shape_predictor_68_face_landmarks.dat"

if not os.path.exists(output_path):
    print("Téléchargement du modèle Dlib...")
    urllib.request.urlretrieve(url, compressed_path)
    print("Décompression...")
    with bz2.BZ2File(compressed_path) as fr, open(output_path, "wb") as fw:
        fw.write(fr.read())
    os.remove(compressed_path)
    print("Modèle prêt !")
else:
    print("Modèle déjà présent.")
