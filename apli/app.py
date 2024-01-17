import os

from ultralytics import YOLO
import cv2

# Charger le modèle
model = YOLO("../runs/detect/train/weights/best.pt")

image_path = "./image_test"

# Exécuter la détection
results = model.predict(image_path)


for result in results:
    # Les boîtes sont dans le format (xmin, ymin, xmax, ymax)
    boxes = result.boxes.xyxy.cpu().tolist()
    
    # Les classes sont également dans la liste
    clss = result.boxes.cls.cpu().tolist()
    
    for box, cls in zip(boxes, clss):
        # Découper l'objet de l'image
        crop_obj = result.orig_img[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
        
        # Extraire le nom du fichier à partir du chemin du fichier
        filename = os.path.basename(result.path)
        
        # Sauvegarder l'objet découpé
        cv2.imwrite(f"./image_result/{model.names[int(cls)]}_{filename}", crop_obj)

    for box, cls in zip(boxes, clss):
        cv2.rectangle(result.orig_img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
        # Convertir l'image en RGB pour l'afficher avec matplotlib
        img_rgb = cv2.cvtColor(result.orig_img, cv2.COLOR_BGR2RGB)

    cv2.imwrite(f"./image_result/{filename}",img_rgb)
