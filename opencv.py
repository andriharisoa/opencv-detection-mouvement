import cv2
import datetime
import os

# Nom du dossier pour enregistrer les captures
dossier_captures = "captures"

# Crée le dossier s’il n’existe pas déjà
if not os.path.exists(dossier_captures):
    os.makedirs(dossier_captures)

cap = cv2.VideoCapture(0)
backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

cv2.namedWindow("Surveillance")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    fgMask = backSub.apply(frame)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mouvement_detecte = False

    for contour in contours:
        if cv2.contourArea(contour) > 1500:
            mouvement_detecte = True
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if mouvement_detecte:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        nom_image = os.path.join(dossier_captures, f"capture_{timestamp}.png")
        cv2.imwrite(nom_image, frame)
        print(f"Mouvement détecté ! Image sauvegardée : {nom_image}")

    cv2.imshow("Surveillance", frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

