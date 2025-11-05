import cv2

cap = cv2.VideoCapture(0)
backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    fgMask = backSub.apply(frame)
    
    # Nettoyage
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Détection de mouvement
    for contour in contours:
        if cv2.contourArea(contour) > 1000:  # Ajuste selon la taille de l'objet
            print("Mouvement précis détecté!")
            break
    
    cv2.imshow("Detection", frame)
    
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

