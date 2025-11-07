import cv2

# Charger le classifieur Haar
faceCascade = cv2.CascadeClassifier(
    "C:/Users/chaim/anaconda3/envs/myenv/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml"
)

# Ouvrir la webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Impossible d'accéder à la caméra")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir en gris pour la détection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Détecter les visages
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Dessiner les rectangles
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)

    # Afficher dans UNE SEULE fenêtre
    cv2.imshow('Détection de visages', frame)

    # Quitter avec 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer ressources
cap.release()
cv2.destroyAllWindows()

cv2.waitKey(0)
