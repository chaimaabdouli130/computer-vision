import cv2 
import cvzone 
from cvzone.FaceMeshModule import FaceMeshDetector 
from cvzone.HandTrackingModule import HandDetector 
Hand_Detector = HandDetector()
MeshDetector = FaceMeshDetector()
video = cv2.VideoCapture(r"C:\Users\chaim\Desktop\computer vision\854004-hd_1920_1080_30fps.mp4")
ret, image = video.read()
if not ret:
    print("Erreur : vidéo introuvable ou vide !")

point11X = point11Y = 0
while True : 
    ret , image = video.read()
    img , faces = MeshDetector.findFaceMesh(image , draw=True)
    results = Hand_Detector.findHands(image , draw=True )
   
    if cv2.waitKey(10) == ord('q'):
        break 
  if len (results[0]) > 0 :
        #print(results[0][0])
        landmarks = results[0][0]['lmList']
        point11X , point11Y = landmarks[11][0] , landmarks[11][1]
        print(point11X , point11Y)
    point14 = faces[0][14] # lip point #mn hne ybda code wjh w smoking
    distance = MeshDetector.findDistance((point11X ,point11Y) , (point14[0],point14[1]))[0]
    print(distance)
    if distance < 40 :
        cvzone.putTextRect(image ,'Smoking' , (30,50))
    else :
        cvzone.putTextRect(image , 'No smoking' , (30,50))#youfa hne
    cv2.imshow('video' , image)
cv2.destroyAllWindows()
    