from fer import FER
import cv2

detector = FER()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Camera not working")
        break

    result = detector.detect_emotions(frame)

    for face in result:
        (x, y, w, h) = face["box"]
        emotions = face["emotions"]

        emotion_name = max(emotions, key=emotions.get)

        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame, emotion_name, (x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

    cv2.imshow("Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()