import cv2
import math
from ultralytics import YOLO

class GlassesDetector:
    def __init__(self, model_path="weights/best.pt"):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 640)
        self.cap.set(4, 480)
        self.model = YOLO(model_path)
        self.class_names = ["Gozluk", "Gunes Gozlugu"]

    def detect_glasses(self):
        while True:
            success, img = self.cap.read()
            results = self.model(img, stream=True)

            for r in results:
                boxes = r.boxes

                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                    confidence = math.ceil((box.conf[0] * 100)) / 100
                    print("Confidence --->", confidence)

                    cls = int(box.cls[0])
                    print("Class name -->", self.class_names[cls])

                    org = [x1, y1]
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 1
                    color = (255, 0, 0)
                    thickness = 2
                    cv2.putText(img, f"{self.class_names[cls]} %{confidence}", org, font, font_scale, color, thickness)

            cv2.imshow('Glass-Sunglass Detection', img)
            if cv2.waitKey(1) == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = GlassesDetector()
    detector.detect_glasses()