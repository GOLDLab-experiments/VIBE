import cv2
from ultralytics import YOLO

class ObjectDetector:
    def __init__(self, model_name="yolov8l-oiv7.pt"):
        """
        Initializes the ObjectDetector with the YOLO model.
        """
        print(f"Loading model {model_name}...")
        self.model_name = model_name
        self.model = YOLO(self.model_name)
        self.labels = self.model.names

    def detect(self, image: cv2.Mat):
        """
        Runs inference on the given image to detect helmets.
        Returns the bounding box of the detected helmet or an empty list if no helmet is detected.
        """
        print("Running inference...")
        results = self.model.predict(image, conf=0.2, save=False, show=False, verbose=False)
        results = results[0].boxes.data.tolist()
        detections = []
        for detection in results:
            object, probability = self.labels[int(detection[5])], round(detection[4], 3)
            detections.append((object, probability))
        return detections    

def main():
    object_detector = ObjectDetector()

    # Capture image from camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    print("Press SPACE to capture an image, or ESC to exit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        cv2.imshow("Camera", frame)
        key = cv2.waitKey(1)
        if key % 256 == 27:  # ESC pressed
            print("Escape hit, closing...")
            break
        elif key % 256 == 32:  # SPACE pressed
            image = frame.copy()
            detections = object_detector.detect(image)
            print("Detections:", detections)
            # Optionally, show detections on the image
            for detection in detections:
                label, prob = detection
                cv2.putText(image, f"{label}: {prob}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.imshow("Detections", image)
            cv2.waitKey(0)
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

"""
Example output:

Loading model yolov8s-oiv7.pt...
Running inference...
Detections: [('Human face', 0.764), ('Woman', 0.524)]

Loading model yolov8l-oiv7.pt...
Running inference...
Detections: [('Remote control', 0.615), ('Headphones', 0.496), ('Pen', 0.387), ('Fashion accessory', 0.367), ('Sock', 0.276)]
"""
