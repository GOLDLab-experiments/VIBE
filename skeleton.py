import cv2
import mediapipe as mp
import numpy as np

class SkeletonDetector:
    def __init__(self):
        """
        Initializes the SkeletonDetector with MediaPipe Pose and other configurations.
        Only keeps essential keypoints for mood/posture recognition.
        """
        print("Initializing SkeletonDetector...")
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose()
        # Only keep essential keypoints for mood/posture
        self.keypoints = [
            'nose',
            'left_eye', 'right_eye',
            'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder',
            'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist',
            'left_hip', 'right_hip',
            'left_knee', 'right_knee',
            'left_ankle', 'right_ankle'
        ]
        # Indices of the above keypoints in the original MediaPipe list
        self.keypoint_indices = [
            0,    # nose
            2, 5, # left_eye, right_eye
            7, 8, # left_ear, right_ear
            11, 12, # left_shoulder, right_shoulder
            13, 14, # left_elbow, right_elbow
            15, 16, # left_wrist, right_wrist
            23, 24, # left_hip, right_hip
            25, 26, # left_knee, right_knee
            27, 28  # left_ankle, right_ankle
        ]

    def detect(self, image: np.ndarray):
        """
        Detects essential skeleton landmarks in the given image.
        Returns a list of (keypoint, (x, y), visibility) for each detected keypoint, or an empty list if none found.
        """
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_image)
        detections = []
        if results.pose_landmarks:
            height, width, _ = image.shape
            for idx, kp_idx in enumerate(self.keypoint_indices):
                landmark = results.pose_landmarks.landmark[kp_idx]
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                visibility = landmark.visibility
                keypoint = self.keypoints[idx]
                if visibility > 0.5:  # Only consider keypoints with visibility above a threshold
                    detections.append((keypoint, round(visibility, 3)))
        return detections

    def draw(self, image: np.ndarray):
        """
        Draws the skeleton on the image and returns the image.
        """
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_image)
        output_image = image.copy()
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                output_image,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=3),
                self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=3)
            )
        return output_image

def main():
    detector = SkeletonDetector()
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
            detections = detector.detect(image)
            print("Skeleton keypoints:", detections)
            output_image = detector.draw(image)
            cv2.imshow("Skeleton", output_image)
            cv2.waitKey(0)
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()