#setting up the necessary libraries
import mediapipe as mp
import numpy as np
import pandas as pd
import cv2
import os
import warnings
warnings.filterwarnings('ignore')
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic
import csv


#directories to the necessary files
running_path = r'C:\Users\stanl\OneDrive\Desktop\Pose Estimation\second pose estimation model\running'
walking_path = r'C:\Users\stanl\OneDrive\Desktop\Pose Estimation\second pose estimation model\walking'


#function code to extract the different videos
def get_videos(running_path):
    video_paths = []
    for dirName, subdirList, fileList in os.walk(running_path):
        for filename in fileList:
            if ".mp4" in filename.lower():
                video_paths.append(os.path.join(dirName, filename))
    return video_paths


#fuction call to the different video directories
running_videos = get_videos(running_path)
walking_videos = get_videos(walking_path)

#quick test to see if the directories have been properly cap
print(len(running_videos)), print(len(walking_videos))

#capture landmarks and export csv files
class_name = 'walking'
csv_filename = 'walking(2).csv'

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    for video_path in walking_videos:
        # Initialize VideoCapture for the current video
        cap = cv2.VideoCapture(video_path)

        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                print(f"Finished processing {video_path}")
                break

            # Rest of your processing code remains the same
            # recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make detection
            results = holistic.process(image)

            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Pose Detection
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                     mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                                     mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                     )

            # Export coordinates to a CSV file for each video
            try:
                # Extract Pose landmarks
                pose = results.pose_landmarks.landmark
                pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())

                # Concatenate rows
                row = pose_row

                # Append class name
                row.insert(0, class_name)

                # Export to CSV for each video
                with open(csv_filename, mode='a', newline='') as f:
                    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    csv_writer.writerow(row)

            except:
                pass

        # Release video capture for the current video
        cap.release()
        # cv2.destroyAllWindows()