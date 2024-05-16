from flask import Flask, render_template, Response
from flask_cors import CORS
import cv2
import re
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
# import crunches

app = Flask(__name__)
CORS(app)



def calculate_angle(a,b,c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle>180.0:
        angle = 360-angle
        
    return angle

# Define your function
def my_function():
    print('Button clicked!')
    
    cap = cv2.VideoCapture(0)  # 0 is for webcam
    counter = 0
    stage = None
    with mp_pose.Pose(min_detection_confidence = 0.5, min_tracking_confidence=0.9) as pose:

        while cap.isOpened():
            ret, frame = cap.read()
        
        ##Detect stuff and render
        
        # Recolor image from BGR to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
        
        # Make detections
            results = pose.process(image)
        
        #Recolor back to BGR bcuz openCV wants it in BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Extract Landmarks
            try:
                landmarks = results.pose_landmarks.landmark
            
            # Get Co-ordinates
                shoulderLeft = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbowLeft = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wristLeft = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
          
          # Calculate Angle
                angleLeft = calculate_angle(shoulderLeft, elbowLeft , wristLeft)
            
            # Visualize
                cv2.putText(image, str(angleLeft),
                       tuple(np.multiply(elbowLeft,[640,480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,5,255), 2, cv2.LINE_AA
                            )
            
            # Curl Counter Logic
                if angleLeft > 150:
                    stage = "down"
                if angleLeft <60 and stage == 'down':
                    stage = 'up'
                    counter+=1
                    print(counter)
             
            except:
                pass
      
            cv2.rectangle(image, (0,0), (225,73),(245,117,16), -1)
        
        # Rep data
            cv2.putText(image, 'REPS',(15,12),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        
            cv2.putText(image, str(counter),
                    (10,60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
        
        # Stage data
            cv2.putText(image, 'STAGE',(65,12),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        
            cv2.putText(image, stage,
                    (60,60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
        
        # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                         mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2),
                                         mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=2))
        
        
            cv2.imshow('Mediapipe Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    

    # crunches.calculate_angle()
    # Your function logic goes here

# Define a route to call the function



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/lunges.html')
def lunges():
    return render_template('lunges.html')

@app.route('/biceps.html')
def biceps():
    return render_template('biceps.html')

@app.route('/shoulder.html')
def shoulder():
    return render_template('shoulder.html')

@app.route('/crunches.html')
def crunches():
    return render_template('crunches.html')


@app.route('/call-my-function')
def call_my_function():
    my_function()
    return 'Function called successfully'

# Start the server
if __name__ == '__main__':
    app.run(debug=True, port=5001)