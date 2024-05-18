from flask import Flask, render_template, Response
from flask_cors import CORS
import cv2
import re
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


app = Flask(__name__)




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
def count_biceps():
    print('Button clicked!')
    
    cap = cv2.VideoCapture(0)  # 0 is for webcam
    counter = 0
    stage = None
    with mp_pose.Pose(min_detection_confidence = 0.5, min_tracking_confidence=0.9) as pose:

        while cap.isOpened():
            ret, frame = cap.read()
        
        ##Detect stuff and render
            try:
        # Recolor image from BGR to RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
            
            # Make detections
                results = pose.process(image)
            
            #Recolor back to BGR bcuz openCV wants it in BGR
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

###################1
            except cv2.error as e:
                print("4")
                cap.release()
                cv2.destroyAllWindows()
                return f"Error processing frame: {str(e)}", 500
##########2
        # Extract Landmarks
            try:
                landmarks = results.pose_landmarks.landmark
            
 ######
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

        cap.release()
        cv2.destroyAllWindows()

    return "biceps exercise processing completed"







## crunches
def count_crunches():
    # VIDEO FEED
    cap = cv2.VideoCapture(0)  # 0 is for webcam

    rep_count = 0
    states = {
                    1:[130,180],
                    2:[30,130]
                }
    # reset = 1
    pattern ='1'
    state = 1
    main_string = "121"
    regex_pattern = ''.join([re.escape(char) + r'+' for char in main_string])

    ## SETUP MEDIAPIPE INSTANCE
    with mp_pose.Pose(min_detection_confidence = 0.5, min_tracking_confidence=0.9) as pose:

        while cap.isOpened():
            ret, frame = cap.read()
            
#################1
            if not ret:
                break
###########2
            ##Detect stuff and render
            
            try: 

            # Recolor image from BGR to RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                
                # Make detections
                results = pose.process(image)
                
                #Recolor back to BGR bcuz openCV wants it in BGR
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
###################1
            except cv2.error as e:
                print("4")
                cap.release()
                cv2.destroyAllWindows()
                return f"Error processing frame: {str(e)}", 500
##########2            
            
            # Extract Landmarks
            try:
                landmarks = results.pose_landmarks.landmark
                
                # Get Co-ordinates
                shoulderLeft = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                hipLeft = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                kneeLeft = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                
                
                # Calculate Angle
                left_angle = calculate_angle(shoulderLeft, hipLeft , kneeLeft)
                
                # Visualize
                cv2.putText(image, str(left_angle),
                        tuple(np.multiply(hipLeft,[640,480]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,5,255), 2, cv2.LINE_AA
                                )
                
                                
                if (states[1][0] <= left_angle <= states[1][1]):
                    state = '1'
                elif (states[2][0] <= left_angle <= states[2][1]):
                    state = '2'

                pattern += state
                    
                if re.fullmatch(regex_pattern, pattern):
    #                 reset = 1
                    rep_count += 1
                    pattern = ''
                
            
            except:
                pass
            
            # Render curl counter
            # setup status box
            cv2.rectangle(image, (0,0), (225,73),(245,117,16), -1)
            
            # Rep data
            cv2.putText(image, 'REPS',(15,12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            
            cv2.putText(image, str(rep_count),
                        (10,60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
            
            # Stage data
            cv2.putText(image, 'STAGE',(65,12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            
            cv2.putText(image, 'STAGE',
                        (60,60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
            
            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                            mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2),
                                            mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=2))
            
            
            cv2.imshow('Mediapipe Feed', image)
            
            
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    return "crunches exercise processing completed"




def count_lunges():
    # VIDEO FEED
    cap = cv2.VideoCapture(0)  # 0 is for webcam

    # CURL COUNT
    rep_count = 0
    states = {
                    1:[160,180],
                    2:[110,160],
                    3:[80,110]
                }
    # phase = 'decrease'
    reset = 1
    pattern =''
    main_string = "232"
    regex_pattern = ''.join([re.escape(char) + r'+' for char in main_string])

    ## SETUP MEDIAPIPE INSTANCE
    with mp_pose.Pose(min_detection_confidence = 0.5, min_tracking_confidence=0.9) as pose:


        while cap.isOpened():
            ret, frame = cap.read()
            
            ##Detect stuff and render
#################1
            if not ret:
                break
###########2
            # Recolor image from BGR to RGB
            try:

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                
                # Make detections
                results = pose.process(image)
                
                #Recolor back to BGR bcuz openCV wants it in BGR
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
###################1
            except cv2.error as e:
                print("4")
                cap.release()
                cv2.destroyAllWindows()
                return f"Error processing frame: {str(e)}", 500
##########2
            # Extract Landmarks
            try:
                landmarks = results.pose_landmarks.landmark
                
                # Get Co-ordinates for right side
                right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                
                
                # Get Co-ordinates for left side
                left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                
                # Calculate Angle
                
                right_angle = calculate_angle(right_hip, right_knee , right_ankle)
                left_angle = calculate_angle(left_hip, left_knee , left_ankle)
                
                # Visualize
                cv2.putText(image, str(right_angle),
                        tuple(np.multiply(right_knee,[640,480]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,5,255), 2, cv2.LINE_AA
                                )
                cv2.putText(image, str(left_angle),
                        tuple(np.multiply(left_knee,[640,480]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,5,255), 2, cv2.LINE_AA
                                )
                
    #             state = '1'
                if (states[1][0] <= right_angle <= states[1][1]) and (states[1][0] <= left_angle <= states[1][1]):
                    state = '1'
                elif (states[2][0] <= right_angle <= states[2][1]) and (states[2][0] <= left_angle <= states[2][1]):
                    state = '2'
                elif (states[3][0] <= right_angle <= states[3][1]) and (states[3][0] <= left_angle <= states[3][1]):
                    state = '3'
                    
                
                
                
                reset = 0
                if state == '1':
                    reset = 1
                    count = 0 
                    pattern = ''
                    
                    
                else:
                    pattern += state
    
                    
                if re.fullmatch(regex_pattern, pattern):
                    reset = 1
                    rep_count += 1
                    pattern = ''
                    
                
                
            except:
                pass
            
            cv2.rectangle(image, (0,0), (225,73),(245,117,16), -1)
            
            # Rep data
            cv2.putText(image, 'REPS',(15,12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            
            cv2.putText(image, str(rep_count),
                        (10,60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
            
            # Stage data
            cv2.putText(image, 'STAGE',(65,12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            
            cv2.putText(image, 'STAGE',
                        (60,60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
            
            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                            mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2),
                                            mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=2))
            
            
            cv2.imshow('Mediapipe Feed', image)

            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
    


        cap.release()
        cv2.destroyAllWindows()
    return "Lunges exercise processing completed"





def count_shoulder():
    cap = cv2.VideoCapture(0)  # 0 for the default camera

    rep_count = 0

    states = {
                    1:[0,90],
                    2:[90,120],
                    3:[120,160],
                    4:[160,180]
                }
    
    reset = 0
    pattern =''
    main_string = "23432"
    regex_pattern = ''.join([re.escape(char) + r'+' for char in main_string])

    ## SETUP MEDIAPIPE INSTANCE
    with mp_pose.Pose(min_detection_confidence = 0.5, min_tracking_confidence=0.9) as pose:

#################1
        if not cap.isOpened():
            return "Error: Could not open video capture", 500
###########2
        while True:
            ret, frame = cap.read()
#################1
            if not ret:
                break
###########2

            # Process the frame (e.g., convert color)
################1
            try:
##########2
                print("1")
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                
                print("2")
                # Make detections
                results = pose.process(image)
                
                #Recolor back to BGR bcuz openCV wants it in BGR
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                print("3")
###################1
            except cv2.error as e:
                print("4")
                cap.release()
                cv2.destroyAllWindows()
                return f"Error processing frame: {str(e)}", 500
##########2
            try:
                landmarks = results.pose_landmarks.landmark
                
                # Get Co-ordinates
                left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                
                
                # Calculate Angle
                
                angle = calculate_angle(left_elbow, left_shoulder , left_hip)
                
                # Visualize
                cv2.putText(image, str(angle),
                        tuple(np.multiply(left_shoulder,[640,480]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,5,255), 2, cv2.LINE_AA
                                )
                
                if states[1][1] > angle >= states[1][0]:
                    state = '1'
                elif states[2][1] >= angle >= states[2][0]:
                    state = '2'
                elif states[3][1] >= angle >= states[3][0]:
                    state = '3'
                elif states[4][1] >= angle >= states[4][0]:
                    state = '4'
                
                reset = 0
                if state == '1':
                    reset = 1
                    count = 0
                    pattern = ''
                    
                    
                else:
                    pattern += state
    #                 reset=0
    #             if reset == 0 and count<4 :
    #                 pattern = pattern+state
    #                 count +=1


                    
                if re.fullmatch(regex_pattern, pattern):
                    reset = 1
                    rep_count += 1
                    pattern = ''
                    
                
                
            except:
                pass
            
            # Render curl counter
            # setup status box
            cv2.rectangle(image, (0,0), (225,73),(245,117,16), -1)
            
            # Rep data
            cv2.putText(image, 'REPS',(15,12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            
            cv2.putText(image, str(rep_count),
                        (10,60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
            
            # Stage data
            cv2.putText(image, 'STAGE',(65,12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            
            cv2.putText(image, 'STAGE',
                        (60,60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
            
            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                            mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2),
                                            mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=2))
            
            # Display the frame
            cv2.imshow('Shoulder Exercise', image)
##fffffff#
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

###################1
    return "Shoulder exercise processing completed"
###########2
@app.route('/')
def home():
    return render_template('index.html')

# @app.route('/')
# def index():
#     return render_template('index.html')


@app.route('/login')
def login():
    return render_template('login.html')

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


# @app.route('/call-my-function')
# def call_my_function():
#     biceps()
#     return 'Function called successfully'

@app.route('/start-biceps')
def start_biceps():
    return count_biceps()
    
@app.route('/start-shoulder')
def start_shoulder():
    return count_shoulder()

@app.route('/start-crunches')
def start_crunches():
    return count_crunches()
    

@app.route('/start-lunges')
def start_lunges():
    return count_lunges()
    


# Start the server
if __name__ == '__main__':
    app.run(debug=True, port=5001)