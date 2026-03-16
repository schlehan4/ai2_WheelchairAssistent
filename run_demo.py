import cv2
import torch
import math
from ultralytics import YOLO
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"

model = YOLO("model/custom_trained_yolov8s.pt").to(device)
classNames = ["sidewalk"]
video_path = "videos/rollicam7.mp4"
csv_file = "data/Can_Data7.csv"

jRL_diff_threshold = 550; # y_max / 2

# for next frame press 'n'
frame_by_frame_mode = False

weight_heuristic = 0.6

diagram_window_size = 30 # seconds

 # Lese die CSV-Datei ein
df = pd.read_csv(csv_file)

# Calculate arrow direction based on current time in seconds
def calculate_arrow_direction(frame_count):
    # Placeholder for arrow direction calculation based on time
    # You can implement your logic to calculate the arrow direction based on time here
    # For example, you could use the current time to determine the direction

    # Suche den nächstgelegenen Zeitwert in der Spalte 'Time' in der CSV-Datei
    #nearest_time_row = df.iloc[(df['Time']-current_time).abs().argsort()[:1]]
    # Extrahiere den Wert aus der 'value'-Spalte des gefundenen Zeilenindex
    #jRL = nearest_time_row['Joystick_Right_Left'][0]
    #jFr = nearest_time_row['Joystick_Front_Rear'][0]

    row_index=frame_count
    nearest_time_row = df.iloc[row_index]
    trackWidth=0.468
    kmh2ms=3.6
    dt=1
    speedRight = nearest_time_row['Speed_Right']
    speedLeft = nearest_time_row['Speed_Left']
    vLinear = np.array( [1/2*(speedLeft+speedRight)/kmh2ms, 0] ) #vektor mit linearer geschwindigkeit auf xachse
    if vLinear[0]!=0 :
        omega = 1/trackWidth*(speedRight-speedLeft)/kmh2ms
        alpha = np.tan(vLinear[1]/vLinear[0])+omega*dt
    else:
        alpha=0
    direction = np.array([np.cos(alpha), np.sin(alpha)])
    arrowLength = 1

    return  (100*direction, arrowLength)

def get_raw_values_direction(current_time, frame_count):
    row_index = frame_count
    
    # Extrahiere die entsprechende Zeile
    nearest_time_row = df.iloc[row_index]
    
    # Extrahiere den Wert aus der 'value'-Spalte
    jRL = nearest_time_row['Joystick_Right_Left']
    jFr = nearest_time_row['Joystick_Front_Rear']
    speedRight = nearest_time_row['Speed_Right']
    speedLeft = nearest_time_row['Speed_Left']
    
    jRL_last1 = 0
    jRL_last2 = 0
    
    if (row_index >= 1):
        jRL_last1 = df.iloc[row_index - 1]['Joystick_Right_Left']
        
    if (row_index >= 2):
        jRL_last2 = df.iloc[row_index - 2]['Joystick_Right_Left']

    
    return f"({jRL}, {jFr}, {speedLeft}, {speedRight}, {jRL_last1}, {jRL_last2})"

def get_raw_values_and_last_two_RL(current_time, frame_count):
    #row_index = get_row_index_by_frame()
    row_index = frame_count
    
    # Extrahiere die entsprechende Zeile
    nearest_time_row = df.iloc[row_index]
    
    # Extrahiere den Wert aus der 'value'-Spalte
    t = nearest_time_row['Time']
    jRL = nearest_time_row['Joystick_Right_Left']
    jFr = nearest_time_row['Joystick_Front_Rear']
    speedRight = nearest_time_row['Speed_Right']
    speedLeft = nearest_time_row['Speed_Left']
    
    jRL_last1 = 0
    jRL_last2 = 0
    
    if (row_index >= 1):
        jRL_last1 = df.iloc[row_index - 1]['Joystick_Right_Left']
        
    if (row_index >= 2):
        jRL_last2 = df.iloc[row_index - 2]['Joystick_Right_Left']

    return t, jRL, jFr, speedRight, speedLeft, jRL_last1, jRL_last2

def joystick_anomaly_heuristic(current_time, threshold, frame_count):
    time, jRL, jFR, speedL, speedR, jRL_last1, jRL_last2 = get_raw_values_and_last_two_RL(current_time, frame_count)
    # z = |x(i) - x(i-1)| + |x(i-1) - x(i-2)|
    
    z = abs(jRL - jRL_last1) + abs(jRL_last1 - jRL_last2)
    
    # calculates values between 0 and 1 representing confidence of anomaly resp. unintended LR movement
    # d = decicion, x = input, t = threshold, c = range_within_threshold
    #d = 1 / ( 1 + e^-( (x - t) / (t * c) ) )
    score = 1 / ( 1 + math.exp(-( (z - threshold) / (threshold * 0.15) ) ) )
    
    return score
    

def calculate_prediction_value(current_time, sidewalk_confidence, frame_count):
    anomaly_score = joystick_anomaly_heuristic(current_time, jRL_diff_threshold, frame_count)

    prediction = weight_heuristic * anomaly_score + (1-weight_heuristic) * (1-sidewalk_confidence)

    # just some traffic light decision
    if prediction < 0.4:
        traffic_light_value = 0
    elif prediction >= 0.4 and prediction < 0.6:
        traffic_light_value = 1
    else:
        traffic_light_value = 2

    return (prediction, traffic_light_value, sidewalk_confidence, anomaly_score) # (prediction, traffic light 0,1,2)


### START OF CV2 STUFF FOR DISPLAYING FRAMES ###

def showDirectionFrame(current_time, frame_count):
    direction, arrow_length = calculate_arrow_direction(frame_count)
    center = (150, 150)  # Center of the arrow window
    endpoint = (center[1]-int(direction[1]), center[0]-int(direction[0]))
    arrow_frame = np.zeros((160, 300, 3), dtype=np.uint8)  # Create black frame for arrow
    cv2.arrowedLine(arrow_frame, center, endpoint, (0, 255, 0), 5)

    # Add text "direction graph"
    org_text = (10, 30)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (255, 255, 255)  # White color
    thickness = 2
    cv2.putText(arrow_frame, 'Direction Graph', org_text, font, fontScale, color, thickness)

    
    cv2.putText(arrow_frame, get_raw_values_direction(current_time, frame_count), (10, 50), font, 0.5, color, 1)
    cv2.imshow("Direction graph", arrow_frame)

def showPercentageAndTrafficLight(current_time, sidewalk_confidence, frame_count):
    prediction = calculate_prediction_value(current_time, sidewalk_confidence, frame_count)
    
    traffic_borders = [2,2,2]
    traffic_borders[prediction[1]] = -1 # set circle full

    # Create a black frame for the window
    frame = np.zeros((330, 210, 3), dtype=np.uint8)

    # Add text "direction graph"
    org_text = (50, 30)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (255, 255, 255)  # White color
    thickness = 2
    cv2.putText(frame, 'Output', org_text, font, fontScale, color, thickness)

    # Draw percentage text
    org_text = (40, 80)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (255, 255, 255)  # White color
    thickness = 1
    
    #cv2.putText(frame, f'{prediction[0]}%', org_text, font, fontScale, color, thickness)
    cv2.putText(frame, "{:.5f}".format(prediction[0]), org_text, font, fontScale, color, thickness)

    # Draw traffic light
    cv2.circle(frame, (100, 140), 30, (0, 255, 0), traffic_borders[0]) 
    cv2.circle(frame, (100, 210), 30, (0,134,209), traffic_borders[1]) 
    cv2.circle(frame, (100, 280), 30, (0, 0, 255), traffic_borders[2]) 

    cv2.imshow('Model prediction', frame)
    return prediction

def showSidewalkConfidence(confidence):
    # Create a black frame for the window
    frame = np.zeros((200, 400, 3), dtype=np.uint8)

    # Draw sidewalk confidence text
    org_text = (10, 70)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.6
    color = (255, 255, 255)  # White color
    thickness = 1
    confidence_formatted = "{:.5f}".format(confidence)
    cv2.putText(frame, f'object detection: {confidence_formatted}', org_text, font, fontScale, color, thickness)

    # Draw title
    fontScale = 1
    thickness = 2
    org_title = (10, 30)
    cv2.putText(frame, 'Sidewalk Confidence', org_title, font, fontScale, color, thickness)

    cv2.imshow('sidewalk confidence', frame)

def showBoxesOnVideo(results, frame, classNames):

    # coordinates
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

            # put box in cam
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # confidence
            confidence = math.ceil((box.conf[0]*100))/100
            print("Confidence --->",confidence)

            # class name
            cls = int(box.cls[0])
            print("Class name -->", classNames[cls])

            # object details
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_DUPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2

            cv2.putText(frame, classNames[cls], org, font, fontScale, color, thickness)

def getConfidenceForSidewalk(results):
    max_conf = 0
    max_conf_coordinates = []
    # Durch die Ergebnisse iterieren
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Wenn die Klasse "sidewalk" ist
            if classNames[int(box.cls[0])] == "sidewalk":
                # Die Konfidenz für den sidewalk zurückgeben
                if box.conf[0] > max_conf:
                    max_conf = box.conf[0].cpu()
                    max_conf_coordinates = box.xyxy[0]
    return (max_conf, max_conf_coordinates)

def showVideo():
    cap = cv2.VideoCapture(video_path)

    frame_width = int(cap.get(3)) 
    frame_height = int(cap.get(4)) 
   
    size = (frame_width, frame_height) 

    # video capturing disabled
    #writer = cv2.VideoWriter('video_output.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 25, size) 

    frame_rate = 10  # framerate video to control speed + some adjustment value, needs to be played with :'(
    delay = 1 / frame_rate

    frame_count_offset = -3 # there might is more video than data, let's skip the first 3 video frames
    frame_count = frame_count_offset
    

    # plot shizle
        
    plt.ion()
    
    timeline = []
    plotarr = []

    # here we are creating sub plots
    figure, ax = plt.subplots(figsize=(5,3))
    line1, = ax.plot(timeline, plotarr)
    plt.title('Joystick Right-Left')
    plt.xlabel('t [s]')
    
    
    sidewalk_scores = []
    heuristic_scores = []
    overall_scores = []
    
    figure2, ax2 = plt.subplots(figsize=(5,3))
    line2, = ax2.plot(timeline, heuristic_scores,label='Heuristics Score', color='blue')
    line3, = ax2.plot(timeline,sidewalk_scores,label='Sidewalk Score', color='green')
    line4, = ax2.plot(timeline,overall_scores, label='Model Score',color='orange')
    plt.title('Model Scores')
    plt.xlabel('t [s]')
    plt.legend()
    
    windowsize = diagram_window_size
    
    ax.set_xlim([0,windowsize])
    ax.set_ylim([-1100,1100])
    
    
    ax2.set_xlim([0,windowsize])
    ax2.set_ylim([0,1])

    start_time = time.time()
    sidewalk_confidences_last = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    while True:
        ret, frame = cap.read()
        results = model(frame, stream=False)

        showBoxesOnVideo(results, frame, classNames)

        if not ret:
            # Wenn das Video zu Ende ist, setze den Cursor zurück auf den Anfang
            cap.release()
            cap = cv2.VideoCapture(video_path)
            start_time = time.time()
            frame_count = frame_count_offset
            
            timeline = []
            plotarr = []
            sidewalk_scores = []
            heuristic_scores = []
            overall_scores = []
            
            ax.set_xlim([0,windowsize])
            ax2.set_xlim([0,windowsize])
            continue

        # Display video
        cv2.imshow('Video', frame)

        # Write video frame
        #writer.write(frame)

        # Calculate current time in seconds
        current_time = time.time() - start_time
        
        sidewalk_confidence = getConfidenceForSidewalk(results)
        sidewalk_confidences_last[frame_count % 5] = sidewalk_confidence[0]

        if frame_count >= 0:
            # Display arrow
            showDirectionFrame(current_time, frame_count)

            sidewalk_mean_last_few_frames = np.mean(sidewalk_confidences_last)

            # Display model prediction
            prediction = showPercentageAndTrafficLight(current_time, sidewalk_mean_last_few_frames, frame_count)

            # Display sidewalk confidence
            showSidewalkConfidence(sidewalk_confidence[0])

            # plot shizzle
            
            vals = get_raw_values_and_last_two_RL(current_time, frame_count)
            t = vals[0]
            jRL = vals[1]
            
            overall_score = prediction[0]
            sidewalk_score = prediction[2]
            heuristic_score = prediction[3]
            
            timeline.append(t)
            plotarr.append(jRL)
            
            overall_scores.append(overall_score)
            sidewalk_scores.append(sidewalk_score)
            heuristic_scores.append(heuristic_score)
            
            end_time = int(t + (windowsize - 10))
            if int(t) >= (windowsize - 10):
                new_xlim = [int(t) - (windowsize - 10),int(t) + 10]
                ax.set_xlim(new_xlim)
                ax2.set_xlim(new_xlim)
            line1.set_xdata(timeline)
            line1.set_ydata(plotarr)
            line2.set_xdata(timeline)
            line2.set_ydata(heuristic_scores)
            line3.set_xdata(timeline)
            line3.set_ydata(sidewalk_scores)
            line4.set_xdata(timeline)
            line4.set_ydata(overall_scores)
            
            
            figure.canvas.draw()
            figure.canvas.flush_events()
            
            figure2.canvas.draw()
            figure.canvas.flush_events()

        if cv2.waitKey(1) == ord('q'):
            break

        frame_count += 1
        if frame_by_frame_mode:
            while(True):
                if cv2.waitKey(1) == ord('n'):
                    break
                #time.sleep(0.001)
        #else:
            #time.sleep(delay) # might required if video speed is not matching real one


    cap.release()
    #writer.release()
    cv2.destroyAllWindows()

showVideo()
