import cv2
import math
from ultralytics import YOLO

model = YOLO("model/custom_trained_yolov8s.pt")
classNames = ["sidewalk", "car", "bicycle", "motorcycle", "bus"]
video_path = "videos/test_video_2.mp4"

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
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2

            cv2.putText(frame, classNames[cls], org, font, fontScale, color, thickness)


def showVideo():
    cap = cv2.VideoCapture(video_path)

    frame_width = int(cap.get(3)) 
    frame_height = int(cap.get(4)) 
   
    size = (frame_width, frame_height) 

    writer = cv2.VideoWriter('video_output.mp4',  
                         cv2.VideoWriter_fourcc(*'MP4V'), 
                         30, size) 

    while True:
        ret, frame = cap.read()
        results = model(frame, stream=True)

        showBoxesOnVideo(results, frame, classNames)


        if not ret:
            break
        cv2.imshow('Video', frame)

        writer.write(frame)

        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    
showVideo()







