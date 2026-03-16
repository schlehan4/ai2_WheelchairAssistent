from ultralytics import YOLO
 
# wrap main logic into guard because of windows
if __name__ == '__main__':
    
    if False:
        model = YOLO('runs/detect/yolov8n_custom_sidewalk5/weights/last.pt')  # load a partially trained model
        results = model.train(resume=True)
    else:
        # Load the model.
        #model = YOLO('yolo-Weights/yolov8n.pt') # using yolo v8 nano
        model = YOLO('yolo-Weights/yolov8s.pt') # using yolo v8 small
        #model = YOLO('yolo-Weights/yolov9c.pt') # using yolo v9c

        # Training.
        results = model.train(
        data='data.yaml',
        imgsz=640,
        epochs=50,
        batch=8,
        name='yolo_custom_sidewalk',
        amp=False)

    print("Done.")