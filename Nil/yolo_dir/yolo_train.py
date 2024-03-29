from ultralytics import YOLO


# Load a model
model = YOLO("yolov8n.pt")  # load pre trained model

model.train(data='yolo_config.yaml', 
            dropout=0.15, 
            epochs = 50, 
            batch = -1, 
            optimizer='AdamW',
            flipud = 0.3,
            # (float) image flip up-down (probability)
            fliplr = 0.5,
            mosaic = 0.6,
            mixup  = 0.3)
            # (float) image flip left-right (probability))  # train a new model from scratch
# model.train(data='yolo_config.yaml0', dropout=0.1, overrides={'epochs': 1000, 'batch-size': 16, 'dropout': 0.2})  # train a new model from scratch