from ultralytics import YOLO



## Try: 1##
#model = YOLO("yolov8n.pt")  # load pre trained model
#
#model.train(data='yolo_config.yaml', 
#            dropout=0.15, 
#            epochs = 50, 
#            batch = -1, 
#            optimizer='AdamW',
#            flipud = 0.3,
#            # (float) image flip up-down (probability)
#            fliplr = 0.5,
#            mosaic = 0.6,
#            mixup  = 0.3)
#            # (float) image flip left-right (probability))  # train a new model from scratch

## Try: 2## Run 27
# Pretrained weights from imagenet and using the default configuration and with the bigger dataset
model = YOLO("yolov8n.pt")
model.train(data='yolo_config.yaml', epochs = 50)


## Try: 3## # Run30
# Pretrained weights from doclynet and using the default configuration and with the bigger dataset
#print("Try: 3")
#model = YOLO("dla-model.pt")
#model.train(data='yolo_config.yaml', epochs = 50, optimizer='AdamW')