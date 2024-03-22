import torch
import os
from pathlib import Path
from PIL import Image
import numpy as np
import cv2


from paths import (WEIGHTS_PATH_prima, CONFIG_PATH_prima, 
                    WEIGHTS_PATH_doclaynet, CONFIG_PATH_doclaynet, 
                    WEIGHTS_PATH_publaynet, CONFIG_PATH_publaynet,
                    WEIGHTS_PATH_tablebank, CONFIG_PATH_tablebank)

from SwinDocSegmenter.test import (setup, build_model, 
                                        DetectionCheckpointer,
                                        draw_box)

from paths import SAMPLE_DOCUMENTS_DIR
from pdf2image import convert_from_path


# Load model and weights
cfg = setup({"config_file": str(CONFIG_PATH_tablebank), "opts": []})
model = build_model(cfg)  # returns a torch.nn.module
DetectionCheckpointer(model).load(str(WEIGHTS_PATH_tablebank))
model.eval()

def preprocess_image(img_pil: Image, max_size=(1024, 1024)):
    
    # Resize using thumbnail to maintain aspect ratio
    img_pil.thumbnail(max_size, Image.Resampling.LANCZOS)

    img_resized = np.array(img_pil)
        
    # Convert hwc to chw format
    img_tensor = torch.tensor(img_resized).permute(2, 0, 1)
    
    img_tensor = img_tensor.float().cuda()
    
    return img_tensor

def layout_detection(pil_images, pdf, out_dir = "output_layout_detector_prima"):
    with torch.no_grad():
        for j, img in enumerate(pil_images):
            img_ = preprocess_image(img)
            outputs = model(
                [{"image": img_, "height": img_.shape[1], "width": img_.shape[2]}]
            )

            # Draw boxes on image
            for output in outputs:
                pred_classes = output["instances"].pred_classes.cpu().numpy()
                boxes = output["instances"].pred_boxes.tensor.cpu().numpy()
                scores = output["instances"].scores.cpu().numpy()

                # filter out low confidence boxes
                boxes = boxes[scores > 0.4]
                pred_classes = pred_classes[scores > 0.4]

                # draw boxes on image
                img_ = img_.cpu().numpy().transpose(1, 2, 0)
                for i, box in enumerate(boxes):
                    img_ = draw_box(img_, box, pred_classes[i], pred_classes[i])

            # Save the image to disk
            output_dir = Path(f"{out_dir}/{pdf.parent.name}")
            output_dir.mkdir(parents=True, exist_ok=True) 
            output_file = output_dir / f"{pdf.stem}_{j}.png"
            cv2.imwrite(str(output_file), img_)

if __name__ == "__main__":
    # Create an empty list to store paths
    pdfs = []

    # Use os.walk to iterate through the directory
    for dirpath, dirnames, filenames in os.walk(SAMPLE_DOCUMENTS_DIR):
        for filename in filenames:
            if filename.endswith(".pdf"):
                # Construct the full path
                full_path = Path(dirpath) / filename
                # Append it to the list
                pdfs.append(full_path)

    for pdf in pdfs:
        images = convert_from_path(pdf)
        layout_detection(images, pdf, out_dir = "output_layout_detector_tables")