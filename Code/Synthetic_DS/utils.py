import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2

from PIL import Image
import numpy as np
import cv2
import time
import random
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import os
import textwrap

def add_transparent_image(background, foreground, x_offset=None, y_offset=None):
    bg_h, bg_w, bg_channels = background.shape
    fg_h, fg_w, fg_channels = foreground.shape

    assert bg_channels == 3, f'background image should have exactly 3 channels (RGB). found:{bg_channels}'
    assert fg_channels == 4, f'foreground image should have exactly 4 channels (RGBA). found:{fg_channels}'

    # center by default
    if x_offset is None: x_offset = (bg_w - fg_w) // 2
    if y_offset is None: y_offset = (bg_h - fg_h) // 2

    w = min(fg_w, bg_w, fg_w + x_offset, bg_w - x_offset)
    h = min(fg_h, bg_h, fg_h + y_offset, bg_h - y_offset)

    if w < 1 or h < 1: return

    # clip foreground and background images to the overlapping regions
    bg_x = max(0, x_offset)
    bg_y = max(0, y_offset)
    fg_x = max(0, x_offset * -1)
    fg_y = max(0, y_offset * -1)
    foreground = foreground[fg_y:fg_y + h, fg_x:fg_x + w]
    background_subsection = background[bg_y:bg_y + h, bg_x:bg_x + w]

    # separate alpha and color channels from the foreground image
    foreground_colors = foreground[:, :, :3]
    alpha_channel = foreground[:, :, 3] / 255  # 0-255 => 0.0-1.0

    # construct an alpha_mask that matches the image shape
    alpha_mask = np.dstack((alpha_channel, alpha_channel, alpha_channel))

    # combine the background with the overlay image weighted by alpha
    composite = background_subsection * (1 - alpha_mask) + foreground_colors * alpha_mask

    # overwrite the section of the background image that has been updated
    background[bg_y:bg_y + h, bg_x:bg_x + w] = composite
    
def get_all_annotations_from_image(dataset, image_id):
    a = [a for a in dataset["annotations"] if a["image_id"] == image_id]
    return a
    
def on_top_of_text(bb, annotations): 
    # bb = [x, y, w, h]
    # it any of the corners of the bb is inside any of the annotations, return True
    x, y, w, h = bb
    for a in annotations:
        print("test", a["bbox"])
        x2, y2, w2, h2 = a["bbox"]
        if x2 <= x <= x2+w2 and y2 <= y <= y2+h2: return True
        if x2 <= x+w <= x2+w2 and y2 <= y <= y2+h2: return True
        if x2 <= x <= x2+w2 and y2 <= y+h <= y2+h2: return True
        if x2 <= x+w <= x2+w2 and y2 <= y+h <= y2+h2: return True
    return False


def create_COCO_annotation(dataset, image_id, category_id, x, y, w, h):
    # print(dataset)
    annotation = {
        "area": w * h,
        "iscrowd": 0,
        "image_id": image_id,
        "bbox": [x, y, w, h],
        "category_id": category_id, 
        "id": len(dataset["annotations"]) + 3265447 + 14423 + 1,
    }
    
    dataset["annotations"].append(annotation)
    return dataset


def add_stamps(n, list_of_imgs, stamps, stamps_ds, dataset, image_id, split="train"):
    label = 7
    
    j = 0
    while j < n:
        idx = np.random.randint(0, len(stamps))
        if stamps[idx]['split'] != split:
            continue
        try: 
            stamp = cv2.imread(f"{stamps_ds}/preprocessed/{stamps[idx]['file_name']}", cv2.IMREAD_UNCHANGED)
            stamp = cv2.cvtColor(stamp, cv2.COLOR_BGR2RGBA)
            
            size = np.random.normal(200, 200)
            size = int(np.clip(size, 50, list_of_imgs[0].shape[1]*0.8))
            size = int(np.clip(size, 50, list_of_imgs[0].shape[0]*0.8))
            
            if np.random.random() > 0:
                stamp = cv2.copyMakeBorder(stamp, 0, stamp.shape[1]//2, stamp.shape[1]//3, stamp.shape[1]//3, cv2.BORDER_CONSTANT, value=(255, 255, 255, 0))
                
                # add a text below the stamp with the name of the stamp
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5 * stamp.shape[1] / 200
                font_thickness = int(size // 100)
                
                text = stamps[idx]['file_name'].split(".")[0]
                text_width = np.random.random() * 0.28 + 0.62  # 0.62 - 0.9
                wrapped_text = textwrap.wrap(text, width=int(text_width*20))
                
                sep = np.random.random() + 3
                for i, line in enumerate(wrapped_text):
                    text_size = cv2.getTextSize(line, font, font_scale, font_thickness)[0]
                    text_x = (stamp.shape[1] - text_size[0]) // 2
                    text_y = stamp.shape[0] - int(stamp.shape[0]//sep) + text_size[1] + i * (text_size[1] + text_size[1]//2)
                
                    cv2.putText(stamp, line, (text_x, text_y), font, font_scale, (0, 0, 0, 255), font_thickness, cv2.LINE_AA)
        
                # plt.imshow(stamp)
                # plt.savefig(f"stamp_{j}.png")
            stamp = cv2.resize(stamp, (0,0), fx=size/stamp.shape[1], fy=size/stamp.shape[0])
            
            angle = np.random.randint(-30, 30)
            center = (stamp.shape[1]//2, stamp.shape[0]//2)
            M = cv2.getRotationMatrix2D(center, angle, 1)
            stamp = cv2.warpAffine(stamp, M, (stamp.shape[1], stamp.shape[0]))
            
            # print(f"stamp: {stamp.shape}, page: {page.shape}", page.shape[1]-stamp.shape[1], page.shape[0]-stamp.shape[0])
            
            # choose x_offset and y_offset while the stamp is on top of any annotation
            x_offset = np.random.randint(0, list_of_imgs[0].shape[1] - stamp.shape[1])
            y_offset = np.random.randint(0, list_of_imgs[0].shape[0] - stamp.shape[0])

            alpha = np.random.normal(1-(1/(list_of_imgs[0].shape[1]/size)), 0.2)
            # print(f"alpha: {0.8-(1/(page.shape[1]/size))}")
            alpha = np.clip(alpha, 0.1, 1)
            
            for img in list_of_imgs:
                add_transparent_image(img, stamp*alpha, x_offset, y_offset)
            # cv2.rectangle(img, (x_offset, y_offset), (x_offset+stamp.shape[1], y_offset+stamp.shape[0]), (0, 255, 0), 4)
            
            dataset = create_COCO_annotation(dataset, image_id, label, x_offset, y_offset, stamp.shape[1], stamp.shape[0])
        
            j += 1
        except Exception as e:
            print(f"Error adding stamp {stamps[idx]['file_name']}")
            raise(e)
        
    return list_of_imgs, dataset

def add_qrs(n, list_of_imgs, qrs, qrs_dataset, dataset, image_id, split='train'):
    label = 8
        
    j = 0
    while j < n:
        idx = np.random.randint(0, len(qrs))
        if qrs[idx]['split'] != split:
            continue
        try: 
            
            # print(f"Adding QR code {qrs_dataset}/preprocessed/{qrs[idx]['file_name']}")
            
            qr = cv2.imread(f"{qrs_dataset}/preprocessed/{qrs[idx]['file_name']}", cv2.IMREAD_UNCHANGED)
            # crop a 5 % of the image per each side
            qr = qr[qr.shape[0]//10:qr.shape[0]-(qr.shape[0]//10), qr.shape[1]//10:qr.shape[1]-(qr.shape[1]//10)]


            size = np.random.randint(50, 175)
            qr = cv2.resize(qr, (0,0), fx=size/qr.shape[1], fy=size/qr.shape[0])
            
            x_offset = np.random.randint(0, list_of_imgs[0].shape[1] - qr.shape[1])
            y_offset = np.random.randint(0, list_of_imgs[0].shape[0] - qr.shape[0])

            alpha = np.random.normal(0.9, 0.2)
            alpha = np.clip(alpha, 0.1, 0.85)
            
            for img in list_of_imgs:
                add_transparent_image(img, qr*alpha, x_offset, y_offset)
            #cv2.rectangle(img, (x_offset, y_offset), (x_offset+qr.shape[1], y_offset+qr.shape[0]), (0, 0, 255), 3)
            
            dataset = create_COCO_annotation(dataset, image_id, label, x_offset, y_offset, qr.shape[1], qr.shape[0])
        
            j += 1
        except:
            print(f"Error adding qr {qrs[idx]['file_name']}")
            raise(Exception)
            
    return list_of_imgs, dataset


def add_barcodes(n, list_of_imgs, barcodes, barcodes_dataset, dataset, image_id, split='train'):
    label = 9
        
    j = 0
    while j < n:
        idx = np.random.randint(0, len(barcodes))
        if barcodes[idx]['split'] != split:
            continue
        try: 
            barcode = cv2.imread(f"{barcodes_dataset}/preprocessed/{barcodes[idx]['file_name']}", cv2.IMREAD_UNCHANGED)

            size = np.random.randint(50, 175)
            barcode = cv2.resize(barcode, (0,0), fx=size/barcode.shape[1], fy=size/barcode.shape[0])
            
            x_offset = np.random.randint(0, list_of_imgs[0].shape[1] - barcode.shape[1])
            y_offset = np.random.randint(0, list_of_imgs[0].shape[0] - barcode.shape[0])

            alpha = np.random.normal(0.9, 0.2)
            alpha = np.clip(alpha, 0.1, 1)
            
            for img in list_of_imgs:
                add_transparent_image(img, barcode*alpha, x_offset, y_offset)

            dataset = create_COCO_annotation(dataset, image_id, label, x_offset, y_offset, barcode.shape[1], barcode.shape[0])
            
            j += 1
        except:
            print(f"Error adding barcode {barcodes[idx]['file_name']}")
            raise(Exception)
            
    return list_of_imgs, dataset


def add_border(dir_borders, list_of_imgs):
    try:
        border = random.choice(os.listdir(dir_borders))
        border = Image.open(f"{dir_borders}/{border}")
        
        img = Image.fromarray(list_of_imgs[0])
        border = border.resize(img.size)
        imgs = []
        for img in list_of_imgs:
            img = Image.fromarray(img)
            img.paste(border, (0, 0), border)
            imgs.append(img)
        return imgs
    
    except: 
        pass
    

def add_bkg_noise(list_of_imgs):
    num_letters = random.randint(5, 25)
    letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789              '
    bkg_text = ''.join(random.choices(letters, k=num_letters))
    alpha = random.randint(0, 50)
    font_size = random.randint(15, 30)
    
    img = Image.fromarray(list_of_imgs[0])
    bkg_text = bkg_text * (img.size[0] * 5 // (len(bkg_text)*font_size) + 1)
    
    # create a new image with the bkg_text
    bkg_img = Image.new('RGB', img.size, (0, 0, 0))
    for j in range(0, img.size[1], int(font_size*1.1)):
        i = random.randint(-200, 0)
        draw = ImageDraw.Draw(bkg_img)
        draw.text((i, j), bkg_text, fill=(alpha, alpha, alpha), font_size=font_size)
    
    solid_color = Image.new('RGB', img.size, (0, 0, 0))
    bkg_img = bkg_img.convert('L')
    solid_color.putalpha(bkg_img)
    
    imgs = []
    for img in list_of_imgs:
        img = Image.fromarray(img)
        img.paste(solid_color, (0, 0), solid_color)
        imgs.append(img)
    return imgs


def generate_gradient_mask(shape, direction='vertical', direction_type='dark_to_light'):
    height, width = shape
    gradient_mask = np.zeros(shape, dtype=np.float32)
    
    maxx = random.uniform(1.1, 1.5)
    minn = random.uniform(0.5, 0.8)
    
    if direction == 'vertical':
        for y in range(height):
            alpha = y / height if direction_type == 'dark_to_light' else 1 - y / height
            alpha = alpha * (maxx - minn) + minn
            gradient_mask[y, :] = alpha
    elif direction == 'horizontal':
        for x in range(width):
            alpha = x / width if direction_type == 'dark_to_light' else 1 - x / width
            alpha = alpha * (maxx - minn) + minn
            gradient_mask[:, x] = alpha
    elif direction == 'diagonal':
        for y in range(height):
            for x in range(width):
                alpha = min(y, x) / max(height, width) if direction_type == 'dark_to_light' else 1 - min(y, x) / max(height, width)
                alpha = alpha * (maxx - minn) + minn
                gradient_mask[y, x] = alpha
    else:
        gradient_mask = np.ones(shape, dtype=np.uint8)
    return gradient_mask

def add_lighting_and_yellow_tint(list_of_imgs):
    img_array = np.array(list_of_imgs[0])
    
    yellow_factor = random.uniform(0.6, 1)  # Random factor between 0.5 and 1.0
    corr_green = random.uniform(0.7, 1)
    corr_red = random.uniform(1, 1.2)
    yellow_factor_red = np.clip(yellow_factor*corr_red, 0, 1)
    
    directions = ['vertical', 'horizontal', 'diagonal', 'none']
    direction = random.choice(directions)
    direction_type = 'dark_to_light' if random.choice([True, False]) else 'light_to_dark'
    
    gradient_mask = generate_gradient_mask(img_array.shape[:2], direction, direction_type)
    
    overlayed_imgs = []
    for img in list_of_imgs:
        img_array = np.array(img)
        img_array[:, :, 0] = img_array[:, :, 0] * yellow_factor_red  # Increasing red channel
        img_array[:, :, 1] = img_array[:, :, 1] * yellow_factor  # Reducing green channel
        img_array[:, :, 2] = img_array[:, :, 2] * yellow_factor * corr_green  # Reducing blue channel
        
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        img_array = img_array.astype(np.float32)
        img_array[:, :, 0] = img_array[:, :, 0] * gradient_mask
        img_array[:, :, 0] = np.clip(img_array[:, :, 0], 0, 254)
        img_array = img_array.astype(np.uint8)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_LAB2RGB)
        overlayed_imgs.append(Image.fromarray(img_array))

    return overlayed_imgs

def add_signatures(n, list_of_imgs, signatures, signatures_ds, dataset, image_id, split='train'):
    label = 6
    
    j = 0
    while j < n:
        idx = np.random.randint(0, len(signatures))
        if signatures[idx]['split'] != split:
            continue
        try: 
            #signature = cv2.imread(f"{signatures_ds}/dataset3/forge_preprocessed/{signatures[idx]['file_name']}", cv2.IMREAD_UNCHANGED)
            signature = cv2.imread(f"{signatures_ds}/preprocessed/{signatures[idx]['file_name']}", cv2.IMREAD_UNCHANGED)
            signature = cv2.cvtColor(signature, cv2.COLOR_BGR2RGBA)
            #print(signature)
            
            size = np.random.randint(50, 200)
            signature = cv2.resize(signature, (0,0), fx=size/signature.shape[1], fy=size/signature.shape[0])
            
            # rotate the image randomly
            angle = np.random.randint(-30, 30)
            center = (signature.shape[1]//2, signature.shape[0]//2)
            M = cv2.getRotationMatrix2D(center, angle, 1)
            signature = cv2.warpAffine(signature, M, (signature.shape[1], signature.shape[0]))
            
            
            x_offset = np.random.randint(0, list_of_imgs[0].shape[1] - signature.shape[1])
            y_offset = np.random.randint(0, list_of_imgs[0].shape[0] - signature.shape[0])

            alpha = np.random.normal(0.9, 0.2)
            alpha = 1
            
            for img in list_of_imgs:
                add_transparent_image(img, signature*alpha, x_offset, y_offset)
            # cv2.rectangle(img, (x_offset, y_offset), (x_offset+signature.shape[1], y_offset+signature.shape[0]), (255, 0, 255), 3)
            
            dataset = create_COCO_annotation(dataset, image_id, label, x_offset, y_offset, signature.shape[1], signature.shape[0])
            
            j += 1
            
        except:
            print(f"Error adding signature {signatures[idx]['file_name']}")
            raise(Exception)
            
    return list_of_imgs, dataset

def add_title(list_of_imgs, image_id, dataset):
    # if there is a blank space in the image, add a title
    label = 1
    
    size = np.random.randint(30, 150)
    bold = random.choice([True, False])
    italic = random.choice([True, False])
    
    letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789              '
    title = ''.join(random.choices(letters, k=random.randint(10, 40)))
    
    x = random.randint(0, list_of_imgs[0].shape[1])
    y = random.randint(0, list_of_imgs[0].shape[0])
    
    # imgs = []
    #add text using cv2.putText
    for img in list_of_imgs:
        cv2.putText(img, title, (x, y), cv2.FONT_HERSHEY_SIMPLEX, size/100, (0, 0, 0), 2, cv2.LINE_AA)
    
    w = 50
    h = 20    
    dataset = create_COCO_annotation(dataset, image_id, label, x, y, w, h)
    print(title, x/list_of_imgs[0].shape[1], y/list_of_imgs[0].shape[0], list_of_imgs[0].shape, size)
        
    return list_of_imgs, dataset
        
    