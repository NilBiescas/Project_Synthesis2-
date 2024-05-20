from utils import *
import json
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm


dir2publaynet = "/hhome/ps2g07/document_analysis/github/Project_Synthesis2-/Datasets/Publaynet"

dir2signatures = "/hhome/ps2g07/document_analysis/github/Project_Synthesis2-/Datasets/Signatures_dataset"
dataset = "preprocessed"

dir2stamps = "/hhome/ps2g07/document_analysis/github/Project_Synthesis2-/Datasets/Stamps_dataset"
dir2qrs = "/hhome/ps2g07/document_analysis/github/Project_Synthesis2-/Datasets/QRs_dataset/"
dir2barcodes = "/hhome/ps2g07/document_analysis/github/Project_Synthesis2-/Datasets/Barcodes_dataset"
dir_borders = "/hhome/ps2g07/document_analysis/github/Project_Synthesis2-/Datasets/Borders"

with open(f"{dir2stamps}/splits.json", 'r') as f:
    stamps = json.load(f)
    
with open(f"{dir2signatures}/splits.json", 'r') as f:
    signatures = json.load(f) 

with open(f"{dir2qrs}/splits.json", 'r') as f:
    qrs = json.load(f)

with open(f"{dir2barcodes}/splits.json", 'r') as f:
    barcodes = json.load(f)

universidades_mexico = '/hhome/ps2g07/document_analysis/github/Project_Synthesis2-/Nil/universidades_mexico'
with open(universidades_mexico, 'r') as f:
    universidades = json.load(f)

escudos_entidades_def = '/hhome/ps2g07/document_analysis/github/Project_Synthesis2-/Nil/escudos_entidades_federativas'
with open(escudos_entidades_def, 'r') as f:
    escudos = json.load(f)
#ds_dir = '/hhome/ps2g07/document_analysis/github/Project_Synthesis2-/Datasets/Publaynet/Publaynet_partition_1/Synthetics_DS_partition1'
# ds_dir = "C:/Users/Maria/OneDrive - UAB/Documentos/3r de IA/Synthesis project II/Github/Project_Synthesis2-/Datasets/Synthetics_DS_v2"
#ds_dir = "/hhome/ps2g07/document_analysis/github/Project_Synthesis2-/Datasets/Synthetic_DS_v4"
ds_dir = "/hhome/ps2g07/document_analysis/github/Project_Synthesis2-/Nil/synthesis_stamps"
os.makedirs(ds_dir, exist_ok=True)

import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

for split in ["train", "val"]:

    with open(f"{dir2publaynet}/{split}_syn.json", 'r') as f:
        publaynet = json.load(f)   

    dataset = {"categories":publaynet["categories"].copy(), "images":publaynet["images"].copy(), "annotations":[]}
    dataset["categories"].append({'supercategory': '', 'id': 6, 'name': 'signature'})
    dataset["categories"].append({'supercategory': '', 'id': 7, 'name': 'stamp'})
    dataset["categories"].append({'supercategory': '', 'id': 8, 'name': 'qr'})
    dataset["categories"].append({'supercategory': '', 'id': 9, 'name': 'barcode'})

    for i, sample in tqdm(enumerate(publaynet["images"])):     
        dataset["annotations"].append(get_all_annotations_from_image(publaynet, int(sample["id"])))
        
        l0 = cv2.imread(f"{dir2publaynet}/train/{sample['file_name']}")
        l0 = cv2.cvtColor(l0, cv2.COLOR_BGR2RGB)
        
        all = l0.copy()
        l1 = np.ones_like(l0) * 255
        
        imgs, dataset = add_title([all, l0], sample['id'], dataset)
        all, l0 = imgs
        
        n = np.random.randint(1, 4)
        imgs, dataset = add_stamps(n, [all, l1], stamps, dir2stamps, dataset, sample['id'])
        all, l1 = imgs

        n = np.random.randint(0, 5)
        imgs, dataset = add_signatures(n, [all, l1], signatures, dir2signatures, dataset, sample['id'])
        all, l1 = imgs
        
        n = np.random.randint(0, 3)
        imgs, dataset = add_qrs(n, [all, l1], qrs, dir2qrs, dataset, sample['id'])
        all, l1 = imgs
        
        n = np.random.randint(0, 2)
        imgs, dataset = add_barcodes(n, [all, l1], barcodes, dir2barcodes, dataset, sample['id'])
        all, l1 = imgs
        
        if random.choice([True, False]):
            imgs = add_bkg_noise([all, l1])
            all, l1 = imgs
        
        if random.choice([True]): #, False, False, False, False, False]):
            imgs = add_border(dir_borders, [all, l0, l1])
            if imgs:
                all, l0, l1 = imgs
            
        img = add_lighting_and_yellow_tint([all])
        all = img[0]
        
        # # Save the image
        # os.makedirs(f"{ds_dir}/{split}", exist_ok=True)
        os.makedirs(f"{ds_dir}/{split}/gt_l0", exist_ok=True)
        os.makedirs(f"{ds_dir}/{split}/gt_l1", exist_ok=True)
        os.makedirs(f"{ds_dir}/{split}/gt_all", exist_ok=True)

        # show the 3 images
        # fig, ax = plt.subplots(1, 3, figsize=(20, 10))
        # ax[0].imshow(l0)
        # ax[1].imshow(l1)
        # ax[2].imshow(all)
        # os.makedirs(f"{ds_dir}/{split}", exist_ok=True)
        # plt.savefig(f"{ds_dir}/{split}/{sample['file_name']}")
        
        # break
        cv2.imwrite(f"{ds_dir}/{split}/gt_l0/{sample['file_name']}", cv2.cvtColor(np.array(l0), cv2.COLOR_RGB2BGR))
        cv2.imwrite(f"{ds_dir}/{split}/gt_l1/{sample['file_name']}", cv2.cvtColor(np.array(l1), cv2.COLOR_RGB2BGR))
        cv2.imwrite(f"{ds_dir}/{split}/gt_all/{sample['file_name']}", cv2.cvtColor(np.array(all), cv2.COLOR_RGB2BGR))
        # print(f"Image {sample['file_name']} saved")
        
        if i % 100 == 0:
            with open(f"{ds_dir}/{split}.json", 'w') as f:
                json.dump(dataset, f)
            print(f"Dataset saved at {ds_dir}/{split}.json") 
            
            
    #remove all [] from the annotations list
    dataset["annotations"] = [item for item in dataset["annotations"] if item != []]

    with open(f"{ds_dir}/{split}.json", 'w') as f:
        json.dump(dataset, f)