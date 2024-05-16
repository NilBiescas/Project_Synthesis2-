from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from pptx.util import Pt
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import MSO_AUTO_SIZE
from pptx.enum.text import MSO_ANCHOR
import matplotlib.pyplot as plt
import aspose.pydrawing as draw
import aspose.slides as slides
import os


def docTR_inference(pdf_path, device="cpu"):
    model = ocr_predictor(pretrained=True)
    model.to(device)
    doc = DocumentFile.from_pdf(pdf_path)
    result = model(doc)
    return result, doc

    
def add_line_in_bounding_boxes(presentation, slide_index, text, coordinates):
    slide = presentation.slides[slide_index]
    left, top, width, height = coordinates
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    # tf.word_wrap = True
    # print(0.75*coordinates[3])
    tf.paragraphs[0].font.size = Pt(int(0.00007*coordinates[3]))
    tf.paragraphs[0].text = text
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.auto_size = MSO_AUTO_SIZE.SHAPE_TO_FIT_TEXT

def plot_slide(presentation, slide, out_dir):

    with slides.Presentation(presentation) as pres:
        # Access the first slide
        sld = pres.slides[slide]

        # User defined dimension
        desiredX = 210*4
        desiredY = 297*4

        # Getting scaled value  of X and Y
        ScaleX = (1.0 / pres.slide_size.size.width) * desiredX
        ScaleY = (1.0 / pres.slide_size.size.height) * desiredY


        # Create a full scale image
        bmp = sld.get_thumbnail(ScaleX, ScaleY)

        # save the image to disk in JPEG format
        print(out_dir)
        bmp.save(out_dir, draw.imaging.ImageFormat.png)
        return bmp
    
    
def create_pptx(result, doc, path2save): 
    presentation = Presentation()
    for i, res in enumerate(result.pages):
                
        slide_layout = presentation.slide_layouts[6]
        presentation.slides.add_slide(slide_layout)
                                                    
        raw_data = res.export()
        
        img = doc[i]
        
        height, width, _ = img.shape
        
        pixels_per_inch = 70  # Standard resolution for screens
        new_width = int(width * 914400 / pixels_per_inch)
        new_height = int(height * 914400/ pixels_per_inch)

        # Resize the slide layout
        presentation.slide_width = new_width
        presentation.slide_height = new_height
        
        for block in raw_data["blocks"]:
            for line in block["lines"]:
                [x1, y1], [x2, y2] = line["geometry"]
                x1, x2 = x1*width, x2*width
                y1, y2 = y1*height, y2*height

                text = " ".join([word["value"] for word in line["words"]])

                add_line_in_bounding_boxes(presentation, i, text, [Pt(x1), Pt(y1), Pt(x2-x1), Pt(y2-y1)])
    
    presentation.save(path2save)
    
    
    
    
    
    
    
    
"""
RESTORMER UTILS 

"""
    
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from runpy import run_path
from skimage import img_as_ubyte
import os

def save_img(filepath, img):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    cv2.imwrite(filepath,cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def load_img(filepath):
    img = cv2.imread(filepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def transform_output(output):
        # given a list of vector of size HxWx6 return a list of 2 tensors of size HxWx3
        if isinstance(output, list):
            pred_l0 = []
            pred_l1 = []
            for out in output: 
                pred_l0.append(out[:, :3, :])
                pred_l1.append(out[:, 3:, :])
            return pred_l0, pred_l1
        else: 
            return output[:, :3, :], output[:, 3:, :]

def separate_layers(file_, out_dir, device):
    # Load model architecture based on selected model
    parameters = {'inp_channels':3, 'out_channels':3, 'dim':48, 'num_blocks':[4,6,6,8], 'num_refinement_blocks':4, 'heads':[1,2,4,8], 'ffn_expansion_factor':2.66, 'bias':False, 'LayerNorm_type':'BiasFree', 'dual_pixel_task':False}
    load_arch = run_path("/hhome/ps2g07/document_analysis/github/Project_Synthesis2-/Code/Denoise/Restormer/basicsr/models/archs/restormer_6out_arch.py")
    model = load_arch['Restormer'](**parameters)
    model.to(device)

    weights = "/hhome/ps2g07/document_analysis/github/Project_Synthesis2-/Code/Denoise/Restormer/experiments/DocumentDenoisingLayers_ds_v4/models/zbest_psnr_l1.pth"
    checkpoint = torch.load(weights)
    model.load_state_dict(checkpoint['params'])
    model.eval()
    
    img_multiple_of = 8

    with torch.no_grad():
        if torch.cuda.is_available():
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()

        input_ = load_img(file_)
        
        h, w = input_.shape[:2]
        if h > w:
            input_ = np.array(TF.resize(TF.to_pil_image(input_), (1000, int(1000*w/h))))
        else:
            input_ = np.array(TF.resize(TF.to_pil_image(input_), (int(1000*h/w), 1000)))
        input_ = torch.from_numpy(input_).float().div(255.).permute(2,0,1).unsqueeze(0).to(device)
        
        # Pad the input if not_multiple_of 8
        height, width = input_.shape[2], input_.shape[3]
        H, W = ((height+img_multiple_of)//img_multiple_of)*img_multiple_of, ((width+img_multiple_of)//img_multiple_of)*img_multiple_of
        padh = H-height if height%img_multiple_of!=0 else 0
        padw = W-width if width%img_multiple_of!=0 else 0
        input_ = F.pad(input_, (0,padw,0,padh), 'reflect')

        restored = model(input_)
        restored_l0, restored_l1 = transform_output(restored)
        
        restored_l0 = torch.clamp(restored_l0, 0, 1)
        restored_l1 = torch.clamp(restored_l1, 0, 1)

        # Unpad the output
        restored_l0 = restored_l0[:,:,:height,:width]
        restored_l1 = restored_l1[:,:,:height,:width]

        restored_l0 = restored_l0.permute(0, 2, 3, 1).cpu().detach().numpy()
        restored_l1 = restored_l1.permute(0, 2, 3, 1).cpu().detach().numpy()
        
        restored_l0 = img_as_ubyte(restored_l0[0])
        restored_l1 = img_as_ubyte(restored_l1[0])

        f = os.path.splitext(os.path.split(file_)[-1])[0]
        save_img((os.path.join(out_dir, f+'_l0.png')), restored_l0)
        save_img((os.path.join(out_dir, f+'_l1.png')), restored_l1)

        return restored_l0, restored_l1
    
from pdf2image import convert_from_path
def pdf2png(pdf_path, png_root): 
    pages = convert_from_path(pdf_path, 500)
    os.makedirs(png_root, exist_ok=True)
    
    pdf_name = os.path.basename(pdf_path)
    for i, page in enumerate(pages):
        page.save(os.path.join(png_root, pdf_name.split(".")[0] + "_" + str(i) + ".png"), "PNG")
    
    return len(pages)


import img2pdf
from PIL import Image
def png2pdf(list_of_png_paths, pdf_root):    
    # converting into chunks using img2pdf
    a4inpt = (img2pdf.mm_to_pt(210),img2pdf.mm_to_pt(297))
    layout_fun = img2pdf.get_layout_fun(a4inpt)

    pdf_bytes = img2pdf.convert(list_of_png_paths, layout_fun=layout_fun)
    
    # writing to pdf file
    with open(pdf_root + "/layer_sep" + ".pdf", "wb") as f:
        f.write(pdf_bytes)
    
    
if __name__ == "__main__":
    input_img = "/hhome/ps2g07/document_analysis/github/Project_Synthesis2-/Sample documents/Academic transcipts/Certificado de estudios de bachillerato con apostilla.pdf"
    out_dir = "/hhome/ps2g07/document_analysis/github/Project_Synthesis2-/output_images"
    
    list_of_png_paths = ["/hhome/ps2g07/document_analysis/github/Project_Synthesis2-/app/uploads/Low_light_paper-RebuttalV2/Low_light_paper-RebuttalV2_0.png", "/hhome/ps2g07/document_analysis/github/Project_Synthesis2-/app/uploads/Low_light_paper-RebuttalV1/Low_light_paper-RebuttalV1_0.png"]
    
    png2pdf(list_of_png_paths, "/hhome/ps2g07/document_analysis/github/Project_Synthesis2-/app/uploads/")