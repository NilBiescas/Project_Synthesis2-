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

def plot_slide(presentation, slide, show=True):

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
        #bmp.save(path, draw.imaging.ImageFormat.jpeg)
     
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