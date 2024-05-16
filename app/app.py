from flask import Flask, render_template, request, send_file, send_from_directory, flash, redirect, url_for
from werkzeug.utils import secure_filename
import os
from helpers import *
from pptx.util import Pt
from pptx import Presentation
import torch
import aspose.slides as slides
import aspose.pydrawing as draw

from doctr import io
from doctr import utils
from time import sleep
#from doctr.io import DocumentFile
#from doctr.utils.visualization import vis_and_synth


from PIL import Image
import matplotlib.pyplot as plt



app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg'}

device = "cuda" if torch.cuda.is_available() else "cpu"

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def inference_models(file, device="cpu"): 
    file_name = secure_filename(file.filename)
    path2file = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
    
    num_pages = pdf2png(path2file, app.config['UPLOAD_FOLDER'] + "/" + file_name[:-4])
    
    files = os.listdir(path2file[:-4])
    separate_layers(os.path.join(path2file[:-4], files[0]), out_dir = app.config['UPLOAD_FOLDER'], device=device)
   
    path2l0 = os.path.join(app.config['UPLOAD_FOLDER'], file_name[:-4])
    png2pdf([path2l0 + "/" + f for f in os.listdir(path2l0)], app.config['UPLOAD_FOLDER'])
    # files[0][:-4] + "_l0.png", files[0][:-4] + "_l1.png"
    l0 = files[0][:-4] + "_l0.png"
    l1 = files[0][:-4] + "_l1.png"
    
    path2l0pdf = os.path.join(app.config['UPLOAD_FOLDER'], "layer_sep.pdf")
    
    result, doc = docTR_inference(path2l0pdf, device=device)
            
    file_name_pptx = file_name.replace(".pdf", ".pptx")
    path2savePptx = os.path.join(app.config['UPLOAD_FOLDER'], file_name_pptx)
    create_pptx(result, doc, path2savePptx)
    
    file_name_png = file_name.replace(".pdf", ".png")
    path2savePng = os.path.join(app.config['UPLOAD_FOLDER'], file_name_png)
    plot_slide(path2savePptx, 0, out_dir=path2savePng)
    
    return l0, l1, file_name_png
    

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part', 'error')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file', 'error')
            return render_template('upload.html')
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            file.save(file_path)
            
            l0, l1, ocr = inference_models(file, device=device)
            
            return render_template('upload.html', l0=l0, l1=l1, ocr_out=ocr)
        
            # return render_template('preview.html', image_path=image_path,power_path=power_pdf, file_path=presen)
    return render_template('upload.html')

@app.route('/contact')
def contact_us():
    return render_template('contact.html')

@app.route('/team')
def team():
    return render_template('team.html')

@app.route('/preview/<filename>', methods=['GET'])
def preview_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename), as_attachment=True)

@app.route('/uploads/<filename>', methods=['GET'])
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

def cleanup_upload_folder():
    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

import atexit
atexit.register(cleanup_upload_folder)

if __name__ == '__main__':
    app.run(debug=True)