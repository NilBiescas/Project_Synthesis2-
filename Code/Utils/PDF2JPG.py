# pass a pdf to jpg

import os
import shutil
from pdf2image import convert_from_path

in_dirr = r"C:\Users\Maria\OneDrive - UAB\Documentos\3r de IA\Synthesis project II\Github\Project_Synthesis2-\Sample documents\Más ejemplos iDISC"

out_dirr = r"C:\Users\Maria\OneDrive - UAB\Documentos\3r de IA\Synthesis project II\Github\Project_Synthesis2-\Sample documents - JPG\Más ejemplos iDISC"

for root, dir, files in os.walk(in_dirr):
    for file in files:
        if file.endswith(".pdf"):
            pages = convert_from_path(os.path.join(root, file), 500)
            os.makedirs(out_dirr, exist_ok=True)
            for i, page in enumerate(pages):
                page.save(os.path.join(out_dirr, file.split(".")[0] + "_" + str(i) + ".jpg"), "JPEG")
                
        if file.endswith(".jpeg") or file.endswith(".jpg") or file.endswith(".png"):
            os.makedirs(out_dirr, exist_ok=True)
            shutil.copy(os.path.join(root, file), os.path.join(out_dirr, file))