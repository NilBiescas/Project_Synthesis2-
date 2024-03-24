from doctr.io import DocumentFile
from doctr.models import ocr_predictor

model = ocr_predictor(pretrained=True)
model.to('cuda')
# PDF
doc = DocumentFile.from_pdf("/hhome/ps2g07/document_analysis/github/Project_Synthesis2-/Sample documents/Certificates of no criminal records/Constancia de no antecedentes penales federales.pdf")
# Analyze
result = model(doc)


# result.show()
res = result.synthesize()

# res is a np array with the image of the document
import matplotlib.pyplot as plt
import time 
import numpy as np

for r,info in res: 
    plt.imsave(f"synth{time.time()}.png", r.astype(np.uint8))
    print(info)

print(result)


# import pickle
# with open("/hhome/ps2g07/document_analysis/OCR/DocTR/exemples/1.pkl", "wb") as f:
#     pickle.dump(result, f)