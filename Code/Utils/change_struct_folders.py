import os

dirr = r"C:\Users\Maria\OneDrive - UAB\Documentos\3r de IA\Synthesis project II\Github\Project_Synthesis2-\Sample documents - JPG"

"""
change from 

Folder1
    - 0.png
    - 1.png
Folder2
    - 0.png
    - 1.png
    
To 
Folder1_0.png
Folder1_1.png
Folder2_0.png
Folder2_1.png
"""

for root, dir, files in os.walk(dirr):
    for file in files:
        os.rename(os.path.join(root, file), os.path.join(dirr, root.split("/")[-1] + "_" + file))
        