import os
import shutil

in_dirr = r"C:\Users\Maria\OneDrive - UAB\Documentos\3r de IA\Synthesis project II\Github\Project_Synthesis2-\Sample documents - JPG"
out_dirr = r"C:\Users\Maria\OneDrive - UAB\Documentos\3r de IA\Synthesis project II\Github\Project_Synthesis2-\Sample documents - JPG - NoName"

"""
change from 
Folder1_0.png
Folder1_1.png
Folder2_0.png
Folder2_1.png

0.jpg
1.jpg
2.jpg
...
mapNames.txt
"""

counter = 0
name2num = {}
for root, dir, files in os.walk(in_dirr):
    for file in files:
        if file.endswith(".jpg"):
            name2num[file] = str(counter)
            counter += 1
            
            # copy the file with new name
            os.makedirs(out_dirr, exist_ok=True)
            shutil.copy(os.path.join(root, file), os.path.join(out_dirr, name2num[file] + ".jpg"))
            
            # write the mapping
            with open(os.path.join(out_dirr, "mapNames.txt"), "a") as f:
                f.write(file + " -> " + name2num[file] + ".jpg\n")