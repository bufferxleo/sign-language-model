import os
import shutil
path="C:\\Users\\saisu\\OneDrive\\Desktop\\proj\\sign_to_text\\img_data\\ref"
for i in os.listdir(path):
    p="C:\\Users\\saisu\\OneDrive\\Desktop\\proj\\sign_to_text\\img_data\\ref\\"+str(i)
    for j in os.listdir(p):
        shutil.move(p+"\\"+j,"C:\\Users\\saisu\\OneDrive\\Desktop\\proj\\sign_to_text\\img_data\\imgs\\"+str(i)+f'{j}')