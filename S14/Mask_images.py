import glob

from PIL import Image

# path = "C:/Users/gajanana_ganjigatti/Documents/Gaju_data/Quest/eva4/S14-15/BACKUP/Captures/*.*"
save_dir = "C:/Users/gajanana_ganjigatti/Documents/Gaju_data/Quest/eva4/S14-15/Mask/"
path = "C:/Users/gajanana_ganjigatti/Documents/Gaju_data/Quest/eva4/S14-15/Mask/*.*"
# i = 1
for file in glob.glob(path):
    im = Image.open(file, 'r').convert('RGBA')
    rgbdata = im.tobytes("raw", "RGB")
    alphadata = im.tobytes("raw", "A")
    alphaimage = Image.frombytes("L", im.size, alphadata)
    alphaimage.save(file)
    # alphaimage.save(f'{save_dir}/Final_{i}.jpg', "JPEG")
    # i+=1

