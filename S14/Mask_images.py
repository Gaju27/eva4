import glob

from PIL import Image
save_dir="/mask_final/"
path = "/Mask/*.*"
# i = 1
for file in glob.glob(path):
    im = Image.open(file, 'r').convert('RGBA')
    rgbdata = im.tobytes("raw", "RGB")
    alphadata = im.tobytes("raw", "A")
    alphaimage = Image.frombytes("L", im.size, alphadata)
    alphaimage.save(file)
    # alphaimage.save(f'{save_dir}/Final_{i}.jpg', "JPEG")
    # i+=1

