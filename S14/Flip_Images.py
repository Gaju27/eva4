import glob
from PIL import Image
fpath = "./Foreground/*.*"
for file in glob.glob(fpath):
    fimg = Image.open(file)
    fimg = fimg.transpose(Image.FLIP_LEFT_RIGHT)
    fimg.save(file)