import os
from random import randint
## Final
import glob
from PIL import Image

save_dir = "C:/Users/gajanana_ganjigatti/Documents/Gaju_data/Quest/eva4/S14-15/Superimposed/Random1/"
save_dir_mask="C:/Users/gajanana_ganjigatti/Documents/Gaju_data/Quest/eva4/S14-15/SuperImposed_mask/"
path = "C:/Users/gajanana_ganjigatti/Documents/Gaju_data/Quest/eva4/S14-15/Superimposed/Random1/*.*"

def main():
    i = 1
    bpath = "C:/Users/gajanana_ganjigatti/Documents/Gaju_data/Quest/eva4/S14-15/Images/"
    fpath = "C:/Users/gajanana_ganjigatti/Documents/Gaju_data/Quest/eva4/S14-15/Captures/"
    for file_back in os.listdir(bpath):

        # bimg.show()
        k=1
        for file_fore in os.listdir(fpath):
            forepath = fpath + file_fore
            location = file_fore[-5]
            # print(location)
            fimg = Image.open(forepath).convert("RGBA")
            # fimg.show()
            filter_size = (100, 100)

            # v1 = randint(10, 100)
            # v2 = randint(10, 100)
            # size = (v1, v2)
            backpath = bpath + file_back
            fimg = fimg.resize(filter_size, Image.BILINEAR)
            im_rgb = Image.open( 'C:/Users/gajanana_ganjigatti/Documents/Gaju_data/Quest/eva4/blank.png')

            k+=1

            for j in range(1,21):
                bimg = Image.open(backpath)
                im_a = Image.new("L", im_rgb.size, 0)
                # im_a = Image.new('RGBA', (250, 250), 'white')
                # print('bimg',bimg.size)
                # print('fimg',fimg.size)
                alphadata = fimg.tobytes("raw", "A")
                alphaimage = Image.frombytes("L", fimg.size, alphadata)
                im_a =im_a.resize((250,250))
                v1 = randint(10, 150)
                v2 = randint(10, 150)
                size = (v1, v2)
                # print(size)
                bimg.paste(fimg, size, fimg)
                bimg.save(f'{save_dir}/Final_{i}_{k}_{j}.png', "PNG") #_{file_back}
                im_a.paste(alphaimage,size,alphaimage)
                im_a.save(f'{save_dir_mask}/Final_{i}_{k}_{j}.jpg', "JPEG")
                # print('Im_a: ',im_a.size)
                #                 # print(alphaimage.size)

        print("completed: ",i)
        i += 1

if __name__ == '__main__':
    # Calling main() function

    main()
