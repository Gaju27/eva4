import cv2
import glob
for file in glob.glob('./Images/*.*'):
    img = cv2.imread(file, cv2.IMREAD_UNCHANGED)

    print('Original Dimensions : ', img.shape)

    width = 162
    height = 162
    dim = (width, height)
    # resize image
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    status = cv2.imwrite(file, resized)
    print("Image written to file-system : ",status)