import os
from PIL import Image
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
cnt = 0
for filename in os.listdir('dataset_old/cells/Q7_test/0'):
    cnt =cnt+1
    path='dataset_old/cells/Q7_test/0/'+filename
    im = Image.open(path)
    data = np.array(im, dtype=float)
    lowpass = ndimage.gaussian_filter(data,6)
    gauss_highpass = data - lowpass
    new_path = 'dataset_old/cells/Q7_test5/0/'+ filename + '.jpg'
    plt.imshow(gauss_highpass)
    plt.gray()
    plt.imsave(new_path,gauss_highpass)
