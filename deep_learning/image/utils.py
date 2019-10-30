# -*- coding: utf8 -*-
# Created by: wuzewei
# Created by: 2019/10/30
import numpy as np
from matplotlib import pyplot as plt


def peek_some_pictures(filenames, num, figsize):
    tmp_filenames = np.random.choice(filenames, num)
    figure = plt.figure(figsize=figsize)
    nrows = int(np.ceil(np.sqrt(num)))
    ncols = nrows
    plt.axis("off")
    for idx, tmp_filename in enumerate(tmp_filenames):
        img_data =  plt.imread(tmp_filename)
        ax = figure.add_subplot(nrows, ncols, idx+1)
        ax.imshow(img_data)
        ax.set_title(tmp_filename.split('\\')[-1])
        plt.axis('off')
    plt.show()
    return


if __name__ == '__main__':
    pass

