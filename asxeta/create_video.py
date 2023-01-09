import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import einops
import matplotlib.animation as animation
from tqdm import tqdm

data = pd.read_csv("~/Documents/Sleep_Project/dataframe.csv", index_col=False)

#Single Plots
data_columns = [label for label in data.keys() if not label.isnumeric()]
for column in  data_columns:
    if column == "label" or column == "full_label" or column == "datetime" or column == "image":
        continue
    this_column = data[column].value
    plt.figure()
    plt.plot(this_column)
    plt.xlabel("time")
    plt.ylabel("value")
    plt.title(column)

#Video
signals = [np.expand_dims(np.array(data["{}".format(i)]), axis=0) for i in range(616)]
signals = np.concatenate(signals,axis=0)
signals = einops.rearrange(signals,"(a b) c -> c a b", a=14, b=44)
fig, ax = plt.subplots()
images = [[ax.imshow(signals[i])] for i in tqdm(range(len(signals)), "Create Image Plots")]
ani = animation.ArtistAnimation(fig, images, blit=True)
writer = animation.FFMpegWriter(fps=1000, extra_args=['-vcodec', 'libx264'])
ani.save('filename.mp4', writer=writer)



#video with images

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from os import listdir
from os.path import isfile, join

mypath = "/users/sista/kkontras/Documents/Sleep_Project/experiments/Broken_Mod_Imgs"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
imgs = []

for i in onlyfiles:
    imgs.append(mpimg.imread(mypath + "/" + i))

fig, ax = plt.subplots()
images = [[ax.imshow(imgs[i])] for i in range(len(imgs))]
ani = animation.ArtistAnimation(fig, images, blit=True)
writer = animation.FFMpegWriter(fps=1, extra_args=['-vcodec', 'libx264'])
ani.save('bad_perf_imgs.mp4', writer=writer)