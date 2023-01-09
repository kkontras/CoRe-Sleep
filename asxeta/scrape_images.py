from os import listdir
from os.path import isfile, join

# %matplotlib inline
# %config InlineBackend.figure_format = ‘retina’

import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
# from torchvision.models import resnet50, ResNet50_Weights
# import helper
import einops
from PIL import Image, ImageEnhance
from scipy.cluster.hierarchy import complete, fcluster
import scipy.cluster.hierarchy as shc
import numpy as np

# mypath = "/users/sista/kkontras/Documents/Sleep_Project/google-images-download/google_images_download/downloads/jack daniels posters"
# mypath = "/users/sista/kkontras/Documents/Sleep_Project/google-images-download/google_images_download/downloads/l'oreal posters"
mypath = "/users/sista/kkontras/Documents/Sleep_Project/google-images-download/google_images_download/downloads/nike posters"


transform = transforms.Compose([transforms.Resize(255),
                                transforms.CenterCrop(255),
                                transforms.ToTensor()])

dataset = datasets.ImageFolder(mypath, transform=transform)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=True)

images, labels = next(iter(dataloader))
print(images.shape)

from torchvision import models
model = models.resnet50(pretrained=True)
# model = models.densenet121(pretrained=True)
features = model(images)

print(features.shape)


clusters = shc.linkage(features.detach().numpy(), method="ward", metric="euclidean")
shc.dendrogram(Z=clusters)
plt.show()

thresholds = np.array([c[2] for c in clusters])

depth_of_clustering = 8
bigger_cluster_threshold = thresholds[-depth_of_clustering] - (thresholds[-depth_of_clustering]-thresholds[-depth_of_clustering-1])/2

cluster_of_each_img = fcluster(clusters, bigger_cluster_threshold, criterion='distance')
cluster_numbers, images_per_class = np.unique(cluster_of_each_img, return_counts=True)

for c_num in cluster_numbers:
    c = 255*0
    new = Image.new("RGBA", (255 * (1+images_per_class[c_num-1]//5), 255 * 5))
    for img_num, img in enumerate(images[cluster_of_each_img==c_num]):
        img_num_h = img_num // 5
        img_num_w = img_num % 5
        print(img_num_h, img_num_w)
        new.paste(Image.fromarray((einops.rearrange(img, "a b c -> b c a").numpy()*255).astype(np.uint8)), (255*img_num_h,255*img_num_w))
    new.save("/users/sista/kkontras/Documents/Sleep_Project/asxeta/yo_{}.png".format(c_num))
#
#
for c_num in cluster_numbers:
    # plt.subplot(int("{}{}{}".format(len(cluster_numbers), c_num, 1)))
    img = plt.imread("/users/sista/kkontras/Documents/Sleep_Project/asxeta/yo_{}.png".format(c_num))
    plt.imshow(img)
    plt.axis("off")
    plt.show()
