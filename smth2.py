from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
from skimage.util import invert
import cv2

test_original = cv2.imread("fingerprints/100__M_Left_index_finger.tif")
# Invert the horse image
#image = invert(test_original)

# perform skeletonization
#skeleton = skeletonize(image)
skeleton = skeletonize(test_original)

# display results
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4), sharex=True, sharey=True)

ax = axes.ravel()

ax[0].imshow(test_original, cmap=plt.cm.gray)
ax[0].axis('off')
ax[0].set_title('original', fontsize=20)

ax[1].imshow(skeleton, cmap=plt.cm.gray)
ax[1].axis('off')
ax[1].set_title('skeleton', fontsize=20)

fig.tight_layout()
plt.show()