import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('training_images.csv', delimiter=',')
print(data.shape)

# Reshape each row vector of the original data set, so we have a 300x300 array as input for imshow()
h = 300
w = 300

img = np.reshape(data[0, :], [h, w], order='F')

# show image
plt.imshow(img, cmap='gray')
plt.show()

