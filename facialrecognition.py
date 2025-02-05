import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('datasets/training_images.csv', delimiter=',')
print(data.shape)

# Define new array dims for Reshape
# Each row vector of the original data set corresponded to a flattened image,
# We want a 300x300 array of data for each image so the data for each image (row) can be fed to imshow()
h = 300
w = 300

# Show a few images
fig1, axs = plt.subplots(2, 4, figsize=(10, 5))
fig1.suptitle("8 Sample Training Data Images")
row_nums = np.array([0, 9, 19, 29, 40, 49, 59, 69])
for i in range(row_nums.size):
    img = np.reshape(data[row_nums[i], :], (h, w), order='F')
    axs[i // 4, i % 4].imshow(img, cmap='gray')
fig1.show()

# PART 1: TRAINING DATA SVD
# We want to distinguish a face amongst all 80 faces in the data set.
# 1. Average all images along each column.
img_avg = np.mean(data, axis=0)
plt.figure(figsize=(12, 12))
plt.imshow(np.reshape(img_avg, (h, w), order='F'), cmap='gray')
plt.title("Training Data Average Image")
plt.show()

# 2. To find how the images deviate from this average, first subtract from this average using the outer product.
X = data - np.ones((80, 1)) @ img_avg.reshape((1, -1))

# 3. Find the reduced SVD of X
U, s, Vt = np.linalg.svd(X, full_matrices=False)
V = Vt.T

# 4. Inspect the singular values
plt.plot(s, 'ko')
plt.title("Singular Values")
plt.show()

# Seeing that s values of 20-40 are most important, plot the scaled energies.
E = np.cumsum(s ** 2) / np.sum(s ** 2)
plt.plot(E, 'ko')
plt.title("Scaled Energies")
plt.show()

# Inspect how many singular values account for at least 99% of the total energy.
print('Number of Singular Values to Account for 99% of Energies', E[70])

# 4. Most eigenfaces (principal values) will be important since we need most or all of the singular values.
# Inspect the first few.
fig2, axs = plt.subplots(3, 3, figsize=(12, 12))
fig2.suptitle("First 8 EigenFacess")
for i in range(9):
    img = np.reshape(V[:, i], (h, w), order='F')
    axs[i // 3, i % 3].imshow(img, cmap='gray')
fig2.show()

# 5. Combine U and sigma to get score matrix
score_array = U @ np.diag(s)

# 6. Reconstruct photos in fig1 by firstly reconstructing X (scores * Vt) and then adding back the average face to each.
X2 = score_array @ V.T
fig3, axs = plt.subplots(2, 4, figsize=(10, 5))
fig3.suptitle("8 Reconstructed Original Images")
row_nums = np.array([0, 9, 19, 29, 40, 49, 59, 69])
for i in range(row_nums.size):
    img = X2[row_nums[i], :] + img_avg
    img = np.reshape(img, (h, w), order='F')
    axs[i // 4, i % 4].imshow(img, cmap='gray')
fig3.show()

# PART 2: TESTING DATA FACIAL RECOGNITION. Each row of the testing data has an image of Bush or Williams.
test_data = np.genfromtxt('datasets/testing_images.csv', delimiter=',')
img1 = test_data[0, :]
img2 = test_data[1, :]

# 1. Calculate and find the minimum difference between each row of scores and each row vector of img1 and 2's scores.
score1 = (img1 - img_avg).reshape(1, -1) @ V
score2 = (img2 - img_avg).reshape(1, -1) @ V
distances1 = np.zeros(80)
distances2 = np.zeros(80)

for i in range(80):
    distances1[i] = np.linalg.norm(score1[0, :] - score_array[i, :])  # Using Euclidean distance
    distances2[i] = np.linalg.norm(score2[0, :] - score_array[i, :])

index1 = np.argmin(distances1)

# 2. Find the index in the original data where this lowest difference occurs, for each of the two testing images.
print(index1)  # Image 1 is closet to the training data's index1=30 image, which is 31;
index2 = np.argmin(distances2)
print(index2)   # This image is closest to the training data's [51, :], image 52.

# 3. Based on index1 and index2, img1 should be a photo of Bush and img2 of Williams.
# Check by reconstructing the two corresponding original images and then compare to test image.
img = X[index1, :] + img_avg
fig4, axs = plt.subplots(1, 2, figsize=(8, 4))
fig4.suptitle("Comparison for Test Image 1 Facial Recognition")
axs[0].imshow(np.reshape(img, (h, w), order='F'), cmap='gray')
axs[0].set_title("Training Data Image 31")
axs[1].imshow(np.reshape(img1, (h, w), order='F'), cmap='gray')
axs[1].set_title("Test Image 1")
fig4.tight_layout()
fig4.show()

img = X[index2, :] + img_avg
fig5, axs = plt.subplots(1, 2, figsize=(8, 4))
fig5.suptitle("Comparison for Test Image 2 Facial Recognition")
axs[0].imshow(np.reshape(img, (h, w), order='F'), cmap='gray')
axs[0].set_title("Training Data Image 52")
axs[1].imshow(np.reshape(img2, (h, w), order='F'), cmap='gray')
axs[1].set_title("Test Image 2")
fig5.tight_layout()
fig5.show()

