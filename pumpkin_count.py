import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

image = cv2.imread('in/cropped.png')
image_annotated = cv2.imread('in/cropped_and_anotated.png')

# Mask pumpkins in annotated image with red color
mask = cv2.inRange(image_annotated, (0, 0, 200), (0, 0, 255))
cv2.imwrite('out/mask.png',mask)
dst = cv2.GaussianBlur(image, (5, 5), 0)
annotated_pumpkins = cv2.bitwise_and(dst,dst,mask=mask)
cv2.imwrite('out/annotated_pumpkins.png',annotated_pumpkins)

# Do statistics on the annotated image on color of pumpkins in RGB
pumpkin_colors = []
for i in range(annotated_pumpkins.shape[0]):
    for j in range(annotated_pumpkins.shape[1]):
        if mask[i,j] == 255:
            pumpkin_colors.append(annotated_pumpkins[i,j])
pumpkin_colors = np.array(pumpkin_colors)
#print('Pumpkin colors in RGB:')
#print(pumpkin_colors)

# Create a combined histogram for RGB channels
histogram_b, _ = np.histogram(pumpkin_colors[:, 0], bins=256, range=(0, 256))
plt.plot(histogram_b, color='b', label='Blue')
histogram_g, _ = np.histogram(pumpkin_colors[:, 1], bins=256, range=(0, 256))
plt.plot(histogram_g, color='g', label='Green')
histogram_r, _ = np.histogram(pumpkin_colors[:, 2], bins=256, range=(0, 256))
plt.plot(histogram_r, color='r', label='Red')
plt.xlim([0, 256])
plt.title('Combined Histogram for RGB channels')
plt.xlabel('Pixel value')
plt.ylabel('Frequency')
plt.legend()
plt.savefig('out/combined_histogram.png')
plt.show()


# Find mean, standard deviation, and covariance of each color channel of the pumpkins
mean = np.mean(pumpkin_colors, axis=0)
std = np.std(pumpkin_colors, axis=0)
cov = np.cov(pumpkin_colors.T)
print('Mean of pumpkin colors in RGB:')
print(mean)
print('Standard deviation of pumpkin colors in RGB:')
print(std)
print('Covariance of pumpkin colors in RGB:')
print(cov)

# Using the statistical information find the pumpkins in the same color range on the original image
mask = cv2.inRange(image, mean - 2*std, mean + 2*std)
cv2.imwrite('out/mask.png',mask)
pumpkins = cv2.bitwise_and(image,image,mask=mask)
cv2.imwrite('out/pumpkins.png',pumpkins)

# Do the above but with the CieLAB color space
image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
cv2.imwrite('out/image_cielab.png',image_lab)
annotated_pumpkins_lab = cv2.cvtColor(annotated_pumpkins, cv2.COLOR_BGR2LAB)
cv2.imwrite('out/annotated_pumpkins_cielab.png',annotated_pumpkins_lab)

# Do statistics on the annotated image on color of pumpkins in CieLAB
pumpkin_colors_lab = []
for i in range(annotated_pumpkins_lab.shape[0]):
    for j in range(annotated_pumpkins_lab.shape[1]):
        if mask[i,j] == 255:
            pumpkin_colors_lab.append(annotated_pumpkins_lab[i,j])
pumpkin_colors_lab = np.array(pumpkin_colors_lab)
print('Pumpkin colors in CieLAB:')
print(pumpkin_colors_lab)

# Create a 3D scatter plot for L*, a*, and b* channels in CieLAB
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pumpkin_colors_lab[:, 0]*(100/255), pumpkin_colors_lab[:, 1]-128, pumpkin_colors_lab[:, 2]-128, c=pumpkin_colors_lab / 255.0, marker='o')
ax.set_xlabel('L*')
ax.set_ylabel('a*')
ax.set_zlabel('b*')
ax.set_title('3D Scatter Plot for L*, a*, and b* channels in CieLAB')
ax.set_xlim([0, 100])
ax.set_ylim([-127, 128])
ax.set_zlim([-127, 128])
plt.savefig('out/cielab_scatter.png')
plt.show()
# Find mean, standard deviation, and covariance of each color channel of the pumpkins in CieLAB
pumpkin_colors_lab_normal = [pumpkin_colors_lab[:, 0]*(100/255), pumpkin_colors_lab[:, 1]-128, pumpkin_colors_lab[:, 2]-128]
pumpkin_colors_lab_normal = np.array(pumpkin_colors_lab_normal).T
#print('Pumpkin colors in CieLAB normalized:')
#np.set_printoptions(threshold=np.inf)
#print(pumpkin_colors_lab_normal)

# Get the mean, standard deviation and covariance in the actual ranges of CieLAB instead of 0 - 255
mean_lab = np.mean(pumpkin_colors_lab_normal, axis=0)
std_lab = np.std(pumpkin_colors_lab_normal, axis=0)
cov_lab = np.cov(pumpkin_colors_lab_normal.T)
print('Mean of pumpkin colors in CieLAB:')
print(mean_lab)
print('Standard deviation of pumpkin colors in CieLAB:')
print(std_lab)
print('Covariance of pumpkin colors in CieLAB:')
print(cov_lab)

# Get the mean, standard deviation and covariance in ranges of 0 - 255 to make sure the statistics are workin properly
mean_lab = np.mean(pumpkin_colors_lab, axis=0)
std_lab = np.std(pumpkin_colors_lab, axis=0)
cov_lab = np.cov(pumpkin_colors_lab.T)

# Using the statistical information find the pumpkins in the same color range on the original image in CieLAB
mask_lab = cv2.inRange(image_lab, mean_lab - 2*std_lab, mean_lab + 2*std_lab)
cv2.imwrite('out/mask_lab.png',mask_lab)
pumpkins_lab = cv2.bitwise_and(image,image,mask=mask_lab)
cv2.imwrite('out/pumpkins_lab.png',pumpkins_lab)

# Use mahlanobis distance using mean as reference color in BGR color space to find the pumpkins
def mahalanobis_distance(x, mean, cov):
    return np.sqrt(np.dot(np.dot((x-mean), np.linalg.inv(cov)), (x-mean)))

mask = np.zeros(image.shape[:2], dtype=np.uint8)
for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        if mahalanobis_distance(image[i,j], mean, cov) < 3.5:
            mask[i,j] = 255
cv2.imwrite('out/mask_mahalanobis.png',mask)
pumpkins = cv2.bitwise_and(image,image,mask=mask)
cv2.imwrite('out/pumpkins_mahalanobis.png',pumpkins)

# Morphological filtering the image
kernel = np.ones((4, 4), np.uint8)
closed_image = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
cv2.imwrite("out/closed.png", closed_image)

# Locate contours.
contours, hierarchy = cv2.findContours(closed_image, cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE)

# Draw a circle above the center of each of the detected contours.
# Find contours
pumpkins = 0

for contour in contours:
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = 0, 0

    if 0 <= cX < image.shape[1] and 0 <= cY < image.shape[0]:
        (x,y),radius = cv2.minEnclosingCircle(contour)
        center = (int(x),int(y))
        radius = int(radius)
        cv2.circle(image,center,radius+4,(0,0,255),2)
        contour_size = cv2.contourArea(contour)
        pumpkins += 1
            
print("Number of detected pumpkins: %d" % len(contours))

cv2.imwrite("out/located-pumpkins.png", image)
