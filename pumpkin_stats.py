import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

def pumpkin_mask(image, image_annotated, nb):
    # Mask pumpkins in annotated image with red color
    mask = cv2.inRange(image_annotated, (0, 0, 200), (0, 0, 255))
    cv2.imwrite('in/mask_' + str(nb) + '.png',mask)
    dst = cv2.GaussianBlur(image, (5, 5), 0)
    image_segmented = cv2.bitwise_and(dst,dst,mask=mask)
    cv2.imwrite('out/annotated_pumpkins' + str(nb) + '.png',image_segmented)

    return mask, image_segmented

def pumpkin_statistics(mask, image_segmented):
    # Find connected components in the mask
    num_labels, labels_im = cv2.connectedComponents(mask)
    
    pumpkin_colors = []
    pumpkin_sizes = []
    
    for label in range(1, num_labels):  # Skip the background label 0
        pumpkin_pixels = image_segmented[labels_im == label]
        pumpkin_colors.append(pumpkin_pixels)
        pumpkin_sizes.append(len(pumpkin_pixels))
    
    # Flatten the list of pumpkin colors
    pumpkin_colors = np.vstack(pumpkin_colors)
    
    # Find mean, standard deviation, and covariance of each color channel of the pumpkins
    mean = np.mean(pumpkin_colors, axis=0)
    cov = np.cov(pumpkin_colors.T)
    print('Mean of pumpkin colors in RGB:')
    print(mean)
    print('Covariance of pumpkin colors in RGB:')
    print(cov)
    print('Pumpkin sizes:')
    print(pumpkin_sizes)

    return mean, cov, pumpkin_sizes

means = []
covs = []
annotated_sizes = []
for i in range(1,6):
    print(i)
    image = cv2.imread("in/cropped_" + str(i) + ".png")
    image_annotated = cv2.imread("in/cropped_" + str(i) + "_annotated.png")

    mask, image_segmented = pumpkin_mask(image,image_annotated,i)
    mean, cov, pumpkin_sizes = pumpkin_statistics(mask,image_segmented)
    means.append(mean)
    covs.append(cov)
    annotated_sizes.extend(pumpkin_sizes)

means = np.array(means)
covs = np.array(covs)
avg_pumpkin_size = np.mean(annotated_sizes)
overall_mean = np.mean(means,axis=0)
overall_cov = np.mean(covs,axis=0)
print('Overall mean of pumpkin colors in RGB:')
print(overall_mean)
print('Overall covariance of pumpkin colors in RGB:')
print(overall_cov)
print('Average pumpkin size:', avg_pumpkin_size)

# write mean and covariance in file to be read later
with open('out/pumpkin_stats.txt', 'w') as f:
    f.write('Overall mean of pumpkin colors in RGB:\n')
    f.write(str(overall_mean) + '\n')
    f.write('Overall covariance of pumpkin colors in RGB:\n')
    f.write(str(overall_cov) + '\n')
    f.write('Average size of a pumpkin:\n')
    f.write(str(avg_pumpkin_size) + '\n')


#Use mahlanobis distance using mean as reference color in BGR color space to find the pumpkins
def mahalanobis_distance(x, mean, cov):
    return np.sqrt(np.dot(np.dot((x-mean), np.linalg.inv(cov)), (x-mean)))

def find_pumpkins(image, mean, cov):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if mahalanobis_distance(image[i,j], mean, cov) < 8.5:
                mask[i,j] = 255
    return mask

def circle_pumpkins(image, mask, chunk_number):
    # Morphological filtering the image
    kernel = np.ones((2,2), np.uint8)
    closed_image = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite("out/masked-pumpkins-in-chunk-" + str(chunk_number) + ".png", mask)
    cv2.imwrite("out/closed-pumpkins-in-chunk-" + str(chunk_number) + ".png", closed_image)

    # Find contours
    pumpkins = 0
    contours, _ = cv2.findContours(closed_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        (x,y),radius = cv2.minEnclosingCircle(contour)
        center = (int(x),int(y))
        radius = int(radius)
        cv2.circle(image,center,radius+4,(0,0,255),2)
        contour_size = cv2.contourArea(contour)
        
        # Compare contour size with average annotated size to determine the number of pumpkins
        if contour_size > (avg_pumpkin_size+8):
            pumpkins += contour_size // (avg_pumpkin_size+8)
        else:
            pumpkins += 1

    print("Number of pumpkins in chunk", chunk_number, ":", pumpkins)
    # Save the image with chunk number appended to the name
    cv2.imwrite("out/located-pumpkins-in-chunk-" + str(chunk_number) + ".png", image)
    return pumpkins

def compute_statistics(true_counts, detected_counts):
    true_counts = np.array(true_counts)
    detected_counts = np.array(detected_counts)
    
    absolute_errors = np.abs(detected_counts - true_counts)
    relative_errors = (absolute_errors / true_counts) * 100
    mse = mean_squared_error(true_counts, detected_counts)
    r2 = r2_score(true_counts, detected_counts)
    correlation = np.corrcoef(true_counts, detected_counts)[0, 1]
    
    stats = {
        "Mean Absolute Error": np.mean(absolute_errors),
        "Mean Relative Error (%)": np.mean(relative_errors),
        "Mean Squared Error": mse,
        "R-squared Score": r2,
        "Correlation Coefficient": correlation
    }
    
    return stats

def plot_comparison(true_counts, detected_counts):
    plt.figure(figsize=(8,6))
    plt.scatter(true_counts, detected_counts, color='blue', label='Detected vs True')
    plt.plot([min(true_counts), max(true_counts)], [min(true_counts), max(true_counts)], 'r--', label='Perfect Match')
    plt.xlabel("True Pumpkin Count")
    plt.ylabel("Detected Pumpkin Count")
    plt.title("Comparison of Detected and True Pumpkin Counts")
    plt.legend()
    plt.grid()
    plt.show()

pumpkin_count = [13, 35, 31, 15, 8]
detected_pumpkins = []
for i in range(1,6):
    print(i)
    image = cv2.imread("in/orthomosaik_cropped_" + str(i) + ".tif")

    mask = find_pumpkins(image,overall_mean,overall_cov)
    detected_pumpkins.append(circle_pumpkins(image,mask,i))
    print("Difference in pumkins in chunk", i, ":", abs(detected_pumpkins[i-1] - pumpkin_count[i-1]))

stats = compute_statistics(pumpkin_count, detected_pumpkins)
print("Statistics:", stats)
plot_comparison(pumpkin_count, detected_pumpkins)