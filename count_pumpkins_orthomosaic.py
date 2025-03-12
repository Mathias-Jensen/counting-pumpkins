import numpy as np
import cv2
from osgeo import gdal
from scipy.spatial import KDTree

# Read mean and covaraicne from file where mean is in the second line and covariance is a matrix from line 4 to 6
with open('in/pumpkin_stats.txt', 'r') as f:
    f.readline()
    mean = np.array(list(map(float, f.readline().strip().split())))
    f.readline()
    cov = np.array([list(map(float, f.readline().strip().split())) for _ in range(3)])
    f.readline()
    avg_pumpkin_size = float(f.readline().strip())
print(mean)
print(cov)
print(avg_pumpkin_size)

# Use mahlanobis distance using mean as reference color in BGR color space to find the pumpkins
def mahalanobis_distance(x, mean, cov):
    return np.sqrt(np.dot(np.dot((x-mean), np.linalg.inv(cov)), (x-mean)))

def mask_pumpkins(image, mean, cov):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if mahalanobis_distance(image[i,j], mean, cov) < 7.5:
                mask[i,j] = 255
    return mask

def detect_pumpkins(mask,chunk_number):
    # Morphological filtering the image
    kernel = np.ones((2,2), np.uint8)
    closed_image = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    #cv2.imwrite("masked-pumpkins-in-chunk-" + str(chunk_number) + ".png", mask)
    #cv2.imwrite("closed-pumpkins-in-chunk-" + str(chunk_number) + ".png", closed_image)

    # Find contours
    contours, _ = cv2.findContours(closed_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Compute centroids of detected pumpkins
    centroids = []
    filtered_contours = []  # Store contours that have centroids
    for cnt in contours:
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centroids.append((cx, cy))
            filtered_contours.append(cnt)  # Add contour to filtered_contours
    
    print("Number of pumpkins in chunk", chunk_number, ":", len(centroids))

    return centroids, filtered_contours

def filter_duplicate_centroids(centroid_list, contour_list, distance_threshold=5):
    """Filter out duplicate centroids using KDTree for fast nearest neighbor search."""
    if not centroid_list:
        return []

    # Convert list of centroids to numpy array
    centroids = np.array(centroid_list)

    # Use KDTree to efficiently find close neighbors
    tree = KDTree(centroids)
    unique_centroids = []
    unique_contours = []
    visited = set()

    for i, centroid in enumerate(centroids):
        if i in visited:
            continue

        # Find neighbors within the distance threshold
        neighbors = tree.query_ball_point(centroid, distance_threshold)

        # Keep only one centroid per group of duplicates (e.g., first occurrence)
        unique_centroids.append(tuple(centroid))
        unique_contours.append(contour_list[i])  # Keep the corresponding contour
        visited.update(neighbors)

    return unique_centroids, unique_contours

def count_pumpkins_from_contours(contours, avg_pumpkin_size, size_threshold=8):
    """Count the number of pumpkins based on contour area and average pumpkin size."""
    total_pumpkins = 0  # Initialize the number of pumpkins

    for contour in contours:
        # Calculate the contour size (area) to estimate the number of pumpkins
        contour_size = cv2.contourArea(contour)
        
        # Compare contour size with average annotated size to determine the number of pumpkins
        if contour_size > (avg_pumpkin_size + size_threshold):  # Threshold for larger contours
            pumpkins_in_contour = contour_size // (avg_pumpkin_size + size_threshold)
            total_pumpkins += pumpkins_in_contour
        else:
            total_pumpkins += 1  # If the contour is small enough, consider it a single pumpkin

    return total_pumpkins

def draw_circles_on_chunk(chunk, chunk_x, chunk_y, contours, centroids, drawn, color=(0, 0, 255), thickness=2):
    """Draw circles around detected pumpkins on the chunk."""
    drew = []
    for i, contour in enumerate(contours):
        if i not in drawn:
            # Convert global coordinates to local chunk coordinates
            local_x, local_y = int(centroids[i][0]) - chunk_x, int(centroids[i][1]) - chunk_y
            if 0 <= local_x < chunk.shape[1] and 0 <= local_y < chunk.shape[0]:  # Ensure within chunk bounds
                drew.append(i)
                (x, y), radius = cv2.minEnclosingCircle(np.array(contour))
                cv2.circle(chunk, (local_x, local_y), int(radius)+5, color, thickness)

    return chunk, drew


def get_chunks_with_overlap(dataset, tile_size=512, overlap=50):
    img_width = dataset.RasterXSize
    img_height = dataset.RasterYSize
    
    # List to store chunk bounding boxes (x_offset, y_offset, width, height)
    chunks = []
    
    for y in range(0, img_height, tile_size - overlap):
        for x in range(0, img_width, tile_size - overlap):
            # Ensure the chunk does not go beyond the image boundaries
            width = min(tile_size, img_width - x)
            height = min(tile_size, img_height - y)
            
            chunks.append((x, y, width, height))
    
    return chunks

def BGR_chunk(chunk):
    # Handle different band formats
    if bands == 1:
        chunk = chunk.astype(np.uint8)  # Grayscale
    else:
        chunk = np.moveaxis(chunk, 0, -1)  # Move bands to last axis (H, W, C)
        chunk = chunk.astype(np.uint8)

    chunk = chunk[:,:,:3] # Keep first 3 bands (RGB)
    chunk = cv2.cvtColor(chunk, cv2.COLOR_RGB2BGR) # Convert to BGR

    return chunk

# Open the raster file
raster_path = "in/Gyldensteensvej-9-19-2017-orthophoto_cropped.tif"
dataset = gdal.Open(raster_path, gdal.GA_ReadOnly)

# Get image dimensions
bands = dataset.RasterCount

# Define chunk size
chunk_size = 512  # Size of each chunk in pixels'
chunk_number = 0

chunks = get_chunks_with_overlap(dataset, tile_size=512, overlap=50)
all_centroids = []
all_contours = []  # New list to store all contours

# Loop through chunks and process them
for (x, y, w, h) in chunks:
    chunk = dataset.ReadAsArray(x, y, w, h)
    chunk = BGR_chunk(chunk)
    chunk_number += 1
    print("Processing chunk", chunk_number)
    mask = mask_pumpkins(chunk, mean, cov)
    centroids, contours = detect_pumpkins(mask,chunk_number)

    # Convert to global coordinates
    global_centroids = [(cx + x, cy + y) for cx, cy in centroids]
    all_centroids.extend(global_centroids)

    # Store the contours along with the centroids (with global coordinates)
    all_contours.extend(contours)

print("Number of contours before filter:", len(all_contours))
print("Number of pumpkins before filtering:", count_pumpkins_from_contours(all_contours, avg_pumpkin_size))

unique_centroids, unique_contours = filter_duplicate_centroids(all_centroids,all_contours,0)

print("Number of contours after filtering:", len(unique_contours))
print("Number of pumpkins:", count_pumpkins_from_contours(unique_contours, avg_pumpkin_size))

chunk_number = 0
drawn = []

# Loop through chunks and process them
for (x, y, w, h) in chunks:
    chunk = dataset.ReadAsArray(x, y, w, h)
    chunk = BGR_chunk(chunk)
    chunk, drew = draw_circles_on_chunk(chunk, x, y, unique_contours, unique_centroids, drawn)
    chunk_number += 1
    drawn.extend(drew)
    cv2.imwrite("out/located-pumpkins-in-chunk-" + str(chunk_number) + ".png", chunk)

# Clean up
cv2.destroyAllWindows()
dataset = None  # Close the GDAL dataset