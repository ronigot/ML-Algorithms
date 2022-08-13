from matplotlib import pyplot as plt
import numpy as np
import sys


image_fname, centroids_fname, out_fname = sys.argv[1], sys.argv[2], sys.argv[3]
z = np.loadtxt(centroids_fname)  # load centroids

orig_pixels = plt.imread(image_fname)
pixels = orig_pixels.astype(float)/255
# Reshape the image(128x128x3) into an Nx3 matrix where N = number of pixels.
pixels = pixels.reshape(-1, 3)


# A function that calculates the new centroids
def findCenters(pixelsArray, centroidsArray, clusterArray, k):
    centers = np.zeros((k, 3))
    for n in range(k):
        # We check the number of pixels in each cluster and sum them up
        counter = sum = 0
        for index, c in enumerate(clusterArray):
            if c == n:
                counter += 1
                sum += pixelsArray[index]
        # if counter != 0, divide the sum by the counter to find the new midpoint of the cluster
        if counter != 0:
            centers[n] = (sum / counter)
        else:
            # if counter == 0, the centroid remains the same
            centers[n] = (centroidsArray[n])
    return centers


# A function that receives two lists and checks if they are identical
def compareLists(list1, list2):
    for inx in range(len(list1)):
        for a in range(3):
            if list1[inx][a] != list2[inx][a]:
                return False
    return True


# A function that calculates the loss values
def loss(centroids, pixels, clusters):
    sum, count = 0, 0
    for i, cent in enumerate(centroids):
        for j in range(len(clusters)):
            cluster = clusters[j]
            if cluster == i:
                sum += pow(pixels[j][0] - cent[0], 2) + pow(pixels[j][1] - cent[1], 2) + \
                       pow(pixels[j][2] - cent[2], 2)
                count += 1
    average = sum / count
    return average


outFile = open(out_fname, 'w')
cluster = np.zeros(pixels.shape[0])
# Counter for the number of iterations of the loop
count = 0
# A list to save the loss values
cost = []
# If we get to 20 iterations, we'll stop the loop run
while count < 20:
    # Assign each pixel to the appropriate cluster
    for i, pixel in enumerate(pixels):
        minDist = -1
        for j, centroid in enumerate(z):
            dist = np.sqrt(pow(centroid[0] - pixel[0], 2) + pow(centroid[1] - pixel[1], 2)
                           + pow(centroid[2] - pixel[2], 2))
            if dist < minDist or minDist == -1:
                minDist = dist
                cluster[i] = j
    # Finding the new centroids
    new = findCenters(pixels, z, cluster, len(z)).round(4)
    # Add the loss value to the list
    cost.insert(len(cost), loss(new, pixels, cluster))
    # Check if the list of new centroids is the same as the old ones
    if compareLists(z, new):
        outFile.write(f"[iter {count}]:{','.join([str(i) for i in new])}\n")
        # If the new centroids are the same as the old ones, we stop the loop run
        break
    else:
        outFile.write(f"[iter {count}]:{','.join([str(i) for i in new])}\n")
        # Updating the centroids to the new ones
        z = new
    count += 1

outFile.close()
