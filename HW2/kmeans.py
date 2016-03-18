import sys
import math
import random

def euclidean_distance(Point1,Point2):
    distance = 0.0
    squares = 0
    Point1 = Point1.split(",")
    Point2 = Point2.split(",")
    for i in range(numDimensions):
        squares+=math.pow((float(Point2[i])-float(Point1[i])),2)
    distance = math.sqrt(squares)
    return distance

def compute_centroid(Points):
    x = {}
    xvalue = []
    retString=""
    
    for i in range(numDimensions):
        x[i]= []
    
    for point in Points:
        for i in range(numDimensions):
            x[i].append(point.split(",")[i])

    for i in range(numDimensions):
        xvalue.append(sum(float(j) for j in x[i])/len(Points))
    
    for i in range(numDimensions-1):
        retString+=str(xvalue[i])+","

    retString+=str(xvalue[numDimensions-1])

    return  retString

numClusters = int(sys.argv[1])
init_method = sys.argv[2]
convergence_threshold = float(sys.argv[3])
max_iterations = int(sys.argv[4])
filename = sys.argv[5]

fin = open(filename,"r")
fout = open(filename+".output","w")

PointList = []
numPoints = 0
for line in fin.readlines():
    point = line.rstrip("\n")
    PointList.append(point)
    numPoints+=1

numDimensions = len(PointList[0].split(","))

if(numPoints < numClusters):
    print("Number of clusters cannot exceed the number of datapoints!!! \nExiting...")
    sys.exit()

Centroids = []
index = []
if(init_method == "first"):
    for i in range(numClusters):
        Centroids.append(PointList[i])

elif(init_method == "rand"):
    while(len(index)!=numClusters):
        num = random.randint(0,numPoints-1)
        if num not in index:
            index.append(num)
    for i in range(numClusters):
        Centroids.append(PointList[index[i]])

cluster = {}
for i in range(numClusters):
    cluster[i] = []

no_of_iterations = 0
changing = 1
Old_Centroids = Centroids[:]
sum_of_centroid_distance = 0.0

while(changing and no_of_iterations < max_iterations):
    for datapoint in PointList:
        dist_arr = []
        for centroid in Centroids:
            dist_arr.append(euclidean_distance(datapoint,centroid))
        clusterIndex = dist_arr.index(min(dist_arr))
        previousIndex = -1
        for i in range(len(cluster)):
            if(datapoint in cluster.values()[i]):
                previousIndex = i
        if(previousIndex < 0):
            cluster[clusterIndex].append(datapoint)
        elif(previousIndex >= 0 and previousIndex!=clusterIndex):
            cluster[clusterIndex].append(datapoint)
            cluster[previousIndex].remove(datapoint)

    for i in range(len(cluster)):
        Centroids[i] = compute_centroid(cluster[i])
    centroid_distance = []
    for i in range(len(Centroids)):
        centroid_distance.append(str(euclidean_distance(Old_Centroids[i],Centroids[i])))
    sum_of_centroid_distance = sum(float(d) for d in centroid_distance)
    no_of_iterations+=1
    Old_Centroids = Centroids[:]
    if(sum_of_centroid_distance < convergence_threshold):
        changing = 0

for centroid in Centroids:
    fout.write(centroid)
    fout.write("\n")
for datapoint in PointList:
    for i in range(len(cluster)):
        if(datapoint in cluster.values()[i]):
            fout.write(str(i))
            fout.write("\n")
fin.close()
fout.close()

