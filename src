# -*- coding: utf-8 -*-
"""f
Created on Thu Oct 13 21:36:23 2022

@author: Daniel and Joep 
"""
 
# Running time stats (~):
# Processing training images: 2:00
# Processing test images: 0:40
 
# general imports
import cv2
import numpy as np
# setting in numpy to surpress scientific notation in console
np.set_printoptions(suppress=True)
 
# imports for lbp
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
from skimage import feature
import os
 
directory = 'C:/Users/USER/Desktop/BloememHerkennen/BloemenHerkennenFotos'
DirEllenHoustonTrain = f'{directory}/Dahlia ellen houston/train'
DirEllenHoustonTest = f'{directory}/Dahlia ellen houston/test'
DirHarlequinTrain = f'{directory}/Dahlia Harlequin/train'
DirHarlequinTest = f'{directory}/Dahlia Harlequin/test'
DirIzarraTrain = f'{directory}/Dahlia Izarra/train'
DirIzarraTest = f'{directory}/Dahlia Izarra/test'
DirWinkTrain = f'{directory}/Dahlia Wink/train'
DirWinkTest = f'{directory}/Dahlia wink/test'
DirZinniaTrain = f'{directory}/Zinnia/train'
DirZinniaTest =  f'{directory}/Zinnia/test'
DirRudbeckiaFulgidaTrain = f'{directory}/Rudbeckia Fulgida/train'
DirRudbeckiaFulgidaTest = f'{directory}/Rudbeckia Fulgida/test'
DirDahliaBishopofYorkTrain = f'{directory}/Dahlia Bishop of York/train'
DirDahliaBishopofYorkTest = f'{directory}/Dahlia Bishop of York/test'
DirNieuweFoto = f'{directory}/Nieuwe foto'

"""
Nummering | Bloemsoort
-------------------------
   [0]    | Ellen Houston
   [1]    | Harlequin
   [2]    | Izarra
   [3]    | Wink
   [4]    | Zinnia
   [5]    | RudbeckiaFulgida
   [6]    | DahliaBishopofYork
"""

TESTFOTO = 0
TEST = False

SCALE = 5 # resize scale for original image resolution
BLUR = 3 # blur of Image
CR_SCALE = 2 # resize scale for cropped image resolution
PARAMETER_COUNT = 7 # amount of Data points used in KNN
DEADZONE_BGR_FILTER = 12 # deadzone used in BGR filter
 
 
def load_all_images():# Function that reads image
    EllenHouston = 0
    Harlequin = 1
    Izarra = 2
    Wink = 3
    Zinna = 4
    Rudbeckia = 5
    BishopOfYork = 6
    
    # Read all image directories
    EllenHousonTrainArray = read_img(DirEllenHoustonTrain)
    HarlequinTrainArray = read_img(DirHarlequinTrain)
    IzarraTrainArray = read_img(DirIzarraTrain)
    WinkTrainArray = read_img(DirWinkTrain)
    ZinniaTrainArray = read_img(DirZinniaTrain)
    RudbeckiaFulgidaTrainArray = read_img(DirRudbeckiaFulgidaTrain)
    DahliaBishopofYorkTrainArray = read_img(DirDahliaBishopofYorkTrain)
 
    # Label each flower type 0-6
    LabelEllenHouston = np.full((len(EllenHousonTrainArray),1),EllenHouston, dtype = np.float32)
    LabelHarlequin = np.full((len(HarlequinTrainArray),1),Harlequin, dtype = np.float32)
    LabelIzarra = np.full((len(IzarraTrainArray),1),Izarra, dtype = np.float32)
    LabelWink = np.full((len(WinkTrainArray),1),Wink, dtype = np.float32)
    LabelZinnia = np.full((len(ZinniaTrainArray),1),Zinna, dtype = np.float32)
    LabelRudbeckiaFulgida = np.full((len(RudbeckiaFulgidaTrainArray),1),Rudbeckia, dtype = np.float32)
    LabelDahliaBishopofYork = np.full((len(DahliaBishopofYorkTrainArray),1),BishopOfYork, dtype = np.float32)
    
    # Create one label that fits knn
    TrainLabel = np.concatenate((LabelEllenHouston,LabelHarlequin,
                                 LabelIzarra,LabelWink,LabelZinnia,
                                 LabelRudbeckiaFulgida, LabelDahliaBishopofYork), axis = 0)
    TrainImages = EllenHousonTrainArray + HarlequinTrainArray + IzarraTrainArray + WinkTrainArray + ZinniaTrainArray + RudbeckiaFulgidaTrainArray + DahliaBishopofYorkTrainArray
 
    # Initialize an array that can be filled with obtained traindata
    TrainArray = [ [0 for _ in range(PARAMETER_COUNT)] for _ in range(len(TrainImages))]
 
    # Read all test images from folers
    EllenHousonTestArray = read_img(DirEllenHoustonTest)
    HarlequinTestArray = read_img(DirHarlequinTest)
    IzarraTestArray = read_img(DirIzarraTest)
    WinkTestArray = read_img(DirWinkTest)
    ZinniaTestArray = read_img(DirZinniaTest)
    RudbeckiaFulgidaTestArray = read_img(DirRudbeckiaFulgidaTest)
    DahliaBishopofYorkTestArray = read_img(DirDahliaBishopofYorkTest)
    NieuweFotoTestArray = read_img(DirNieuweFoto)
 
    # Label each flower type 0-6
    TestLabelEllenHouston = np.full((len(EllenHousonTestArray),1),EllenHouston, dtype = np.float32)
    TestLabelHarlequin = np.full((len(HarlequinTestArray),1),Harlequin, dtype = np.float32)
    TestLabelIzarra = np.full((len(IzarraTestArray),1),Izarra, dtype = np.float32)
    TestLabelWink = np.full((len(WinkTestArray),1),Wink, dtype = np.float32)
    TestLabelZinnia = np.full((len(ZinniaTestArray),1),Zinna, dtype = np.float32)
    TestLabelRudbeckiaFulgida = np.full((len(RudbeckiaFulgidaTestArray),1),Rudbeckia, dtype = np.float32)
    TestLabelDahliaBishopofYork = np.full((len(DahliaBishopofYorkTestArray),1),BishopOfYork, dtype = np.float32)
    
    TestLabelNieuweFoto = np.full((len(DirNieuweFoto),1),TESTFOTO, dtype = np.float32)
    
    if TEST == False:
        TestLabel = np.concatenate((TestLabelEllenHouston,TestLabelHarlequin,
                                    TestLabelIzarra,TestLabelWink,TestLabelZinnia,
                                    TestLabelRudbeckiaFulgida, TestLabelDahliaBishopofYork), axis = 0)
        
        TestImages = EllenHousonTestArray + HarlequinTestArray + IzarraTestArray + WinkTestArray + ZinniaTestArray + RudbeckiaFulgidaTestArray + DahliaBishopofYorkTestArray
    elif TEST == True:
        TestLabel = TestLabelNieuweFoto
        TestImages = NieuweFotoTestArray
    
    # Initialize an array that can be filled with obtained TestData
    TestArray = [ [0 for _ in range(PARAMETER_COUNT)] for _ in range(len(TestImages))]
    return TrainImages, TrainLabel, TrainArray, TestImages, TestLabel, TestArray
 
def read_img(directory):
    original_images = os.listdir(directory)
 
    # Load each file in the given directory
    original_images = [f"{directory}/{file_name}" for file_name in original_images]
    original_images = [cv2.imread(file) for file in original_images]
    return original_images
 
 
def resize_image(image):
    image = cv2.resize(image, None, fx=1/SCALE, fy=1/SCALE, interpolation=cv2.INTER_AREA)
    return image
 
def blur_image(image):
    image = cv2.medianBlur(image, BLUR)
    return image
 
def normalize_image(image):
    width, height, channels = image.shape
    norm = np.zeros((width, height))
    image = cv2.normalize(image, norm, 55, 200, cv2.NORM_MINMAX)
    return image
 
def bilateral_filter(image):
    for x in range(20):
        image = cv2.bilateralFilter(image, 15, 20, 20, cv2.BORDER_DEFAULT)
    return image
 
# Remove blue and green from image
def background_filter(image):
 
    h = image.shape[0]
    w = image.shape[1]
    for y in range(0,h):
        for x in range (0,w):
            b, g, r = image[y][x]
            if (g + DEADZONE_BGR_FILTER > b and g + DEADZONE_BGR_FILTER > r) or (b + DEADZONE_BGR_FILTER >g and b + DEADZONE_BGR_FILTER > r):
                image[y][x] = [0, 0, 0]
 
    # hueList = list()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
 
    lower_threshold = np.array([70,0,0], np.uint8)
    upper_threshold = np.array([280,20,50], np.uint8)
    mask = cv2.inRange(image, lower_threshold, upper_threshold)
    mask = cv2.bitwise_not(mask)
    result = cv2.bitwise_and(image, image, mask=mask)
 
    return result
 
 
def find_largest_contour(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    im, contours, hierarchy = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  
 
    # Flter out all the smallest contours
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 100 and h > 50:
            break    
 
    all_areas = []
 
    # Find the area of all found contours
    for cnt in contours:
        area = cv2.contourArea(cnt)
        all_areas.append(area)
 
    # Sort contours by size -> sorted_contours[0] is the largest   
    sorted_contours = sorted(contours, key = cv2.contourArea, reverse = True)
    largest_contour = sorted_contours[0]
 
    return largest_contour
 
# Draw contour on images for testing purposes
def draw_contour(image, largest_contour):
    cv2.drawContours(image, [largest_contour], 0, (0, 255, 0), 3)
 
# Scale contour back to original image
def scale_contour(cnt, SCALE):
    M = cv2.moments(cnt)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
 
    cnt_norm   = cnt - [cx, cy]
    cnt_scaled = cnt_norm * SCALE
    cnt_scaled = cnt_scaled + [(cx * SCALE), (cy * SCALE)]
    cnt_scaled = cnt_scaled.astype(np.int32)
 
    x, y, w, h = cv2.boundingRect(cnt_scaled)
 
    return cnt_scaled, x, y, w, h

# Crop out the kernal from the flower
def crop_kernal(image, x, y, w, h, percentage):
    print("X: ", x)
    print("Y: ", y)
    print("W: ", w)
    print("H: ", h)
    print("Percentage: ", percentage)
 
    percentage = percentage / 100 
    x_1 = round(x + (w * percentage))
    y_1 = round(y + (h * percentage))
    x_2 = round(x + (w - (percentage * w)))
    y_2 = round(y + (h - (percentage * h)))
 
    print("X_1: ", x_1)
    print("Y_1: ", y_1)
    print("X_2: ", x_2)
    print("Y_2: ", y_2)
    print("Percentage: ", percentage)
 
    cv2.rectangle(image, (x_1, y_1), (x_2, y_2), (0,0,0),3)
    cropped = image[y_1:y_2, x_1:x_2]
    return cropped

# crop the flower from the original image 
def crop_flower(image, x, y, w, h, contour):  
    mask = np.zeros(image.shape[:2], dtype=image.dtype)
    cv2.drawContours(mask, [contour], 0, (255), -1)
    image = cv2.bitwise_and(image,image, mask= mask)
 
    #cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),3)
    cropped = image[y:y+h, x:x+w]
    return cropped

# calcuate flower area based on rectangle
def flower_area(w, h):
    area_flower = w * h
    print("Area of flower (~):", area_flower)

# Detect kernal in image 
def kernal_detection(image, x, y):
    image = blur_image(image)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray_image,cv2.HOUGH_GRADIENT, 1, 50,
                                param1=50,param2=30,minRadius=0,maxRadius=0)
    circles = np.uint16(np.around(circles))
    for x, y, r in circles[0, :]:
        # draw the outer circle
        cv2.circle(image,(x, y),r,(0,255,0),2)
        # draw the center of the circle
        cv2.circle(image,(x, y),2,(0,0,255),3)
    cv2.imshow('Detected circles', image)
    return image

# count amount of leaves in the image 
def count_leaves(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cnts = cv2.findContours(gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]
    cv2.drawContours(image, cnts, -1, (0,255,0), 3)
    xcnts = []
    s1 = 800
    for cnt in cnts:
        if cv2.contourArea(cnt) > s1:
            xcnts.append(cnt)
 
    flower_leaves = int(len(xcnts))
    #print("Number of leaves (~):", flower_leaves) 
    return flower_leaves, image
 
# calculate mean color in the image
def avg_color(image): #contour
    channels = cv2.mean(image) #, mask
    h = channels[0]
    s = channels[1]
    v = channels[2]
    h = int(round(h,0))
    s = int(round(s,0))
    v = int(round(v,0))
    return h, s, v
 
def lbp(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 
    features = feature.local_binary_pattern(gray, 10, 5, method="default") # method="uniform")
    LbpFoto = features.astype("uint8")
    features_unique = len(np.unique(features))
    features_ravel = features.ravel()
    features_count = len(features)
    # print("Features.ravel: ",features_ravel)
    # print("Unique features: ",features_unique)
    # print("Features: ",len(features))
    return features_count, features_unique, features_ravel, LbpFoto
 
def shape_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = blur_image(image)
    # setting threshold of gray image
    _, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    im, contours, hierarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    i = 0
    # list for storing names of shapes
    for contour in contours:
        # here we are ignoring first counter because 
        # findcontour function detects whole image as shape
        if i == 0:
            i = 1
            continue
        # cv2.approxPloyDP() function to approximate the shape
        approx = cv2.approxPolyDP(
            contour, 0.01 * cv2.arcLength(contour, True), True)
 
        # using drawContours() function
        cv2.drawContours(image, [contour], 0, (0, 0, 255), 5)
 
    try:
        return image, len(approx)
    except:
        return image, 0
 
def knn(trainData, trainResponse, newcomer):
    neighbour_count = 1
    # Create KNN
    knn = cv2.ml.KNearest_create()
    # Train KNN with trainData
    knn.train(trainData, cv2.ml.ROW_SAMPLE, trainResponse)
    # Validate KNN with newcomer
    ret, results, neighbours ,dist = knn.findNearest(newcomer, neighbour_count)
    print( "result:  {}\n".format(results) )
    if neighbour_count > 1:
        print( "neighbours:  {}\n".format(neighbours) )
    print( "distance:  {}\n".format(dist) )
    return results
 
def ProcessImages(ImageArray, DataArray, debugBool):
    if debugBool == True:
        DebugDirectionary = '1train'
    else:
        DebugDirectionary = '2test'    
    for count, image  in enumerate(ImageArray):
        # Resize image to global SCALE variable
        ResizedImage = resize_image(image)
        #cv2.imshow("Resized image", ResizedImage)
 
        # Normalize the image
        NormalizedImage = normalize_image(ResizedImage)
        #cv2.imshow("Normalized image", NormalizedImage)
 
        # Medianblur the image
        blurred_image = blur_image(NormalizedImage) # image
        #cv2.imshow("Blurred image", blurred_image)
 
        # Bilateral filter on the image
        bf_image = bilateral_filter(blurred_image) # NormalizedImage
        # cv2.imshow("Bilateral filtered image", bf_image)
 
        # Background filter testing
        bg_filtered_image = background_filter(bf_image)
        #cv2.imshow("Background filtered image", bg_filtered_image)
 
        # Find the largest contour in the image
        contour = find_largest_contour(bg_filtered_image)
        
        #scale the contour to the original format
        contour, x, y, w, h = scale_contour(contour, SCALE) 
 
        # Filtered cropped image based on contour
        cropped = crop_flower(image, x, y, w, h, contour)
        # cv2.imshow("Non-filtered cropped image", cropped)
 
        #calculate avg color of picture within contours
        avg_h, avg_s, avg_v = avg_color(cropped)
 
        CroppedToCountLeaves = cropped.copy()
        CroppedToShapeDetection = cropped.copy()
        #gain texture features using lbp
        features_count, features_unique, features_ravel, lbpIm = lbp(cropped)
 
        CroppedToShapeDetection, shapeData = shape_detection(CroppedToShapeDetection)
 
        counted_leaves, ContourInCountLeaves = count_leaves(CroppedToCountLeaves)
 
        # for debugging purposes
        cv2.imwrite(f"{directory}/Dump/{DebugDirectionary}{count+1}Original.jpg", image)
        cv2.imwrite(f"{directory}/Dump/{DebugDirectionary}{count+1}ResizedImage.jpg", ResizedImage)
        cv2.imwrite(f"{directory}/Dump/{DebugDirectionary}{count+1}NormalizedImage.jpg", NormalizedImage)
        cv2.imwrite(f"{directory}/Dump/{DebugDirectionary}{count+1}blurred_image.jpg", blurred_image)
        cv2.imwrite(f"{directory}/Dump/{DebugDirectionary}{count+1}bf_image.jpg", bf_image)
        cv2.imwrite(f"{directory}/Dump/{DebugDirectionary}{count+1}bg_filtered_image.jpg", bg_filtered_image)
        cv2.imwrite(f"{directory}/Dump/{DebugDirectionary}{count+1}Cropped.jpg", cropped)
        cv2.imwrite(f"{directory}/Dump/{DebugDirectionary}{count+1}lbpIm.jpg", lbpIm)
        cv2.imwrite(f"{directory}/Dump/{DebugDirectionary}{count+1}CroppedToShapeDetection.jpg", CroppedToShapeDetection)
 
        # create DataArray in 5 x {len(ImageArray)} matrix
        for i in range(PARAMETER_COUNT):
            DataArray[count][0] = avg_h
            DataArray[count][1] = avg_s
            DataArray[count][2] = avg_v
            DataArray[count][3] = features_count
            DataArray[count][4] = features_unique
            DataArray[count][5] = counted_leaves
            DataArray[count][6] = shapeData
 
 
        # time tracker    
        print((count+1),"/",len(ImageArray))
    return DataArray

# Calcute the percentage of correct detected images from KNN 
def calc_correct_percentage(results):
    Guessed_correct = 0
    for counter, result in enumerate(results):
        if result == TestLabel[counter]:
            Guessed_correct += 1
 
    percentage_correct = (Guessed_correct / len(results))*100
    percentage_correct = round(percentage_correct,2)
    return percentage_correct
 
 
# MAIN
#####################################################
# Read all images from the folders and create a labelarray that defines each image
print("Loading all images")
 
TrainImages, TrainLabel, TrainArray, TestImages, TestLabel, TestArray = load_all_images()

#print("TrainLabel = ", TrainLabel)
#print("TestLabel = ", TestLabel)
 
print("processing train images")
TrainArray = ProcessImages(TrainImages, TrainArray, True)
TrainArray = np.array(TrainArray, dtype = np.float32)
TrainArray = np.round_(TrainArray, decimals = 0, out = None)
 
 
print("processing validation images")
TestArray = ProcessImages(TestImages, TestArray, False)
TestArray = np.array(TestArray, dtype = np.float32)
TestArray = np.round_(TestArray, decimals = 0, out = None)
 
print("Training data for KNN = \n",TrainArray)
print("Testing data for KNN = \n",TestArray)
 
print("Start KNN")
#knn met input labels en printarray 
results = knn(TrainArray, TrainLabel, TestArray)
 
percentage_correct = calc_correct_percentage(results)
print("percentage goed: ",percentage_correct)
#####################################################
     
cv2.waitKey(0); 
cv2.destroyAllWindows()
