from PIL import Image
import cv2
import numpy as np
import random
import os
import numpy as np
from urllib.request import urlopen
import urllib.request as ulib
from bs4 import BeautifulSoup as Soup
import ast
from selenium import webdriver
import requests
from io import BytesIO
import time
import urllib3
import re

#Path to chrome driver
chromePath=r'/Users/blackhole/Desktop/ChromeDrivers/chromedriver_87'
driver = webdriver.Chrome(chromePath)
#URL path of background
URL='https://www.google.com/search?q=plain+streets&sxsrf=ALeKk03nWfX7Y04KgK5RX8WZzcsNtoL1tw:1606495152718&source=lnms&tbm=isch&sa=X&ved=2ahUKEwj806TnlKPtAhUOOisKHX-sAuYQ_AUoAXoECAYQAw'
#Directory to Store images
directory = '/Users/blackhole/Desktop/Drone Images'


#Rotates the cv_image through an angle 'angle'
def rotate(angle,cv_img):
    img=cv_img
    rows = img.shape[0]
    cols = img.shape[1]
    img_center = (cols / 2, rows / 2)
    M = cv2.getRotationMatrix2D(img_center, angle, 1)
    rotated_image = cv2.warpAffine(img, M, (cols, rows),borderValue=(255,255,255))
    image = Image.fromarray(rotated_image.astype('uint8'), 'RGB')
    return image

#Creating a white image of same size as that of final image so that the white image containing the object can be superimposed on the background
def paste(PIL_white,angle,cv_img,size=(50,50),area=(100,100,150,150)):
    PIL_image=rotate(angle,cv_img)
    PIL_image = PIL_image.resize(size)
    PIL_white = PIL_white.resize((416,416))
    PIL_white.paste(PIL_image,area)
    image = cv2.cvtColor(np.array(PIL_white), cv2.COLOR_RGB2BGR)
    return image

#Generates the text file given the following parameters
def gen_textfile(area,size,x_size=416,y_size=416,object_class=0):
    x_center=(area[0]+(size[0]/2))/x_size
    y_center=(area[1]+(size[1]/2))/y_size
    width=size[0]/x_size
    height=size[1]/y_size
    data=str(object_class)+' '+str(x_center)+' '+str(y_center)+' '+str(width)+' '+str(height)
    return data

#Superimposes object image and background image to a final image
def add_images(img1_back,img2_drone):
    background=cv2.resize(img1_back,(416,416))
    drone=cv2.resize(img2_drone,(416,416))
    drone_gray = cv2.cvtColor(drone, cv2.COLOR_BGR2GRAY)
    #edit this range for different white values
    ret, mask = cv2.threshold(drone_gray, 254, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    background = cv2.bitwise_and(background, background, mask=mask)
    drone = cv2.bitwise_and(drone, drone, mask=mask_inv)
    result = cv2.add(background, drone)
    result = cv2.add(background, drone)
    return result

#Gets the image src URL's of the URL provided
def getURLs(URL):
    driver.get(URL)
    a=input()
    page = driver.page_source
    images = driver.find_elements_by_tag_name('img')
    URLS=[]
    for image in images:
        URLS.append(image.get_attribute('src'))
    return URLS

#Saves images to directory
def save_images(URLs, directory):

    if not os.path.isdir(directory):
        os.mkdir(directory)

    for i, url in enumerate(URLs):
        #add 100 here to k for every iteration eg.->k=100+i
        #k=i+Tau is used because for YOLO we need images with unqiue paths, if they have the same path as 100.jpg , 100.jpg then it throws an error
        k=i+20603
        try:

            response = requests.get(url)
            img = Image.open(BytesIO(response.content))

            #THIS COMMENTED CODE IS USEFUL IF YOU HAVE DIFFERENT IMAGES OF THE SAME OBJECT
            # drone_number = random.randint(0, 14)
            # drone_path = str('/Users/blackhole/Downloads/manas')
            # drone_number = str(drone_number)
            # drone_path = drone_path + drone_number + '.jpg'

            drone_path = '/Users/blackhole/Downloads/drone0-removebg-preview.png'


            #Generating the white image as discussed above(paste())
            img_white = np.zeros([416, 416, 3], dtype=np.uint8)
            img_white.fill(255)
            white = Image.fromarray(img_white.astype('uint8'), 'RGB')

            #Size preprocessing
            img2 = cv2.imread(drone_path)
            print(img2.shape[0],img2.shape[1])
            img2=cv2.resize(img2,(416,416))

            #Removing the colors which are in this range
            hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
            black_low = np.array([0, 0, 0])
            black_high = np.array([0, 0, 0])
            mask = cv2.inRange(hsv, black_low, black_high)
            img2[mask > 0] = (255, 255, 255)

            #generating random sizes for the objects
            all_size = random.randint(75, 200)
            x_area = random.randint(15, 200)
            y_area = random.randint(15, 200)
            size = (all_size, all_size)
            area = (x_area, y_area, (x_area + all_size), (y_area + all_size))
            angle = random.randint(0, 360)
            #angle=0
            final_image = paste(PIL_white=white, angle=angle, cv_img=img2, size=size, area=area)
            #EDIT OBJECT CLASS
            labels = gen_textfile(area=area, size=size, x_size=416, y_size=416,object_class=0)
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            result = add_images(img, final_image)
            result = Image.fromarray(result.astype('uint8'), 'RGB')
            general_path = str(os.path.join(directory, '{:06}'.format(k)))
            image_path = general_path + '.jpg'
            text_path = general_path + '.txt'
            result.save(image_path)
            f = open(text_path, 'w+')
            f.write(labels)
            f.close()
            #if i % 10 == 0:
                # shower = Image.fromarray(result.astype('uint8'), 'RGB')
                #result.show()

        except:
            print('I failed with', url)
URLs = getURLs(URL)
print(URLs)
save_images(URLs, directory)
