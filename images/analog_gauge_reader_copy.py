'''  
Copyright (c) 2017 Intel Corporation.
Licensed under the MIT license. See LICENSE file in the project root for full license information.
'''

import cv2
import numpy as np
import math
# import paho.mqtt.client as mqtt
import time


def avg_circles(circles, b):
    avg_x = 0
    avg_y = 0
    avg_r = 0
    for i in range(b):
        # optional - average for multiple circles (can happen when a gauge is at a slight angle)
        avg_x = avg_x + circles[0][i][0]
        avg_y = avg_y + circles[0][i][1]
        avg_r = avg_r + circles[0][i][2]
    avg_x = int(avg_x / (b))
    avg_y = int(avg_y / (b))
    avg_r = int(avg_r / (b))
    return avg_x, avg_y, avg_r


def dist_2_pts(x1, y1, x2, y2):
    # print np.sqrt((x2-x1)^2+(y2-y1)^2)
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def calibrate_gauge(file_name, file_type):
    '''
        This function should be run using a test image in order to calibrate the range available to the dial as well as the
        units.  It works by first finding the center point and radius of the gauge.  Then it draws lines at hard coded intervals
        (separation) in degrees.  It then prompts the user to enter position in degrees of the lowest possible value of the gauge,
        as well as the starting value (which is probably zero in most cases but it won't assume that).  It will then ask for the
        position in degrees of the largest possible value of the gauge. Finally, it will ask for the units.  This assumes that
        the gauge is linear (as most probably are).
        It will return the min value with angle in degrees (as a tuple), the max value with angle in degrees (as a tuple),
        and the units (as a string).
    '''

    img = cv2.imread('%s.%s' % (file_name, file_type))
    height, width = img.shape[:2]

    #b_channel, g_channel, r_channel = cv2.split(img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert to gray
    # gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gray = cv2.medianBlur(gray, 5)

    # th, red_thresh = cv2.threshold(r_channel, 170, 255, cv2.THRESH_BINARY_INV);


    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, np.array([]), 100, 50, minRadius=int(height * 0.25),
                               maxRadius=int(height * 0.5))
    # average found circles, found it to be more accurate than trying to tune HoughCircles parameters to get just the right one
    a, b, c = circles.shape
    x, y, r = avg_circles(circles, b)

    # draw center and circle
    cv2.circle(img, (x, y), r, (255, 0, 255), 3, cv2.LINE_AA)  # draw circle
    cv2.circle(img, (x, y), 2, (0, 255, 0), 3, cv2.LINE_AA)  # draw center of circle

    # for testing, output circles on image
    # cv2.imwrite('gauge-%s-circles.%s' % (gauge_number, file_type), img)

    # for calibration, plot lines from center going out at every 10 degrees and add marker
    # for i from 0 to 36 (every 10 deg)

    '''
    goes through the motion of a circle and sets x and y values based on the set separation spacing.  Also adds text to each
    line.  These lines and text labels serve as the reference point for the user to enter
    NOTE: by default this approach sets 0/360 to be the +x axis (if the image has a cartesian grid in the middle), the addition
    (i+9) in the text offset rotates the labels by 90 degrees so 0/360 is at the bottom (-y in cartesian).  So this assumes the
    gauge is aligned in the image, but it can be adjusted by changing the value of 9 to something else.
    '''
    separation = 10.0  # in degrees
    interval = int(360 / separation)
    p1 = np.zeros((interval, 2))  # set empty arrays
    p2 = np.zeros((interval, 2))
    p_text = np.zeros((interval, 2))
    for i in range(0, interval):
        for j in range(0, 2):
            if (j % 2 == 0):
                p1[i][j] = x + 0.9 * r * np.cos(separation * i * 3.14 / 180)  # point for lines
            else:
                p1[i][j] = y + 0.9 * r * np.sin(separation * i * 3.14 / 180)
    text_offset_x = 10
    text_offset_y = 5
    for i in range(0, interval):
        for j in range(0, 2):
            if (j % 2 == 0):
                p2[i][j] = x + r * np.cos(separation * i * 3.14 / 180)
                p_text[i][j] = x - text_offset_x + 1.2 * r * np.cos(
                    (separation) * (i + 9) * 3.14 / 180)  # point for text labels, i+9 rotates the labels by 90 degrees
            else:
                p2[i][j] = y + r * np.sin(separation * i * 3.14 / 180)
                p_text[i][j] = y + text_offset_y + 1.2 * r * np.sin(
                    (separation) * (i + 9) * 3.14 / 180)  # point for text labels, i+9 rotates the labels by 90 degrees

    # add the lines and labels to the image
    for i in range(0, interval):
        cv2.line(img, (int(p1[i][0]), int(p1[i][1])), (int(p2[i][0]), int(p2[i][1])), (0, 255, 0), 2)
        cv2.putText(img, '%s' % (int(i * separation)), (int(p_text[i][0]), int(p_text[i][1])), cv2.FONT_HERSHEY_SIMPLEX,
                    0.3, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.imwrite('%s-calibration.%s' % (file_name, file_type), img)

    # get user input on min, max, values, and units
    print('File: %s.%s' % (file_name, file_type))
    # min_angle = input('Min angle (lowest possible angle of dial) - in degrees: ') #the lowest possible angle
    # max_angle = input('Max angle (highest possible angle) - in degrees: ') #highest possible angle
    # min_value = input('Min value: ') #usually zero
    # max_value = input('Max value: ') #maximum reading of the gauge
    # units = input('Enter units: ')

    # for testing purposes: hardcode and comment out raw_inputs above
    min_angle = 44
    max_angle = 315
    min_value = 0
    max_value = 16
    units = "ka"

    return min_angle, max_angle, min_value, max_value, units, x, y, r


def get_current_value(img, min_angle, max_angle, min_value, max_value, x, y, r, file_name, file_type):
    # for testing purposes
    # img = cv2.imread('gauge-%s.%s' % (gauge_number, file_type))

    gray2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Set threshold and maxValue
    thresh = 60

    center = (x, y)
    radius = r

    # apply thresholding which helps for finding lines
    th, dst2 = cv2.threshold(gray2, thresh, 255, cv2.THRESH_BINARY_INV);

    # found Hough Lines generally performs better without Canny / blurring, though there were a couple exceptions where it would only work with Canny / blurring
    # dst2 = cv2.medianBlur(dst2, 5)
    # dst2 = cv2.Canny(dst2, 50, 150)
    # dst2 = cv2.GaussianBlur(dst2, (5, 5), 0)

    # for testing, show image after thresholding
    cv2.imwrite('%s-tempdst2.%s' % (file_name, file_type), dst2)

    # Calculating pixels for every degree rotation
    image = dst2

    # Initializing variables
    max_pixels = 0
    needle_angle = 0

    correct_end_point = (0, 0)


    for angle in range(int(min_angle), int(max_angle)):
        angle_radians = math.radians(angle)
        start_point = center
        end_point = find_point_on_circle(center_x=x, center_y=y, r=radius, angle_in_degrees=angle)
        line_pixels = cv2.line(np.zeros(image.shape), start_point, end_point, 1, 1)
        no_of_pixels = len(np.nonzero(cv2.bitwise_and(image, image, mask=np.uint8(line_pixels)))[0])

        # Update when we get maximum white pixels on line
        if no_of_pixels > max_pixels:
            max_pixels = no_of_pixels
            needle_angle = angle
            correct_end_point = end_point

    cv2.line(img, center, correct_end_point, (0, 255, 0), 2)

    # for testing purposes, show the line overlayed on the original image
    # cv2.imwrite('gauge-1-test.jpg', img)
    cv2.imwrite('%s-lines-2.%s' % (file_name, file_type), img)

    angle_range = max_angle - min_angle
    start_to_needle_range = needle_angle - min_angle
    value_range = max_value - min_value

    measured_value = start_to_needle_range / angle_range * value_range + min_value

    return measured_value

def find_point_on_circle(center_x, center_y, r, angle_in_degrees):

    angle_in_degrees *= -1

    # convert angle from degrees to radians and adjust the starting point
    angle_in_radians = math.radians(90-angle_in_degrees)

    x = int(r * math.cos(angle_in_radians) + center_x)
    y = int(r * math.sin(angle_in_radians) + center_y)

    return x, y

def main():
    file_name = 'gauge-3'
    file_type = 'png'
    # name the calibration image of your gauge 'gauge-#.jpg', for example 'gauge-5.jpg'.  It's written this way so you can easily try multiple images
    min_angle, max_angle, min_value, max_value, units, x, y, r = calibrate_gauge(file_name, file_type)
    # feed an image (or frame) to get the current value, based on the calibration, by default uses same image as calibration
    img = cv2.imread('%s.%s' % (file_name, file_type))
    val = get_current_value(img, min_angle, max_angle, min_value, max_value, x, y, r, file_name, file_type)
    print("Current reading: %s %s" % (val, units))


if __name__ == '__main__':
    main()
