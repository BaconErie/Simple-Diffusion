'''
HELPER.PY

Contains helper functions

- Turning image into a 0-1 matrix
'''

import cv2 as cv
import numpy as np

'''
NORMALIZER

Turns an image matrix (values between 0-255, inclusive) to a matrix with values
-1 to 1 inclusive.
'''
def normalize(image):
    return image / 127.5 - 1


'''
TO_DISPLAY

Essentially the reverse of normalize. Turns a matrix with values -1 to 1 into 
values between 0-255, inclusive. The output of this function can then be
directly displayed on the screen.
'''
def to_display(image):
    return (image * 127.5 + 127.5).astype(np.uint8)


'''
DISPLAY

Wrapper function that quickly displays an 0-255 image
'''
def display(image):
    cv.imshow("Display result", image)
    cv.waitKey(0)
    cv.destroyAllWindows()


'''
COUNT_BRIGHTNESS

Given a 0-255 image, count how many pixels have value of 0, how many 1... etc.
'''
def count_brightness(image):
    count = [0 for _ in range(0, 256)]

    for row in image:
        for pixel in row:
            count[pixel] += 1
        
    return count

# Same thing, but writes to CSV file
def count_brightness_and_output(image, filename):
    count = [0 for _ in range(0, 256)]

    for row in image:
        for pixel in row:
            count[pixel] += 1
        
    f = open(filename + '.csv', 'w')
    for x in range(0, 256):
        f.write(str(x) + ',')
        f.write(str(count[x]) + ',')
        f.write('\n')
    f.close()