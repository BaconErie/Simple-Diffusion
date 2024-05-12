'''
DIFFUSION.PY

This contains all the functions required for the noising part of the diffusion model

'''

import cv2 as cv
import numpy as np
from numpy import sqrt, mean, var
from helper import normalize, to_display, display, count_brightness
from math import cos, pi

MAX_TIMESTEPS = 100


'''
ALPHA BAR

In the noise process equation, alpha essentially determines how much to noise
the image at a particular timestep, t. Lower alpha bar is more noise/less original image

You can modify how alpha bar changes over time to change how quickly the image
gets noisy over time; this is called the **noise schedule**. 

In the original diffusion model paper, the researchers used a simple linear
noise schedule. However, the schedule quickly made the image noise up
even before we reached max timesteps.

Instead, of a linear schedule, this code uses a cosine schedule based on
Nichol and Dhariwal's work at OpenAI (https://arxiv.org/abs/2102.09672)

You can see tests between linear vs cosine schedule at 
https://hackmd.io/@BaconErie/lin-vs-cos-schedule (might be a bit complicated)
'''
def alpha_bar(t):
    f_t = cos( ((t/MAX_TIMESTEPS + 0.008)/(1+0.008))  * (pi/2) ) ** 2
    f_0 = cos( ((0.008)/(1+0.008))  * (pi/2) ) ** 2

    return f_t/f_0


'''
EPSILON

This is the noise that we apply to the image. It is a matrix of values, where
each number is a random number. The probability of getting each number is
based on a normal distribution; Values closer to zero have a higher chance of 
getting chosen.

The matrix size is y by x, as defined in the function parameters.
'''
def epsilon(y, x):
    sample_matrix = np.random.normal(loc=0, scale=0.3, size=(y, x)).reshape(y, -1)
    
    return sample_matrix



'''
NOISE FUNCTION

This is the star of the show, and implements the main noising equation.

Given a matrix (that represents an image) and timestep t, it will return a
noised image that can then be used to train the denoiser.
'''
def noise(image, t):
    num_rows = len(image)
    num_cols = len(image[0])

    sample = epsilon(num_rows, num_cols)
    result = sqrt(alpha_bar(t)) * image + sqrt(1 - alpha_bar(t)) * sample

    return np.clip(result, -1, 1)


'''
NOISE RUNNER

This function makes it easy for users to quickly test the noise function.

It asks the user for image input, and then the timestep t, and displays
the result.
'''
def noise_runner():
    image_name = input('What is the name of the image file? (Include file extension): ')
    timestep = int(input('What timestep to noise to?: '))

    image = cv.imread(image_name, cv.IMREAD_GRAYSCALE)
    normalized_image = normalize(image)
    noised_image = noise(normalized_image, timestep)

    noised_display_image = to_display(noised_image)

    display(noised_display_image)

# If this file is being run directly, start the noise_runner
if __name__ == '__main__':
    noise_runner()