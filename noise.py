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
BETA: THE NOISE SCHEDULE

In the noise process equation, beta essentially determines how much to noise
the image at a particular timestep, t. Higher beta is more noise.

For example, what you can do is to noise it slowly at first, and then increase
it later on.

Beta is finally used in the calculation of alpha and alpha bar, which is directly
used in the equation to change how much the image gets noised.
'''
def beta(t):
    return t/MAX_TIMESTEPS


'''
ALPHA BAR

Variable directly used in the equation to change how much the image is noised.

Alpha bar is calculated through repeated multiplication of alpha without bar,
which is just alpha in the function. 
'''
def alpha_bar(t):
    return alpha_bar_linear(t)

    f_t = cos( ((t/MAX_TIMESTEPS + 0.008)/(1+0.008))  * (pi/2) ) ** 2
    f_0 = cos( ((0.008)/(1+0.008))  * (pi/2) ) ** 2

    return f_t/f_0

def alpha_bar_linear(t):
    product = 1

    for x in range (1, t):
        alpha = 1 - beta(x)

        product *= alpha

    return product


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

# Prints mean and variance of pixel values in the image. Pixel values range from -1 to 1
def variance_and_mean_test():
    image_name = 'number.jpg'

    print('timestep,mean,variance')

    for timestep in range (0, 101):
        image = cv.imread(image_name, cv.IMREAD_GRAYSCALE)
        normalized_image = normalize(image)
        noised_image = noise(normalized_image, timestep)
        print(f'{timestep},{mean(noised_image)},{var(noised_image)}')

def schedule_test():
    for timestep in range (0, MAX_TIMESTEPS+1, int(MAX_TIMESTEPS/10)):
        print('what)')
        image = cv.imread('seven.png', cv.IMREAD_GRAYSCALE)

        normalized_image = normalize(image)
        noised_image = noise(normalized_image, timestep)

        noised_display_image = to_display(noised_image)

        cv.imwrite('linear-'+str(timestep/MAX_TIMESTEPS)+'.png', noised_display_image)

def get_alpha_beta():
    print('Timesteps (t/T),Linear,Cosine')
    for timestep in range (0, MAX_TIMESTEPS+1):
        print(f'{timestep/MAX_TIMESTEPS},{alpha_bar_linear(timestep)},{alpha_bar(timestep)}')


# If this file is being run directly, start the noise_runner
if __name__ == '__main__':
    noise_runner()