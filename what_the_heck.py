import math

timesteps = 1000

def alpha_bar_scheduler(t):
    return math.cos((t / timesteps + 0.008) / 1.008 * math.pi / 2) ** 2

def what():
    beta = [min(1 - alpha_bar_scheduler(t + 1) / alpha_bar_scheduler(t), 0.999) * 100 for t in range(timesteps) ]  
    return beta       

