"""
@author: OU,TING-WEI @ M.S. in Robotics 
date : 10-15-2023
Self-Driving-Cars HW2 ( NYCU FALL-2023 )
"""
import random
import numpy as np

transition_probabilities =[[0.8, 0.2, 0], [0.4, 0.4, 0.2], [0.2, 0.6, 0.2]]

def random_genenerator(init='Sunny', days=10):
    if init == "Sunny" or "sunny":
        state = [1, 0, 0]
    if init == "Cloudy" or "cloudy":
        state = [0, 1, 0]
    if init == "Rainy" or "rainy":
        state = [0, 0, 1]
    
    for i in range(int(days)):
        state = np.dot(transition_probabilities , state)
        print(state)
        

def main():
    init_weather = init_weather = input("today's weather = ")
    days = input("How many days do you want to predict? ")
    
    random_genenerator(init_weather, days)

if __name__ == '__main__':
    main()