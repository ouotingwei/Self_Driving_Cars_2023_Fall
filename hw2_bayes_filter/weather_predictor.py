"""
@author: OU,TING-WEI @ M.S. in Robotics 
date : 10-15-2023
Self-Driving-Cars HW2 ( NYCU FALL-2023 )
"""
import random
import numpy as np

transition_probabilities =[[0.8, 0.4, 0.2], [0.2, 0.4, 0.6], [0, 0.2, 0.2]]

def random_genenerator(init='Sunny', days=10):
    transition_probabilities = {
        ('Sunny', 'Sunny'): 0.8,
        ('Sunny', 'Cloudy'): 0.2,
        ('Sunny', 'Rainy'): 0,
        ('Cloudy', 'Sunny'): 0.4,
        ('Cloudy', 'Cloudy'): 0.4,
        ('Cloudy', 'Rainy'): 0.2,
        ('Rainy', 'Sunny'): 0.2,
        ('Rainy', 'Cloudy'): 0.6,
        ('Rainy', 'Rainy'): 0.2,
    }

    sequence = [init]

    for _ in range(int(days) - 1):
        current_weather = sequence[-1]
        next_weather = random.choices(['Sunny', 'Cloudy', 'Rainy'], 
                                    [transition_probabilities[(current_weather, 'Sunny')],
                                    transition_probabilities[(current_weather, 'Cloudy')],
                                    transition_probabilities[(current_weather, 'Rainy')]])[0]
        sequence.append(next_weather)

    print("the randomly generate sequence of weather ：")
    for day, weather in enumerate(sequence, start=1):
        if weather == 'Sunny':
            tomorrow = [0.8, 0.2, 0]
        if weather == 'Rainy':
            tomorrow = [0.2, 0.6, 0.2]
        if weather == 'Cloudy':
            tomorrow = [0.4, 0.4, 0.2]
        print(f"day: {day} ：{weather}", ', tomorrow will be : ', tomorrow)

def stationary_distribution(init='Sunny', days=10):
    if init == "Sunny" :
        state = np.array([[1, 0, 0]]).T
    if init == "Cloudy" :
        state = np.array([[0, 1, 0]]).T
    if init == "Rainy" :
        state = np.array([[0, 0, 1]]).T

    for i in range(int(days)):
        state = transition_probabilities @ state
        print("the probability of tomorrow's weather is : ", state.T)

def main():
    init_weather = init_weather = input("today's weather = ")
    days = input("How many days do you want to predict? ")
    
    random_genenerator(init_weather, days)

    print('----------------------------------------------------------------')

    stationary_distribution(init_weather, days)

if __name__ == '__main__':
    main()