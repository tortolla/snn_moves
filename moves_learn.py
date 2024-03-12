from neuron import LIF_simple
from parameters import num_in_neu,num_out_neu, range_stdp,epochs
from encoding import Poisson_generator, read_img,reconst_weights,spikes_graph
from STDP import update
from some_func import *
from learning import *
import numpy as np
import matplotlib.pyplot as plt
import csv
import os  
import random
import pandas as pd
import time as t


"Начинаем считать время работы кода"

start_time = t.time()

" Вводим путь до датасета"
path = '/Users/tortolla/Desktop/proga/SNN_movement/neuro_spike/movment_recognition/dataset(1)/train1'


"Задаем начальные случайные веса"

############ Random Weights #####################
W = np.random.uniform(0.1,0.3,size = [num_in_neu,num_out_neu])
#################################


# T = 6
# dt = 0.1
# time = np.arange(0,T+dt,dt)

"Задаем количество эпох"

epochs = 3

"Запускаем цикл по всему количеству эпох"
for i in range(epochs):

    print(i)

    "Обновляем веса для каждой эпохи"
    W = learn(path, time, num_in_neu, num_out_neu, W)


"Считываем время конца работы программы"
end_time = t.time()

"Записываем время работы программы в файл"
elapsed_time = end_time - start_time
write_execution_time_to_file('/Users/tortolla/Desktop/proga/SNN_movement/neuro_spike/movment_recognition', 'code_time', elapsed_time, T, epochs, W)
