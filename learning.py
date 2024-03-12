from neuron import LIF_simple
from parameters import num_in_neu,num_out_neu, range_stdp,epochs
from encoding import Poisson_generator, read_img,reconst_weights,spikes_graph
from STDP import update
from some_func import *
import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import random
import pandas as pd

############ Random Weights #####################

W = np.random.uniform(0.1,0.3,size = [num_in_neu,num_out_neu])


T = 6
dt = 0.1
time = np.arange(0,T+dt,dt)

"""
Функция обучения
На вход принимает:
 -путь до тренировочного датасета
 -время обучения
 -количество входных нейронов
 -количество выходных нейронов
 -массив весов
 -номер эпохи (нужно исключительно для датафрейма)

Возвращает:
 -массив весов всех нейронов
"""


def learn(path, time, num_in_neu, num_out_neu, W, epoch_number):

	"Создаем DataFramе"
	df = create_dataframe()
	df_path = '/Users/tortolla/Desktop/proga/SNN_movement/neuro_spike/movment_recognition/df'

	"Массивы входных и выходныхь спайков"
	in_spikes = np.empty(shape=(num_in_neu, len(time)))
	out_spikes = np.zeros(shape=(num_out_neu, len(time)))

	"Список для выходных нейронов"
	out_neurons = []

	"Список путей до изображений тренировочного датасета"
	list = take_data_from_folder(path)
	list = shuffle_list(list)

	"Цикл, который проходится по всем изображением датасета"
	for o in list:

		"Создаем соответствующие массивы спайков - выходных/входных "
		in_spikes = np.zeros(shape=(num_in_neu, len(time)))
		out_spikes = np.zeros(shape=(num_out_neu, len(time)))
		
		"Список выходных нейронов"
		out_neurons = []

		"Создаем выходные нейроны"
		for i in range(num_out_neu):

			a = LIF_simple()
			out_neurons.append(a)


		"Считываем изображение"
		img = read_img(o)
		
		"Генерируем входные спайки в зависимости от изображения"
		for l in range(num_in_neu):

			in_spikes[l] = Poisson_generator(T, dt, 0 + img[l], 1)
		
		"Массив внутренних токов выходных нейронов"
		I = np.zeros(shape=(num_out_neu,))

		"Запускаем цикл для каждого дискретного момента времени"
		for t in range(len(time)):

			"Запускаем цикл по выходным нейронам"
			for j,neu in enumerate(out_neurons):
				
				"Изначально токи в нейронах равны нулю"
				I[j] = 0

				"Запускаем цикл по входным нейронам"
				for i in range(num_in_neu):
					# I[j] = 0
					
					"Увеличиваем токи в выходных нейронах в соответствии с весом"
					I[j] += np.dot(W[i][j],in_spikes[i][t])
				
				"Условие на рефракторный период"
				if t>=neu.initRefrac:
						
						"Считаем потенциал нейрона"
						v = neu.vprev + (-neu.vprev + I[j] * neu.R) / neu.tau_m * dt  # LIF

						# v= neu.vprev + np.dot(W[i][j],in_spikes[i][t])
						"Реализуем утечку"
						if (v > neu.v_base):
							v -= 0.012
							if v < neu.v_base:
								v = neu.v_base
								pass
						
						"В случае превышения порога выходной нейрон спайкует"
						if v >= neu.v_thresh:

							"Нейрон спайкует"
							neu.num += 1
							out_spikes[j][t] = 1
							
							"Запускаем рефракторный период"
							neu.initRefrac = t + neu.refracTime

							"Реализуем адаптивность порога нейрона"
							neu.v_thresh += 0.001

				"Обновляем потенциал нейрона"			
				neu.vprev = v
			
			"Далее добавляем конкуренцию в выходные нейроны"


			"Переменная нейрона с максимальным потенциалом"
			max_index = 0

			"Создаем переменную максимального значения потенциала"
			max_value = out_neurons[0].vprev
			
			"Запускаем цикл по выходным нейронам, находя в нем нейрон с максимальным потенциалом"
			for i in range(0,len(out_neurons)):
				if out_neurons[i].vprev > max_value:
					max_index = i
					max_value = out_neurons[i].vprev
			
			"Запускаем цикл по выходным нейронам, в котором оставляем выходной спайк только у нейрона с максимальным потенциалом"
			for j, neu in enumerate(out_neurons):

				neu.vprev = 0
				if j!= max_index:
					out_spikes[j][t] = 0
					pass
			
			"Запускаем цикл по выходным нейронам"
			for j,neu in enumerate(out_neurons):

				"Реализуем механизм STDP"
				for i in range(num_in_neu):
					for t1 in range(-1, -range_stdp, -1):
						if 0 <= t + t1 < len(time):
							if in_spikes[i][t + t1] == 1 and out_spikes[j][t] == 1:
								W[i][j] = update(W[i][j], t1)
					for t1 in range(1, range_stdp, 1):
						if 0 <= t + t1 < len(time):
							if in_spikes[i][t + t1] == 1 and out_spikes[j][t] == 1:
								W[i][j] = update(W[i][j], t1)
		

		"Находим нейрон с наибольшим количеством спайков"
		max_spikes_number, were_there_spikes = neuron_with_max_spikes(out_spikes)

		"Считываем правильную информацию о направлении движения"
		true_label = o[-5]
		
		"Добавляем соответствующую запись в датафрейм"
		df = add_row_to_dataframe(df, true_label, max_spikes_number, were_there_spikes)

		"Сохраняем датафрейм"
		save_dataframe(df, df_path+'/'+str(epoch_number))


	return W



