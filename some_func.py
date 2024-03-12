import os 
import random
import pandas as pd
import numpy as np


"Функция находит номер спайка с максимальным количеством импульсов, таким образом функция считывает предсказние SNN"

def neuron_with_max_spikes(neuron_data):


    spikes_sum = np.sum(neuron_data, axis=1)

    all_spikes = sum(spikes_sum)

    "Проверяем были ли вообще спайки"
    if all_spikes == 0:
        
        were_there_spikes = 0
    
    else:

        were_there_spikes = 1

    neuron_id_with_max_spikes = np.argmax(spikes_sum)


    return neuron_id_with_max_spikes, were_there_spikes


"Функция убирает файл .DS_Store из папки - нужно только для macOS"

def remove_ds_store(directory):
   
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file == '.DS_Store':
                os.remove(os.path.join(root, file))
                print(f"Removed .DS_Store from {root}")


"Функция считывает все пути до изображений из папки"

def take_data_from_folder(directory):

    remove_ds_store(directory)

    files_list = []  
    files = os.listdir(directory)

    for file in files:
        full_path = os.path.join(directory, file)  
        files_list.append(full_path)

    return files_list

"Функция перемещивающая список, который подают на вход"

def shuffle_list(input_list):

    random.shuffle(input_list)

    return input_list


"Функция, создающая DataFrame"

def create_dataframe():

    df = pd.DataFrame(columns=['label', 'neuron number', 'were_there_spikes'])
    return df

"Функция сохраняет DataFrame в csv файл"

def save_dataframe(df, path):
 
    df.to_csv(path, index=False)

"Функция добавляет новую строчку в DataFrame"

def add_row_to_dataframe(df, label, neuron_number, were_there_spikes):

    new_row = {'label': label, 'neuron number': neuron_number, 'were_there_spikes': were_there_spikes}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    return df


"Функция записи в файл время выполнение кода"

def write_execution_time_to_file(directory_path, file_name, execution_time, T, epochs, W):
    
    full_path = os.path.join(directory_path, file_name)
    
    with open(full_path, 'w') as file:
        file.write(f"Время выполнения кода: {execution_time} секунд.")
        file.write(f"Epochs_number:{T}, Epochs_time:{epochs}")
        file.write(f"\n W_array \n f{W}")

