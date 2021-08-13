from math import nan
import re
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.signal import savgol_filter
from scipy import signal
from prettytable import PrettyTable
from scipy.signal import gauss_spline, bspline
from datetime import datetime
import time
import statistics
import os
import glob
from scipy.optimize import curve_fit
from numpy import arange
#const
ADC_freq_Hz = 500000
ticks = 1/ADC_freq_Hz
velocity_sweep = 5000000000
#без нормировки уровень 0,4, фильтр 1300
trashold = 0.5
N_filt = 1
F_filt_ch1 = 900
F_filt_ch2 = 900
#download file mesurment
#read_file = pd.read_csv (r'C: \ Users \ Ron \ Desktop \ Test \ Product_List.txt ')
#read_file.to_csv (r'C: \ Users \ Ron \ Desktop \ Test \ New_Products.csv ', index = None)

#counter files CSV in work dir
myPath = 'E:\KDA\\files\\230721\\tektronix'
files = os.listdir(myPath)
csv_files = list(filter(lambda s: s.endswith('.csv'), files))
#print(csv_files)
csv_files_copy = csv_files.copy()
'''ang_velocities =[]
# создать массиа для точек угловой скоротси
for i in range(len(csv_files)):
    csv_files[i] = csv_files[i].replace('.csv','')
    ang_velocities.append(csv_files[i])
    #print(type(csv_files[i]))
print('массив угловых скоростей: ',ang_velocities)'''
mas_avg_freq = []
mas_ang_vel = []
ang_velocity = ''
mas_freq_noise = []
for root, dirs, files in os.walk('E:\KDA\\files\\230721\\tektronix'):
    for file in files:
        filename, extension = os.path.splitext(file)
# Часть цикла для положительного направления вращения
        if filename.find('Ch1') != -1 and filename.find('-') ==-1:
            print(filename)
            #print(filename.find('_degs_'))
            ang_velocity = int(filename[0:filename.find('_degs_')])
            print(ang_velocity)
            mas_ang_vel.append(ang_velocity)
            #mas_ang_vel.append()
            # download file mesurment
            x1 = pd.read_csv('E:\KDA\\files\\230721\\tektronix\\%s.csv' % filename, delimiter=',', names=['2', '1', 'ch1'],
                             skiprows=6, engine='python')
            #x2 = pd.read_csv('E:\KDA\\files\\230721\\tektronix\\%s.csv' % filename, delimiter=',', names=['2', '1', 'ch2'],
                             #skiprows=6, engine='python')
            #df_ch2 = pd.DataFrame(x2, columns=['ch2'])
            df_ch1 = pd.DataFrame(x1, columns=['ch1'])
            t = pd.DataFrame(x1, columns=['1'])

            # for ch1
            y1 = df_ch1.values.flatten()
            b, a = signal.butter(N_filt, F_filt_ch1, fs=(1 / ticks))
            z1 = signal.filtfilt(b, a, y1)
            # z1 = savgol_filter(y1, 2001, 5)
            zdf_ch1 = pd.DataFrame(data=z1, columns=['column1'])
            df_ch1_norm = (zdf_ch1 - zdf_ch1.min()) / (zdf_ch1.max() - zdf_ch1.min())
            zdf_ch1['min'] = df_ch1_norm.column1[(df_ch1_norm.column1.shift(1) > df_ch1_norm.column1) & (
                        df_ch1_norm.column1.shift(-1) > df_ch1_norm.column1)]
            sortdf_ch1 = zdf_ch1['min'].where(zdf_ch1['min'] < trashold)
            # Find only peaks
            indexdf = sortdf_ch1.notnull()
            # Find index peaks
            index_peaks_ch1 = zdf_ch1[indexdf].index
            peaks_time_ch1 = index_peaks_ch1 * ticks
            peaks_time_ch1_array = len(peaks_time_ch1.values)

            #print('Index peaks ch1', index_peaks_ch1)
            #print('Quant peaks ch1', peaks_time_ch1_array)
            #print('Time peaks ch1', peaks_time_ch1)
            #mas.append(peaks_time_ch1)
        else:
            if filename.find('Ch2') != -1 and filename.find('-') == -1:
                print(filename)
                #print(filename.find('_degs_'))
                ang_velocity = filename[0:filename.find('_degs_')]
                print(ang_velocity)
                #mas_ang_vel.append(ang_velocity)
                x2 = pd.read_csv('E:\KDA\\files\\230721\\tektronix\\%s.csv' % filename, delimiter=',',
                                 names=['2', '1', 'ch2'],
                                 skiprows=6, engine='python')
                df_ch2 = pd.DataFrame(x2, columns=['ch2'])
                # for ch2
                y2 = df_ch2.values.flatten()
                b, a = signal.butter(N_filt, F_filt_ch2, fs=(1 / ticks))
                z2 = signal.filtfilt(b, a, y2)
                # z2 = savgol_filter(y2, 2001, 5)
                zdf_ch2 = pd.DataFrame(data=z2, columns=['column1'])
                df_ch2_norm = (zdf_ch2 - zdf_ch2.min()) / (zdf_ch2.max() - zdf_ch2.min())
                zdf_ch2['min'] = df_ch2_norm.column1[(df_ch2_norm.column1.shift(1) > df_ch2_norm.column1) & (
                    df_ch2_norm.column1.shift(-1) > df_ch2_norm.column1)]
                sortdf_ch2 = zdf_ch2['min'].where(zdf_ch2['min'] < trashold)
                # Find only peaks
                indexdf = sortdf_ch2.notnull()
                # Find index peaks
                index_peaks_ch2 = zdf_ch2[indexdf].index
                peaks_time_ch2 = index_peaks_ch2 * ticks
                peaks_time_ch2_array = len(peaks_time_ch2.values)

                #print('Index peaks ch2', index_peaks_ch2)
                #print('Quant peaks ch2', peaks_time_ch2_array)
                #print('Time peaks ch2', peaks_time_ch2)
                dif = peaks_time_ch2 - peaks_time_ch1
                # print('Difference', dif1)
                #print('Разность времен: ', dif)
                freq = dif * velocity_sweep
                #print('Разность частот: ', freq)
                avg_freq = np.mean(freq)
                print("Сдвиг частот ср.: ", avg_freq)
                mas_avg_freq.append(avg_freq)
                noise_freq = np.std(freq)
                print("Шум частот : ", noise_freq)
                mas_freq_noise.append(noise_freq)
# Часть цикла для отрицательного направления вращения
for root, dirs, files in os.walk('E:\KDA\\files\\230721\\tektronix'):
    for file in files:
        filename, extension = os.path.splitext(file)
        if filename.find('Ch1') != -1 and filename.find('-') !=-1:
            print(filename)
            #print(filename.find('_degs_'))
            ang_velocity = int(filename[0:filename.find('_degs_')])
            print(ang_velocity)
            mas_ang_vel.append(ang_velocity)
            # download file mesurment
            x1 = pd.read_csv('E:\KDA\\files\\230721\\tektronix\\%s.csv' % filename, delimiter=',', names=['2', '1', 'ch1'],
                             skiprows=6, engine='python')
            #x2 = pd.read_csv('E:\KDA\\files\\230721\\tektronix\\%s.csv' % filename, delimiter=',', names=['2', '1', 'ch2'],
                             #skiprows=6, engine='python')
            #df_ch2 = pd.DataFrame(x2, columns=['ch2'])
            df_ch1 = pd.DataFrame(x1, columns=['ch1'])
            t = pd.DataFrame(x1, columns=['1'])

            # for ch1
            y1 = df_ch1.values.flatten()
            b, a = signal.butter(N_filt, F_filt_ch1, fs=(1 / ticks))
            z1 = signal.filtfilt(b, a, y1)
            # z1 = savgol_filter(y1, 2001, 5)
            zdf_ch1 = pd.DataFrame(data=z1, columns=['column1'])
            df_ch1_norm = (zdf_ch1 - zdf_ch1.min()) / (zdf_ch1.max() - zdf_ch1.min())
            zdf_ch1['min'] = df_ch1_norm.column1[(df_ch1_norm.column1.shift(1) > df_ch1_norm.column1) & (
                        df_ch1_norm.column1.shift(-1) > df_ch1_norm.column1)]
            sortdf_ch1 = zdf_ch1['min'].where(zdf_ch1['min'] < trashold)
            # Find only peaks
            indexdf = sortdf_ch1.notnull()
            # Find index peaks
            index_peaks_ch1 = zdf_ch1[indexdf].index
            peaks_time_ch1 = index_peaks_ch1 * ticks
            peaks_time_ch1_array = len(peaks_time_ch1.values)

            #print('Index peaks ch1', index_peaks_ch1)
            #print('Quant peaks ch1', peaks_time_ch1_array)
            #print('Time peaks ch1', peaks_time_ch1)
            #mas.append(peaks_time_ch1)
        else:
            if filename.find('Ch2') != -1 and filename.find('-') != -1:
                print(filename)
                #print(filename.find('_degs_'))
                ang_velocity = filename[0:filename.find('_degs_')]
                print(ang_velocity)
                #mas_ang_vel.append(ang_velocity)
                x2 = pd.read_csv('E:\KDA\\files\\230721\\tektronix\\%s.csv' % filename, delimiter=',',
                                 names=['2', '1', 'ch2'],
                                 skiprows=6, engine='python')
                df_ch2 = pd.DataFrame(x2, columns=['ch2'])
                # for ch2
                y2 = df_ch2.values.flatten()
                b, a = signal.butter(N_filt, F_filt_ch2, fs=(1 / ticks))
                z2 = signal.filtfilt(b, a, y2)
                # z2 = savgol_filter(y2, 2001, 5)
                zdf_ch2 = pd.DataFrame(data=z2, columns=['column1'])
                df_ch2_norm = (zdf_ch2 - zdf_ch2.min()) / (zdf_ch2.max() - zdf_ch2.min())
                zdf_ch2['min'] = df_ch2_norm.column1[(df_ch2_norm.column1.shift(1) > df_ch2_norm.column1) & (
                    df_ch2_norm.column1.shift(-1) > df_ch2_norm.column1)]
                sortdf_ch2 = zdf_ch2['min'].where(zdf_ch2['min'] < trashold)
                # Find only peaks
                indexdf = sortdf_ch2.notnull()
                # Find index peaks
                index_peaks_ch2 = zdf_ch2[indexdf].index
                peaks_time_ch2 = index_peaks_ch2 * ticks
                peaks_time_ch2_array = len(peaks_time_ch2.values)

                #print('Index peaks ch2', index_peaks_ch2)
                #print('Quant peaks ch2', peaks_time_ch2_array)
                #print('Time peaks ch2', peaks_time_ch2)
                dif = peaks_time_ch2 - peaks_time_ch1
                #print('Разность времен: ', dif)
                freq = dif * velocity_sweep
                #print('Разность частот: ', freq)
                avg_freq = np.mean(freq)
                print("Сдвиг частот ср.: ", avg_freq)
                mas_avg_freq.append(avg_freq)
                noise_freq = np.std(freq)
                print("Шум частот : ", noise_freq)
                mas_freq_noise.append(noise_freq)
print('Массив ср. сдвигов частот: ', mas_avg_freq)
#print('Массив угловых скоростей: ', mas_ang_vel)
#z = np.polyfit(mas_ang_vel, mas_avg_freq, 1)
print(len(mas_ang_vel))
print(len(mas_avg_freq))
# define the true objective function
def objective(x, a, b):
    return a * x + b
# choose the input and output variables
x, y = mas_ang_vel, mas_avg_freq
# curve fit
popt, _ = curve_fit(objective, x, y)
# summarize the parameter values
a, b = popt
print('y = %.5f * x + %.5f' % (a, b))
print('Масштабный коэффициент: ', a)
print('Смещение нуля: ', b)
ang_velocity_measured = []
#print(len(mas_avg_freq))
for i in range(len(mas_avg_freq)):
    rot_rate = (mas_avg_freq[i] / a) - (b/a)
    ang_velocity_measured.append(rot_rate)
print('Массив угловых скоростей: ', mas_ang_vel)
print("Массив измеренных угловых скоростей: ", ang_velocity_measured)
mas_rotation_noise =[]
for i in range(len(mas_freq_noise)):
    rotation_noise = mas_freq_noise[i]/a
    mas_rotation_noise.append(rotation_noise)
print("Шум угловой скорости: ", mas_rotation_noise)
# plot input vs output
plt.scatter(x, y)
# define a sequence of inputs between the smallest and largest known inputs
x_line = arange(min(x), max(x), 1)
# calculate the output for the range
y_line = objective(x_line, a, b)
# create a line plot for the mapping function
plt.plot(x_line, y_line, '--', color='red')
plt.show()

'''#print(len(z))
fit = np.polyfit(mas_ang_vel, mas_avg_freq, 1)  # make a fit
print(fit)
xfit = mas_ang_vel  # save the fit
print(xfit)
line = fit[0] * xfit + fit[1]  # plot the lin3
#plt.plot(mode)  # compare to imput
plt.plot(line)
plt.scatter(mas_ang_vel, mas_avg_freq, c='r'''
plt.show()

'''for i in range(len(mas)-1):
    #mas[i] = mas[i].to_numpy()
    # calc dif times and freq
    print(mas[1])
    dif = mas[i] - mas[i+1]
    #mas.remove(mas[i])
    #mas.remove(mas[i+1])
    print(mas)
    print('Разность времен: ', dif)
    print('diff time', dif)
    freq = dif * velocity_sweep
    print('Разность частот: ', freq)
    avg_freq = np.mean(freq)
    print("Сдвиг частот ср.: ", avg_freq)
    noise_freq = np.std(freq)
    print("Шум частот : ", noise_freq)
    rotation = (avg_freq / 272) + (4401 / 272)
    print("Угловая скорость : ", rotation)
    rotation_noise = noise_freq / 272
    print("Шум угловой скорости : ", rotation_noise)
            # Do Some Task
for root, dirs, files in os.walk('E:\KDA\\files\\230721\\tektronix'):
    for file in files:
        filename, extension = os.path.splitext(file)
        if filename.find('Ch2') != -1 and filename.find('-') ==-1:
            print(filename)
            # for ch2
            y2 = df_ch2.values.flatten()
            b, a = signal.butter(N_filt, F_filt_ch2, fs=(1 / ticks))
            z2 = signal.filtfilt(b, a, y2)
            # z2 = savgol_filter(y2, 2001, 5)
            zdf_ch2 = pd.DataFrame(data=z2, columns=['column1'])
            df_ch2_norm = (zdf_ch2 - zdf_ch2.min()) / (zdf_ch2.max() - zdf_ch2.min())
            zdf_ch2['min'] = df_ch2_norm.column1[(df_ch2_norm.column1.shift(1) > df_ch2_norm.column1) & (
                        df_ch2_norm.column1.shift(-1) > df_ch2_norm.column1)]
            sortdf_ch2 = zdf_ch2['min'].where(zdf_ch2['min'] < trashold)
            # Find only peaks
            indexdf = sortdf_ch2.notnull()
            # Find index peaks
            index_peaks_ch2 = zdf_ch2[indexdf].index
            peaks_time_ch2 = index_peaks_ch2 * ticks
            peaks_time_ch2_array = len(peaks_time_ch2.values)

            print('Index peaks ch2', index_peaks_ch2)
            print('Quant peaks ch2', peaks_time_ch2_array)
            print('Time peaks ch2', peaks_time_ch2)'''
# for i in csv_files_copy:
   # myFile = open(i, 'r')
   # # And rest of code goes in here.
    #myFile.close()
   # print(type(ang_velocities[i]))

    #x = pd.read_csv(csv_files[i], delimiter=',,', names=['time', 'ch1', 'ch2'], skiprows = 11, engine = 'python')
    #df_ch1 = pd.DataFrame(x, columns=['ch1'])
    #df_ch2 = pd.DataFrame(x, columns=['ch2'])
   # ticks = pd.DataFrame(x, columns=['time'])
    #time = ticks * 0.000002
    #cycle_clc = time['time'].shift(0)-time['time'].shift(1)f
   # #cycle = cycle_clc[1].astype(str).astype(float)
    #print(cycle)
    #stro = csv_files[i]
   # print(stro[:4])
    # здесь весь расчет
# массив угловых сороктей
# массив частот