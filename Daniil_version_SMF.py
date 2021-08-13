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


#const###
ADC_freq_Hz = 500000
ticks = 1/ADC_freq_Hz
velocity_sweep = 5000000000
#без нормировки уровень 0,4, фильтр 1300
trashold = 0.4
N_filt = 1
F_filt_ch1 = 900
F_filt_ch2 = 900

#counter files CSV in work dir
myPath = 'E:\\KDA\\files\\smfresonator'
files = os.listdir(myPath)
csv_files = list(filter(lambda s: s.endswith('.csv'), files))
#print(csv_files)
csv_files_copy = csv_files.copy()

mas_avg_freq = []
mas_ang_vel = []
ang_velocity = ''
mas_freq_noise = []
# Цикл для открытия нескольких файлов
for root, dirs, files in os.walk('E:\\KDA\\files\\smfresonator'):
    for file in files:
        filename, extension = os.path.splitext(file)
# Часть цикла для положительного направления вращения
        if filename.find('Ch1') != -1 and filename.find('-') ==-1:
            print(filename)
            #print(filename.find('_degs_'))
            ang_velocity = int(filename[0:filename.find('ds_')])
            print(ang_velocity)
            mas_ang_vel.append(ang_velocity)
            #mas_ang_vel.append()
            # download file mesurment
            x1 = pd.read_csv('E:\\KDA\\files\\smfresonator\\%s.csv' % filename, delimiter=',', names=['2', '1', 'ch1'], skiprows=6, engine='python')
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
            zdf_ch1['max'] = df_ch1_norm.column1[(df_ch1_norm.column1.shift(1) < df_ch1_norm.column1) & (
                        df_ch1_norm.column1.shift(-1) < df_ch1_norm.column1)]
            sortdf_ch1 = zdf_ch1['max'].where(zdf_ch1['max'] > trashold)

            # Find only peaks
            indexdf = sortdf_ch1.notnull()
            # Find index peaks
            index_peaks_ch1 = zdf_ch1[indexdf].index
            peaks_time_ch1 = index_peaks_ch1 * ticks
            peaks_time_ch1_array = len(peaks_time_ch1.values)

            # print('Index peaks ch1', index_peaks_ch1)
            # print('Quant peaks ch1', peaks_time_ch1_array)
            # print('Time peaks ch1', peaks_time_ch1)
            # mas.append(peaks_time_ch1)
        else:
            if filename.find('Ch2') != -1 and filename.find('-') == -1:
                print(filename)
                #print(filename.find('_degs_'))
                ang_velocity = filename[0:filename.find('ds_')]
                print(ang_velocity)
                #mas_ang_vel.append(ang_velocity)
                x2 = pd.read_csv('E:\\KDA\\files\\smfresonator\\%s.csv' % filename, delimiter=',',
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
                zdf_ch2['max'] = df_ch2_norm.column1[(df_ch2_norm.column1.shift(1) < df_ch2_norm.column1) & (
                            df_ch2_norm.column1.shift(-1) < df_ch2_norm.column1)]
                sortdf_ch2 = zdf_ch2['max'].where(zdf_ch2['max'] > trashold)

                # Find only peaks
                indexdf = sortdf_ch2.notnull()
                # Find index peaks
                index_peaks_ch2 = zdf_ch2[indexdf].index
                peaks_time_ch2 = index_peaks_ch2 * ticks
                peaks_time_ch2_array = len(peaks_time_ch2.values)

                #print('Index peaks ch2', index_peaks_ch2)
                #print('Quant peaks ch2', peaks_time_ch2_array)
                #print('Time peaks ch2', peaks_time_ch2)
                dif = peaks_time_ch1 - peaks_time_ch2
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
for root, dirs, files in os.walk('E:\\KDA\\files\\smfresonator'):
    for file in files:
        filename, extension = os.path.splitext(file)
        if filename.find('Ch1') != -1 and filename.find('-') !=-1:
            print(filename)
            #print(filename.find('_degs_'))
            ang_velocity = int(filename[0:filename.find('ds_')])
            print(ang_velocity)
            mas_ang_vel.append(ang_velocity)
            # download file mesurment
            x1 = pd.read_csv('E:\\KDA\\files\\smfresonator\\%s.csv' % filename, delimiter=',', names=['2', '1', 'ch1'],
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
            zdf_ch1['max'] = df_ch1_norm.column1[(df_ch1_norm.column1.shift(1) < df_ch1_norm.column1) & (
                        df_ch1_norm.column1.shift(-1) < df_ch1_norm.column1)]
            sortdf_ch1 = zdf_ch1['max'].where(zdf_ch1['max'] > trashold)

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
                ang_velocity = filename[0:filename.find('ds_')]
                print(ang_velocity)
                #mas_ang_vel.append(ang_velocity)
                x2 = pd.read_csv('E:\\KDA\\files\\smfresonator\\%s.csv' % filename, delimiter=',',
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
                zdf_ch2['max'] = df_ch2_norm.column1[(df_ch2_norm.column1.shift(1) < df_ch2_norm.column1) & (
                            df_ch2_norm.column1.shift(-1) < df_ch2_norm.column1)]
                sortdf_ch2 = zdf_ch2['max'].where(zdf_ch2['max'] > trashold)

                # Find only peaks
                indexdf = sortdf_ch2.notnull()
                # Find index peaks
                index_peaks_ch2 = zdf_ch2[indexdf].index
                peaks_time_ch2 = index_peaks_ch2 * ticks
                peaks_time_ch2_array = len(peaks_time_ch2.values)

                #print('Index peaks ch2', index_peaks_ch2)
                #print('Quant peaks ch2', peaks_time_ch2_array)
                #print('Time peaks ch2', peaks_time_ch2)
                dif = peaks_time_ch1 - peaks_time_ch2
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

# Блок для расчета нелинейности масщтабного коэффицииента
mas_fit_avg_freq = []
mas_nonlin = []
x1 = mas_ang_vel
for i in range(len(mas_ang_vel)):
    x1 = mas_ang_vel[i]
    mas_fit_avg_freq.append(a * x1 + mas_ang_vel[i] + b)
print(mas_fit_avg_freq)
for i in range(len(mas_avg_freq)):
    nonlin = (mas_fit_avg_freq[i] - mas_avg_freq[i])*100/abs(max(mas_avg_freq))
    mas_nonlin.append(nonlin)
print('Нелинейность масштабного коэффициента: ', mas_nonlin)
plt.scatter(mas_ang_vel, mas_nonlin, c='g')
plt.show()

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

#plot graf
plt.scatter(zdf_ch2.index, sortdf_ch1, c='g')
plt.scatter(zdf_ch1.index, sortdf_ch2, c='r')
plt.plot(zdf_ch1.index, z1 , c='m', label = 'filtCH1')
plt.plot(zdf_ch1.index, z2 , c='b', label = 'filtCH2')
plt.plot(zdf_ch1.index, df_ch1 , c='m', label = 'CH1')
plt.plot(zdf_ch2.index, df_ch2 , c='b', label = 'CH2')
plt.legend(loc='best')
ax = plt.gca()
ax2 = ax.twinx()
ax2.set_ylim([-0.1,1])
plt.plot(zdf_ch1.index,df_ch1_norm, label = 'CH1_norm')
plt.plot(zdf_ch1.index,df_ch2_norm, label = 'CH2_norm')
plt.legend(loc='best')
plt.show()

#filter noise
#b, a = signal.butter(1, 10, fs=1/0.003)
#z3 = signal.filtfilt(b, a, freq)
#plt.scatter(peaks_time_ch1, freq, label='freqdif')
#lt.scatter(peaks_time_ch1, z3, label='filtfreqdif')
#plt.show()
#rotation_filt = (np.mean(z3)/272) + (4401/272)
#rotation_noise_filt = np.std(z3) / 272
#print("Угловая скорость после фнч : ", rotation_filt)
#print("Шум угловой скорости после фнч : ", rotation_noise_filt)

#nonlin scale factor


#ARW calculation
#ARW = ((rotation_noise * 3600)/60) * (1/np.sqrt(1))
#print('ARW', ARW)

