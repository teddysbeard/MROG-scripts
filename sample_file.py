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
from scipy.optimize import curve_fit
from lmfit.models import VoigtModel, LorentzianModel, GaussianModel
import os
import glob
from scipy.optimize import curve_fit
from numpy import arange

# const
ADC_freq_Hz = 500000
ticks = 1 / ADC_freq_Hz
velocity_sweep = 5000000000
# без нормировки уровень 0,4, фильтр 1300
threshold = 0.35
N_filt = 1
F_filt = 900
F_filt_ch2 = 900
model_func = LorentzianModel()

# mas_avg_freq = []
# mas_ang_vel = []
# ang_velocity = ''
# mas_freq_noise = []
#int_fit_ch1 = []
# int_fit_ch2 = []
#center_values_ch1 = []
# center_values_ch2 = []

mas_fwhm_ch1 = []
mas_fwhm_ch2 = []

mas_FSR_ch1 = []
mas_FSR_ch2 = []
files = os.listdir(path='E:\\KDA\\files\\smfresonator')
measurements = list(filter(lambda x: x.endswith('.csv'), files))  # Фильтруем список названий файлов
print(files)
print(measurements)

path = 'E:\\KDA\\files\\smfresonator'
channel = "Ch1"
sign = '+'
end_cut_filename = 'ds_'


def open_file_and_create_data_frame(filename):
    path = 'E:\\KDA\\files\\smfresonator'
    channel = "Ch1"
    sign = '+'
    end_cut_filename = 'ds_'
    mass_y = []
    # download file mesurment
    # filename = '+0ds_Ch1'
    x = pd.read_csv('E:\\KDA\\files\\smfresonator\\%s' % filename, delimiter=',',
                    names=['2', '1', 'ch1'],
                    skiprows=6, engine='python')
    df_ch = pd.DataFrame(x, columns=['ch1'])
    t = pd.DataFrame(x, columns=['1'])
    # for ch1
    y = df_ch.values.flatten()
    print('y', y, len(y))
    # print(measurements[0])
    return y


open_file_and_create_data_frame(filename=measurements[0])
y = open_file_and_create_data_frame(filename=measurements[0])


def ang_velocities(channel, sign, end_cut_filename):
    ''' Из имени файла вырезаем теоретическое значение угловой скорости'''

    if measurements[0].find(channel) != -1 and measurements[0].find(sign) != -1:
        print(measurements[0])
        # print(filename.find('_degs_'))
        ang_velocity = int(measurements[0][0:measurements[0].find(end_cut_filename)])
        print(ang_velocity)
        # mas_ang_vel.append(ang_velocity)  # create massive of angular velocities
        return ang_velocity


ang_velocities(channel="Ch1", sign='+', end_cut_filename='ds_')


def filter_signal(var_y):
    b, a = signal.butter(N_filt, F_filt, fs=(1 / ticks))
    z = signal.filtfilt(b, a, var_y)
    zdf_ch = pd.DataFrame(data=z, columns=['column1'])
    print('zdf_ch', zdf_ch, len(zdf_ch), type(zdf_ch))
    print(type(z))
    return zdf_ch


filter_signal(var_y=y)
zdf_ch1 = filter_signal(var_y=y)


# normalization
def normalization_and_find_peaks(zdf_ch_var):
    df_ch_norm = (zdf_ch_var - zdf_ch_var.min()) / (zdf_ch_var.max() - zdf_ch_var.min())
    print('df_ch_norm', df_ch_norm)
    # Find all peaks
    zdf_ch_var['max'] = df_ch_norm.column1[(df_ch_norm.column1.shift(1) < df_ch_norm.column1) & (
            df_ch_norm.column1.shift(-1) < df_ch_norm.column1)]
    sortdf_ch = zdf_ch_var['max'].where(zdf_ch_var['max'] > threshold)
    print('sortdf_ch', sortdf_ch)
    # Find only peaks
    indexdf = sortdf_ch.notnull()
    index_peaks_ch = zdf_ch_var[indexdf].index
    peaks_time_ch = index_peaks_ch * ticks
    number_of_peaks = len(peaks_time_ch.values)

    print('Index peaks ch', index_peaks_ch, len(index_peaks_ch))
    print('number_of_peaks_ch', number_of_peaks)
    # print('Time peaks ch', peaks_time_ch)
    print('df_ch_norm#', df_ch_norm)
    return df_ch_norm, index_peaks_ch


tuple_df_norm_and_index_peaks = normalization_and_find_peaks(zdf_ch_var=zdf_ch1)
print(tuple_df_norm_and_index_peaks)


# tuple_df_norm_and_index_peaks = normalization_and_find_peaks(zdf_ch_var=zdf_ch1)

# print(index_peaks_ch1, type(index_peaks_ch1), len(index_peaks_ch1))

# Find interval fitting ch1
def find_fitting_interval(df_ch_norm, index_peaks_ch_var):
    int_fit_ch = []
    center_values_ch = []
    print('len', len(index_peaks_ch_var))
    for i in range(1, len(index_peaks_ch_var) - 1):
        dif_peaks_ch = ((index_peaks_ch_var[i + 1] - index_peaks_ch_var[i]))
        print('index_peaks_ch_var[i]', index_peaks_ch_var[i])
        print('dif_peaks_ch', dif_peaks_ch)
        dif_peaks_ch_del = round(dif_peaks_ch / 2)
        print('dif_peaks_ch_del', dif_peaks_ch_del)
        int_fit_ch.append(dif_peaks_ch)
        print('int_fit_ch', int_fit_ch)
        print('index_peaks_ch_var[i]', i, index_peaks_ch_var[i])
        print('df_ch_norm', df_ch_norm, type(df_ch_norm))
        # s = index_peaks_ch1[0][index_peaks_ch_var[i] - dif_peaks_ch_del: index_peaks_ch_var[i] + dif_peaks_ch_del]
        s = df_ch_norm[index_peaks_ch_var[i] - dif_peaks_ch_del: index_peaks_ch_var[i] + dif_peaks_ch_del]
        print('s', s, type(s))
        # print('s', s, len(s))
        x = s.index * ticks
        # print('x', x, len(x))

        y_fit = s.values.flatten()
        x_fit = x.values.flatten()
        print(y_fit, len(y_fit))
        print(x_fit, len(x_fit))

        # choose model
        model = model_func

        # construction fit
        params = model.guess(y_fit, x=x_fit)
        result = model.fit(y_fit, params, x=x_fit)

        # report fit
        # print(result.fit_report())
        return result


result_ch1 = find_fitting_interval(df_ch_norm=tuple_df_norm_and_index_peaks[0],
                                   index_peaks_ch_var=tuple_df_norm_and_index_peaks[1])


# calculation fwhm (averaged over all peaks)
def FWHM(result_var):
    #center_values_ch = []
    cut_report_ch = result_var.fit_report()[result_var.fit_report().find('fwhm'):]
    fwhm_ch = float(cut_report_ch[10:cut_report_ch.find('+/-')])
    print('fwhm_ch', fwhm_ch)
    # mas_fwhm_ch.append(fwhm_ch)

    # plot fit
    # result.plot_fit()
    # plt.show()
    return fwhm_ch


FWHM(result_var=result_ch1)


def find_center(result_var):
    # find center
    center_values_ch = []
    center_values_ch.append(result_var.params['center'].value)
    print('center_values_ch', center_values_ch)
    # int_fit_ch.append(dif_peaks_ch)
    return center_values_ch


find_center(result_var=result_ch1)


# calculation FSR ch1
def calc_FSR():
    for i in range(len(center_values_ch1) - 1):
        mas_FSR_ch1.append(center_values_ch1[i + 1] - center_values_ch1[i])
    print('mas_FSR_Ch1: ', mas_FSR_ch1)
    print('FSR_Ch1', sum(mas_FSR_ch1) / len(mas_FSR_ch1) * velocity_sweep)
    print(mas_fwhm_ch1)
    print('FWHM_Ch1: ', sum(mas_fwhm_ch1) / len(mas_fwhm_ch1) * velocity_sweep)



# FOR CH2
open_file_and_create_data_frame(filename=measurements[1])

y2 = open_file_and_create_data_frame(filename=measurements[1])

ang_velocities(channel="Ch2", sign='+', end_cut_filename='ds_')
zdf_ch2 = filter_signal(var_y=y2)
tuple_df = normalization_and_find_peaks(zdf_ch_var=zdf_ch2)
print(tuple_df)
result_ch2 = find_fitting_interval(df_ch_norm=tuple_df[0], index_peaks_ch_var=tuple_df[1])
FWHM(result_var=result_ch2)
find_center(result_var=result_ch2)

# calculation FSR ch1
def calc_FSR():
    for i in range(len(center_values_ch1) - 1):
        mas_FSR_ch1.append(center_values_ch1[i + 1] - center_values_ch1[i])
    print('mas_FSR_Ch1: ', mas_FSR_ch1)
    print('FSR_Ch1', sum(mas_FSR_ch1) / len(mas_FSR_ch1) * velocity_sweep)
    print(mas_fwhm_ch1)
    print('FWHM_Ch1: ', sum(mas_fwhm_ch1) / len(mas_fwhm_ch1) * velocity_sweep)

# for i in measurements:
# open_files(filename=i)'''

