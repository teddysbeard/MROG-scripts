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
fit = False  # Включает и выключает апроксимацию Лоренцом, Гауссом, Фойгхтом из библиотеки lmfit
build_graphs = False
nonfit = False  # Включает  нахождение пиков в изначально отфильтрованном функцией  filter_signal и /
# отнормированном сигнале функцией normalization_and_find_peaks
moving_average_const = False  # Включает алгоритм со скользящим средним
convolution_const = True
mas_avg_freq = []
mas_ang_vel = []
# ang_velocity = ''
mas_freq_noise = []
# int_fit_ch1 = []
# int_fit_ch2 = []
# center_values_ch1 = []
# center_values_ch2 = []

# mas_fwhm_ch1 = []
# mas_fwhm_ch2 = []

mas_FSR_ch1 = []
mas_FSR_ch2 = []
files = os.listdir(path='E:\\KDA\\files\\smfresonator')
# print(files)
measurements = list(filter(lambda x: x.endswith('.csv'), files))  # Фильтруем список названий файлов
print(measurements, len(measurements))

# Сортируем массив названий файлов измерений по угловой скорости
count = 0
for run in range(len(measurements) - 1):
    for i in range(len(measurements) - 1 - run):
        # print(i)
        # print('сравниваю ',mas[i], 'c',mas[i+1])
        if int(measurements[i][:measurements[i].index('ds_')]) > int(
                measurements[i + 1][:measurements[i + 1].index('ds_')]):
            # print(measurements[i][:measurements[i].index('ds_')])
            # print(measurements[i+1][:measurements[i+1].index('ds_')])
            measurements[i], measurements[i + 1] = measurements[i + 1], measurements[i]
            count += 1
print(measurements)
print(len(measurements))
# print(files)
# print(measurements, len(measurements))

for i in range(0, len(measurements) - 1, 2):
    print(i)


    def open_file_and_create_data_frame(filename):
        # download file measurement
        print(filename)
        x = pd.read_csv('E:\\KDA\\files\\smfresonator\\%s' % filename, delimiter=',',
                        names=['2', '1', 'ch1'],
                        skiprows=6, engine='python')
        df_ch = pd.DataFrame(x, columns=['ch1'])
        # print(df_ch)
        t = pd.DataFrame(x, columns=['1'])
        # print(t)
        # for ch1
        y = df_ch.values.flatten()
        # print('y', y, len(y))
        # print(measurements[0])
        return y, df_ch, t


    # open_file_and_create_data_frame(filename=measurements[i])
    y1 = open_file_and_create_data_frame(filename=measurements[i])[0]
    df_ch1 = open_file_and_create_data_frame(filename=measurements[i])[1]

    y2 = open_file_and_create_data_frame(filename=measurements[i + 1])[0]
    df_ch2 = open_file_and_create_data_frame(filename=measurements[i + 1])[1]


    def ang_velocities(end_cut_filename, filename):
        ''' Из имени файла вырезаем теоретическое значение угловой скорости'''

        # if filename.find(channel) != -1 and filename.find(sign) != -1:
        # print(filename)
        # print(filename.find('_degs_'))
        ang_velocity = int(filename[0:filename.find(end_cut_filename)])
        print(ang_velocity)
        if ang_velocity not in mas_ang_vel:
            mas_ang_vel.append(ang_velocity)  # create massive of angular velocities
        return ang_velocity


    ang_velocities(end_cut_filename='ds_', filename=measurements[i])
    if convolution_const == True:
        print('measurements[i]', measurements[i])

        alpha = 20
        w_size = 1100
        Nd = 199994
        size_wind = 128  # размер адаптивного окна справа и слева
        kth = 1.1  # коэффициент адаптивного окна
        TH_FIX = 0.1  # уровень фиксированного порога для отсекания ложных пиков
        windowSize = 20  # размер окна усреднения
        windet = 1 / size_wind

        build_graphs_conv = False  # Включает/выключает построение графиков

        # for ch1
        y1 = df_ch1.values.flatten()


        # функция создания окна
        def gauss_window(alpha, w_size):

            sigma = (w_size - 1) / (2 * alpha)

            window = signal.windows.gaussian(w_size, sigma)

            return window


        # функция свертки
        def conv(data, window):

            conv = np.convolve(data, window, mode='same')
            print(type(conv))
            return conv


        # вызов функций для ch1

        x = gauss_window(alpha, w_size)
        y1_graph = conv(y1, x) * (alpha / w_size)
        print(y1_graph)

        # построение графиков
        if build_graphs_conv == True:
            plt.plot(x, c='m', label='window')
            plt.plot(y1, c='b', label='conv')
            plt.plot(y1_graph, c='g', label='data')
            plt.legend(loc='best')
            plt.show()

        # for ch2
        print('measurements[i]', measurements[i+1])
        y2 = df_ch2.values.flatten()

        # вызов функций для ch2

        x = gauss_window(alpha, w_size)
        y2_graph = conv(y2, x) * (alpha / w_size)
        print(y2_graph)
        # построение графиков для ch2

        if build_graphs_conv == True:
            plt.plot(x, c='m', label='window')
            plt.plot(y2_graph, c='b', label='conv')
            plt.plot(y2, c='g', label='data')
            plt.legend(loc='best')
            plt.show()
        # Поиск пиков для данных ch1
        mas_data_ch1 = []
        mas_data_indices_ch1 = []
        for k in range(130, (Nd - 128 - 1 + 1)):
            #th1 = sum(y1_graph[k - size_wind - 1:k - 1 - 1]) * kth * windet
            #th2 = sum(y1_graph[k + 1 + 1:k + size_wind + 1]) * kth * windet
            # print(th1)
            # print(th2)
            if  y1_graph[k] > TH_FIX and y1_graph[k] > y1_graph[k + 1] and y1_graph[k] > y1_graph[k - 1]:
                mas_data_ch1.append(y1_graph[k])

                mas_data_indices_ch1.append(k)
        num = 0
        Nd_Det = []
        print('mas_data initial', len(mas_data_ch1), mas_data_ch1)
        mas_data_current_ch1 = []
        # Дополнительная проверка пиков для данных ch1
        for k in mas_data_indices_ch1:
            # print('iteration')
            mas_data_current_ch1.append(max(y1_graph[k - 50:k + 51]))
        print('mas_data current', len(mas_data_current_ch1), mas_data_current_ch1)
        if mas_data_current_ch1 == mas_data_ch1:
            print('Yes')

        mas_fwhm_ch = []
        mas_FSR_ch = []

        # calculation fwhm (averaged over all peaks)
        # ??????

        # calculation FSR ch1

        for j in range(len(mas_data_indices_ch1) - 1):
            mas_FSR_ch.append((mas_data_indices_ch1[j + 1] - mas_data_indices_ch1[j]) * ticks)
        print('mas_FSR_Ch: ', mas_FSR_ch, len(mas_FSR_ch))
        fsr_avg = sum(mas_FSR_ch) / len(mas_FSR_ch) * velocity_sweep
        # fwhm_avg = sum(mas_fwhm_ch) / len(mas_fwhm_ch) * velocity_sweep
        # print('fwhm_avg', fwhm_avg)
        print('fsr_avg', fsr_avg)
        mas_FSR_ch.clear()

        # Поиск пиков для данных ch2
        mas_data_ch2 = []
        mas_data_indices_ch2 = []
        for k in range(130, (Nd - 128 - 1 + 1)):
            #th1 = sum(y2_graph[k - size_wind - 1:k - 1 - 1]) * kth * windet
            #th2 = sum(y2_graph[k + 1 + 1:k + size_wind + 1]) * kth * windet
            # print(th1)
            # print(th2)
            if y2_graph[k] > TH_FIX and y2_graph[k] > y2_graph[k + 1] and y2_graph[k] > y2_graph[k - 1]:
                mas_data_ch2.append(y2_graph[k])

                mas_data_indices_ch2.append(k)
        num = 0
        Nd_Det = []
        print('mas_data initial', len(mas_data_ch2), mas_data_ch2)
        mas_data_current_ch2 = []

        # Дополнительная проверка пиков для данных ch2

        for k in mas_data_indices_ch2:
            # print('iteration')
            mas_data_current_ch2.append(max(y2_graph[k - 50:k + 51]))
        print('mas_data current', len(mas_data_current_ch2), mas_data_current_ch2)
        if mas_data_current_ch2 == mas_data_ch2:
            print('Yes')


        # calculation FSR ch2

        for j in range(len(mas_data_indices_ch2) - 1):
            mas_FSR_ch.append((mas_data_indices_ch2[j + 1] - mas_data_indices_ch2[j]) * ticks)
        print('mas_FSR_Ch: ', mas_FSR_ch, len(mas_FSR_ch))
        fsr_avg = sum(mas_FSR_ch) / len(mas_FSR_ch) * velocity_sweep
        # fwhm_avg = sum(mas_fwhm_ch) / len(mas_fwhm_ch) * velocity_sweep
        # print('fwhm_avg', fwhm_avg)
        print('fsr_avg', fsr_avg)

        # Расчет разности частот, шума
        mas_freq = []
        mas_dif = []
        if len(mas_data_indices_ch1) == len(mas_data_indices_ch2):
            for j in range(len(mas_data_indices_ch1)):
                dif = ((mas_data_indices_ch1[j] - mas_data_indices_ch2[j])) * ticks
                # dif = (mas_data_indices_ch1[j] - mas_data_indices_ch2[j]) * ticks
                # print('Difference', dif1)
                # print('Разность времен: ', dif)
                freq = dif * velocity_sweep
                # print('Разность частот: ', freq)
                mas_freq.append(freq)
                # mas_dif.append(dif)
            # print(mas_dif)
            avg_freq = np.mean(mas_freq)
            print("Сдвиг частот ср.: ", avg_freq)
            mas_avg_freq.append(avg_freq)
            noise_freq = np.std(mas_freq)
            print("Шум частот : ", noise_freq)
            mas_freq_noise.append(noise_freq)

    elif moving_average_const == True:
        print('moving average')


        def moving_average(df_ch_arg):
            T_OBL = 199994  # число последовательных отсчетов АЦП из входного файла для обработки и визуализации работы алгоритма
            obl_vibora = 100  # ед.изм. - отсчеты АЦП, область определения единственного,
            # самого максимального (или минимального) экстремума среди прошедших окно обнаружения
            size_wind = 128  # размер адаптивного окна справа и слева
            kth = 1.1  # коэффициент адаптивного окна
            TH_FIX = 0.1  # уровень фиксированного порога для отсекания ложных пиков
            windowSize = 20  # размер окна усреднения

            # Сглаживание методом скользящего среднего
            rolling_mean = df_ch_arg.ch1.rolling(window=20).mean()  # Размер окна усреднения 20
            rolling_mean2 = df_ch_arg.ch1.rolling(window=50).mean()  # Размер окна усреднения 50
            # Адаптивное скользящее окно обнаружения
            Nd = T_OBL
            mas_data = []  # то же самое что и M_Det
            mas_data_indices = []
            windet = 1 / size_wind
            DATA_THRESHOLD = rolling_mean.values.flatten().tolist()
            counter = 0
            for k in range(130, (Nd - 128 - 1 + 1)):
                th1 = sum(DATA_THRESHOLD[k - size_wind - 1:k - 1 - 1]) * kth * windet
                th2 = sum(DATA_THRESHOLD[k + 1 + 1:k + size_wind + 1]) * kth * windet
                counter += 1
                # print(th1)
                # print(th2)
                if DATA_THRESHOLD[k] > th1 and DATA_THRESHOLD[k] > th2 and DATA_THRESHOLD[k] > TH_FIX and \
                        DATA_THRESHOLD[k] > \
                        DATA_THRESHOLD[k + 1] and DATA_THRESHOLD[k] > DATA_THRESHOLD[k - 1]:
                    mas_data.append(DATA_THRESHOLD[k])

                    mas_data_indices.append(k)
            num = 0
            Nd_Det = []
            print('mas_data initial', len(mas_data), mas_data)
            mas_data_current = []
            for k in mas_data_indices:
                # print('iteration')
                mas_data_current.append(max(DATA_THRESHOLD[k - 50:k + 51]))
            print('mas_data current', len(mas_data_current), mas_data_current)
            if mas_data_current == mas_data:
                print('Yes')

            '''for j in range(51, (len(DATA_THRESHOLD) - 50)):
                mas_left = DATA_THRESHOLD[j - 50:j - 1]
                mas_right = DATA_THRESHOLD[j + 1:j + 50]
                #print(len(mas_left))
                #print(len(mas_right))
                for k in mas_data_indices:
                    for m in range(len(mas_left)):
                        if DATA_THRESHOLD[k] > mas_left[m] and DATA_THRESHOLD[k] > mas_right[m]:
                            Nd_Det.append(DATA_THRESHOLD[k])
            print('Nd_Det', Nd_Det)'''

            '''for j in range(51, (len(DATA_THRESHOLD) - 50)):
                for k in range():
                mas_left = DATA_THRESHOLD[j - 50:j - 1]
                mas_right = DATA_THRESHOLD[j + 1:j + 50]
                print(len(mas_left))
                print(len(mas_right))'''

            '''for j in range(51, (len(mas_data_indices) - 50)):
                mas_left = mas_data[j - 50:j - 1]
                mas_right = mas_data[j + 1:j + 50]
                print(len(mas_left))
                print(len(mas_right))
                for k in range(len(mas_left)):
                    for m in range(len(mas_right)):
                        if mas_data[j] > mas_left[k] and mas_data[j] > mas_right[m]:
                            mas_data[j] = mas_data[j]
                            num = num + 1
                            Nd_Det.append(j)
                        else:
                            mas_data[j] = 0
            print('mas_data current', len(mas_data), mas_data)'''
            # print(mas_data)
            # print(mas_data_indices)
            # print(counter)
            # plt.plot(mas_data_indices, mas_data, '*', label='M_Det', color='green')
            # plt.plot([i for i in range(1, len(rolling_mean) + 1)], rolling_mean, label='20', color='orange')
            # plt.plot([i for i in range(1, len(rolling_mean2) + 1)], rolling_mean2, label='50', color='magenta')
            # plt.legend(loc='upper left')
            # plt.show()
            return mas_data, mas_data_indices


        moving_average(df_ch_arg=df_ch1)
        mas_data_mov_av_ch1 = moving_average(df_ch_arg=df_ch1)[0]
        mas_data_indices_ch1 = moving_average(df_ch_arg=df_ch1)[1]
        print(mas_data_mov_av_ch1)
        print(len(mas_data_mov_av_ch1))
        print('mas_data_indices_ch1', mas_data_indices_ch1)
        mas_fwhm_ch = []
        mas_FSR_ch = []

        # calculation fwhm (averaged over all peaks)
        # ??????

        # calculation FSR ch1

        for j in range(len(mas_data_indices_ch1) - 1):
            mas_FSR_ch.append((mas_data_indices_ch1[j + 1] - mas_data_indices_ch1[j]) * ticks)
        print('mas_FSR_Ch: ', mas_FSR_ch, len(mas_FSR_ch))
        fsr_avg = sum(mas_FSR_ch) / len(mas_FSR_ch) * velocity_sweep
        # fwhm_avg = sum(mas_fwhm_ch) / len(mas_fwhm_ch) * velocity_sweep
        # print('fwhm_avg', fwhm_avg)
        print('fsr_avg', fsr_avg)
        mas_FSR_ch.clear()

        # FOR CH2
        open_file_and_create_data_frame(filename=measurements[i + 1])

        y2 = open_file_and_create_data_frame(filename=measurements[i + 1])[0]
        df_ch2 = open_file_and_create_data_frame(filename=measurements[i + 1])[1]

        ang_velocities(end_cut_filename='ds_', filename=measurements[i + 1])

        moving_average(df_ch_arg=df_ch2)
        mas_data_mov_av_ch2 = moving_average(df_ch_arg=df_ch2)[0]
        mas_data_indices_ch2 = moving_average(df_ch_arg=df_ch2)[1]
        print(mas_data_mov_av_ch2)
        print('mas_data_indices_ch2', mas_data_indices_ch2)
        print(len(mas_data_mov_av_ch2))

        # calculation FSR ch2

        for j in range(len(mas_data_indices_ch2) - 1):
            mas_FSR_ch.append((mas_data_indices_ch2[j + 1] - mas_data_indices_ch2[j]) * ticks)
        print('mas_FSR_Ch: ', mas_FSR_ch, len(mas_FSR_ch))
        fsr_avg = sum(mas_FSR_ch) / len(mas_FSR_ch) * velocity_sweep
        # fwhm_avg = sum(mas_fwhm_ch) / len(mas_fwhm_ch) * velocity_sweep
        # print('fwhm_avg', fwhm_avg)
        print('fsr_avg', fsr_avg)

        mas_freq = []
        mas_dif = []
        if len(mas_data_indices_ch1) == len(mas_data_indices_ch2):
            for j in range(len(mas_data_indices_ch1)):
                dif = ((mas_data_indices_ch1[j] - mas_data_indices_ch2[j])) * ticks
                # dif = (mas_data_indices_ch1[j] - mas_data_indices_ch2[j]) * ticks
                # print('Difference', dif1)
                # print('Разность времен: ', dif)
                freq = dif * velocity_sweep
                # print('Разность частот: ', freq)
                mas_freq.append(freq)
                # mas_dif.append(dif)
            # print(mas_dif)
            avg_freq = np.mean(mas_freq)
            print("Сдвиг частот ср.: ", avg_freq)
            mas_avg_freq.append(avg_freq)
            noise_freq = np.std(mas_freq)
            print("Шум частот : ", noise_freq)
            mas_freq_noise.append(noise_freq)
        else:
            print('Error')

    elif moving_average_const == False and convolution_const == False:

        def filter_signal(var_y):
            b, a = signal.butter(N_filt, F_filt, fs=(1 / ticks))
            z = signal.filtfilt(b, a, var_y)
            zdf_ch = pd.DataFrame(data=z, columns=['column1'])
            # print('zdf_ch', zdf_ch, len(zdf_ch), type(zdf_ch))
            # print(type(z))
            return zdf_ch


        filter_signal(var_y=y1)
        zdf_ch1 = filter_signal(var_y=y1)


        # normalization
        def normalization_and_find_peaks(zdf_ch_var):
            df_ch_norm = (zdf_ch_var - zdf_ch_var.min()) / (zdf_ch_var.max() - zdf_ch_var.min())
            # print('df_ch_norm', df_ch_norm)
            # Find all peaks
            zdf_ch_var['max'] = df_ch_norm.column1[(df_ch_norm.column1.shift(1) < df_ch_norm.column1) & (
                    df_ch_norm.column1.shift(-1) < df_ch_norm.column1)]
            sortdf_ch = zdf_ch_var['max'].where(zdf_ch_var['max'] > threshold)
            # print('sortdf_ch', sortdf_ch)
            # Find only peaks
            indexdf = sortdf_ch.notnull()
            index_peaks_ch = zdf_ch_var[indexdf].index
            peaks_time_ch = index_peaks_ch * ticks
            number_of_peaks = len(peaks_time_ch.values)

            # print('Index peaks ch', index_peaks_ch, len(index_peaks_ch))
            # print('number_of_peaks_ch', number_of_peaks)
            # print('Time peaks ch', peaks_time_ch)
            # print('df_ch_norm#', df_ch_norm)
            return df_ch_norm, index_peaks_ch, sortdf_ch


        tuple_df_norm_and_index_peaks = normalization_and_find_peaks(zdf_ch_var=zdf_ch1)

        # print(tuple_df_norm_and_index_peaks)

        # tuple_df_norm_and_index_peaks = normalization_and_find_peaks(zdf_ch_var=zdf_ch1)

        # print(index_peaks_ch1, type(index_peaks_ch1), len(index_peaks_ch1))

        # Find interval fitting ch1
        if fit == True:
            def find_fitting_interval(df_ch_norm, index_peaks_ch_var):
                int_fit_ch = []
                center_values_ch = []
                mas_fwhm_ch = []
                mas_FSR_ch = []
                # print('len', len(index_peaks_ch_var))
                for i in range(1, len(index_peaks_ch_var) - 1):
                    dif_peaks_ch = ((index_peaks_ch_var[i + 1] - index_peaks_ch_var[i]))
                    # print('index_peaks_ch_var[i]', index_peaks_ch_var[i])
                    # print('dif_peaks_ch', dif_peaks_ch)
                    dif_peaks_ch_del = round(dif_peaks_ch / 2)
                    # print('dif_peaks_ch_del', dif_peaks_ch_del)
                    int_fit_ch.append(dif_peaks_ch)
                    # print('int_fit_ch', int_fit_ch)
                    # print('index_peaks_ch_var[i]', i, index_peaks_ch_var[i])
                    # print('df_ch_norm', df_ch_norm, type(df_ch_norm))
                    # s = index_peaks_ch1[0][index_peaks_ch_var[i] - dif_peaks_ch_del: index_peaks_ch_var[i] + dif_peaks_ch_del]
                    s = df_ch_norm[index_peaks_ch_var[i] - dif_peaks_ch_del: index_peaks_ch_var[i] + dif_peaks_ch_del]
                    # print('s', s, type(s))
                    # print('s', s, len(s))
                    # print(s.index)
                    x = s.index * ticks
                    # print('x',x)
                    # print('x', x, len(x))

                    y_fit = s.values.flatten()
                    x_fit = x.values.flatten()
                    # print(y_fit, len(y_fit))
                    # print(x_fit, len(x_fit))

                    # choose model
                    model = model_func

                    # construction fit
                    params = model.guess(y_fit, x=x_fit)
                    result = model.fit(y_fit, params, x=x_fit)

                    # report fit
                    # print(result.fit_report())

                    # calculation fwhm (averaged over all peaks)
                    cut_report_ch = result.fit_report()[result.fit_report().find('fwhm'):]
                    fwhm_ch = float(cut_report_ch[10:cut_report_ch.find('+/-')])
                    # print('fwhm_ch', fwhm_ch)
                    mas_fwhm_ch.append(fwhm_ch)
                    # plot fit
                    # result.plot_fit()
                    # plt.show()

                    # find center

                    center_values_ch.append(result.params['center'].value)
                    int_fit_ch.append(dif_peaks_ch)
                print('mas_fwhm_ch', mas_fwhm_ch)
                # print('center_values_ch', center_values_ch)

                # calculation FSR ch

                for i in range(len(center_values_ch) - 1):
                    mas_FSR_ch.append(center_values_ch[i + 1] - center_values_ch[i])
                print('mas_FSR_Ch: ', mas_FSR_ch)
                fsr_avg = sum(mas_FSR_ch) / len(mas_FSR_ch) * velocity_sweep
                fwhm_avg = sum(mas_fwhm_ch) / len(mas_fwhm_ch) * velocity_sweep
                print('fwhm_avg', fwhm_avg)
                print('fsr_avg', fsr_avg)
                return fsr_avg, fwhm_avg, center_values_ch


            result_ch1 = find_fitting_interval(df_ch_norm=tuple_df_norm_and_index_peaks[0],
                                               index_peaks_ch_var=tuple_df_norm_and_index_peaks[1])
            print('Центры пиков функции ch1:', result_ch1[2])
        print()

        '''Часть скрипта для сигнала со 2-го канала'''

        # FOR CH2
        open_file_and_create_data_frame(filename=measurements[i + 1])

        y2 = open_file_and_create_data_frame(filename=measurements[i + 1])[0]
        df_ch2 = open_file_and_create_data_frame(filename=measurements[i + 1])[1]

        ang_velocities(end_cut_filename='ds_', filename=measurements[i + 1])
        zdf_ch2 = filter_signal(var_y=y2)
        tuple_df = normalization_and_find_peaks(zdf_ch_var=zdf_ch2)
        # print(tuple_df)

        if fit == True:
            result_ch2 = find_fitting_interval(df_ch_norm=tuple_df[0], index_peaks_ch_var=tuple_df[1])
            print('Центры пиков функции ch2:', result_ch2[2])

        # calc dif times and freq
        if fit == True:
            def dif_times_and_freqs_polyn(center_values_ch1, center_values_ch2):
                dif = []
                freq = []
                print(len(center_values_ch1))
                print(len(center_values_ch2))
                for i in range(0, len(center_values_ch1)):
                    dif.append((center_values_ch1[i] - center_values_ch2[i]))
                for i in range(0, len(center_values_ch1)):
                    freq.append((center_values_ch1[i] - center_values_ch2[i]) * velocity_sweep)
                # print('Разность времен: ', dif)
                # print('Разность частот: ', freq)
                avg_freq = np.mean(freq)
                print("Сдвиг частот ср.: ", avg_freq)
                mas_avg_freq.append(avg_freq)
                noise_freq = np.std(freq)
                # print("Шум частот : ", noise_freq)
                mas_freq_noise.append(noise_freq)
                return


            dif_times_and_freqs_polyn(center_values_ch1=result_ch1[2], center_values_ch2=result_ch2[2])
        '''elif nonfit == True:
            def dif_times_and_freqs(peaks_time_ch1, peaks_time_ch2):
                dif = (peaks_time_ch1 - peaks_time_ch2) * ticks
                # print('Difference', dif1)
                # print('Разность времен: ', dif)
                freq = dif * velocity_sweep
                # print('Разность частот: ', freq)
                avg_freq = np.mean(freq)
                print("Сдвиг частот ср.: ", avg_freq)
                mas_avg_freq.append(avg_freq)
                noise_freq = np.std(freq)
                print("Шум частот : ", noise_freq)
                mas_freq_noise.append(noise_freq)
                return


            dif_times_and_freqs(peaks_time_ch1=tuple_df_norm_and_index_peaks[1], peaks_time_ch2=tuple_df[1])'''

        if build_graphs == True:
            def plot_graphs(a1, b1, c1, a2, b2, c2):
                # Plot
                plt.scatter(a1, b1, c='g')
                plt.scatter(a2, b2, c='r')
                plt.plot(a1, c1, label='CH1_norm')
                plt.plot(a2, c2, label='CH2_norm')
                plt.show()

                return


            plot_graphs(a1=zdf_ch1.index, b1=tuple_df_norm_and_index_peaks[2], c1=tuple_df_norm_and_index_peaks[0],
                        a2=zdf_ch2.index, b2=tuple_df[2], c2=tuple_df[0])
        print('i', i)
        print()
print('Массив угловых скоростей: ', mas_ang_vel)
print('Массив ср. сдвигов частот: ', mas_avg_freq)


# Функция для аппроксимации точек линейной функцией
def objective(x, a, b):
    return a * x + b


# choose the input and output variables
x, y = mas_ang_vel, mas_avg_freq
# curve fit
popt, _ = curve_fit(objective, x, y)
# summarize the parameter values
a, b = popt
print('Передаточная характеристика: ', 'y = %.5f * x + %.5f' % (a, b))
print('Масштабный коэффициент: ', a)
print('Смещение нуля: ', b)

# plot input vs output
plt.scatter(x, y)
# define a sequence of inputs between the smallest and largest known inputs
x_line = arange(min(x), max(x), 1)
# calculate the output for the range
y_line = objective(x_line, a, b)
# create a line plot for the mapping function
plt.plot(x_line, y_line, '--', color='red')
plt.show()

# Блок для расчета угловых скоростей

ang_velocity_measured = []  # Массив измеренных угловых скоростей

for i in range(len(mas_avg_freq)):
    rot_rate = (mas_avg_freq[i] / a) - (b / a)
    ang_velocity_measured.append(rot_rate)
print('Массив угловых скоростей: ', mas_ang_vel)
print("Массив измеренных угловых скоростей: ", ang_velocity_measured)

# Блок для расчета шума угловой скорости

mas_rotation_noise = []
for i in range(len(mas_freq_noise)):
    rotation_noise = mas_freq_noise[i] / a
    mas_rotation_noise.append(rotation_noise)
print("Шум угловой скорости: ", mas_rotation_noise)

# Блок для расчета нелинейности масщтабного коэффицииента

mas_fit_avg_freq = []
mas_nonlin = []
x1 = mas_ang_vel
for i in range(len(mas_ang_vel)):
    mas_fit_avg_freq.append(a * mas_ang_vel[i] + b)
# print(mas_fit_avg_freq)
for i in range(len(mas_avg_freq)):
    nonlin = (mas_fit_avg_freq[i] - mas_avg_freq[i]) * 100 / abs(max(mas_avg_freq))
    mas_nonlin.append(nonlin)
print('Нелинейность масштабного коэффициента: ', mas_nonlin)
plt.scatter(mas_ang_vel, mas_nonlin, c='g')
plt.show()

# Запись в файл отчета
my_file = open("report.txt", "w+")
my_file.write('Массив ср. сдвигов частот: ' + str(mas_avg_freq) + '\n')
my_file.write("Массив угловых скоростей:" + str(mas_ang_vel) + '\n')
my_file.write("Массив измеренных угловых скоростей: " + str(ang_velocity_measured) + '\n')
my_file.write('Передаточная характеристика: y = %.5f * x + %.5f' % (a, b) + '\n')
my_file.write('Масштабный коэффициент: ' + str(a) + '\n')
my_file.write('Смещение нуля: ' + str(b) + '\n')
my_file.write("Шум угловой скорости: " + str(mas_rotation_noise) + '\n')
my_file.write('Нелинейность масштабного коэффициента: ' + str(mas_nonlin))
my_file.close()
