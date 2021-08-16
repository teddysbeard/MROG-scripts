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



#const
ADC_freq_Hz = 500000
ticks = 1/ADC_freq_Hz
velocity_sweep = 5000000000
#без нормировки уровень 0,4, фильтр 1300
trashold = 0.35
N_filt = 1
F_filt_ch1 = 900
F_filt_ch2 = 900
model_func = LorentzianModel ()

#download file mesurment
x1 = pd.read_csv('D:\\Прозоров\\python\\smfresonator\\100ds_Ch1.csv', delimiter=',', names=['2', '1', 'ch1'], skiprows = 6, engine = 'python')
x2 = pd.read_csv('D:\\Прозоров\\python\\smfresonator\\100ds_Ch2.csv', delimiter=',', names=['2', '1', 'ch2'], skiprows = 6, engine = 'python')
df_ch2 = pd.DataFrame(x2, columns=['ch2'])
df_ch1 = pd.DataFrame(x1, columns=['ch1'])
t = pd.DataFrame(x1, columns=['1'])

#for ch1
y1 = df_ch1.values.flatten()

#fiters
b, a = signal.butter(N_filt, F_filt_ch1, fs=(1/ticks ))
z1 = signal.filtfilt(b, a, y1)
#z1 = savgol_filter(y1, 2001, 5)

zdf_ch1 = pd.DataFrame(data=z1, columns=['column1'])

#normalization
df_ch1_norm = (zdf_ch1 - zdf_ch1.min()) / (zdf_ch1.max() - zdf_ch1.min())

# Find all peaks
zdf_ch1['max'] = df_ch1_norm.column1[(df_ch1_norm.column1.shift(1) < df_ch1_norm.column1) & (df_ch1_norm.column1.shift(-1) < df_ch1_norm.column1)]
sortdf_ch1 = zdf_ch1['max'].where(zdf_ch1['max'] > trashold)

# Find only peaks
indexdf = sortdf_ch1.notnull()
# Find index peaks
index_peaks_ch1 = zdf_ch1[indexdf].index
peaks_time_ch1 = index_peaks_ch1 * ticks
peaks_time_ch1_array = len(peaks_time_ch1.values)

print('Index peaks ch1', index_peaks_ch1)
print('Quant peaks ch1', peaks_time_ch1_array)
print('Time peaks ch1', peaks_time_ch1)

# for ch2
y2 = df_ch2.values.flatten()

#fiters
b, a = signal.butter(N_filt, F_filt_ch2, fs=(1/ticks))
z2 = signal.filtfilt(b, a, y2)
#z2 = savgol_filter(y2, 2001, 5)

zdf_ch2 = pd.DataFrame(data=z2, columns=['column1'])

#normalization
df_ch2_norm = (zdf_ch2 - zdf_ch2.min()) / (zdf_ch2.max() - zdf_ch2.min())

# Find all peaks
zdf_ch2['max'] = df_ch2_norm.column1[(df_ch2_norm.column1.shift(1) < df_ch2_norm.column1) & (df_ch2_norm.column1.shift(-1) < df_ch2_norm.column1)]
sortdf_ch2 = zdf_ch2['max'].where(zdf_ch2['max'] > trashold)

# Find only peaks
indexdf = sortdf_ch2.notnull()

# Find index peaks
index_peaks_ch2 = zdf_ch2[indexdf].index
peaks_time_ch2 = index_peaks_ch2 * ticks
peaks_time_ch2_array = len(peaks_time_ch2.values)

print('Index peaks ch2', index_peaks_ch2)
print('Quant peaks ch2', peaks_time_ch2_array)
print('Time peaks ch2', peaks_time_ch2)

# Plot
plt.scatter(zdf_ch2.index, sortdf_ch1, c='g')
plt.scatter(zdf_ch1.index, sortdf_ch2, c='r')
plt.plot(zdf_ch1.index,df_ch1_norm, label = 'CH1_norm')
plt.plot(zdf_ch1.index,df_ch2_norm, label = 'CH2_norm')
plt.show()

# Find interval fitting ch1
int_fit_ch1 = []
int_fit_ch2 = []
center_values_ch1 = []
center_values_ch2 = []

for i in range(1, len(index_peaks_ch1) - 1):
	dif_peaks_ch1 = ((index_peaks_ch1[i + 1] - index_peaks_ch1[i]))
	dif_peaks_ch1_del = round(dif_peaks_ch1 / 2)

	s = df_ch1_norm[index_peaks_ch1[i] - dif_peaks_ch1_del : index_peaks_ch1[i] + dif_peaks_ch1_del ]
	x = s.index * ticks

	y_fit = s.values.flatten()
	x_fit = x.values.flatten()

	# choise model
	model = model_func

	# construction fit
	params = model.guess(y_fit, x=x_fit)
	result = model.fit(y_fit, params, x=x_fit)

	# report fit
	print(result.fit_report())

	# plot fit
	#result.plot_fit()
	#plt.show()

	# find center
	center_values_ch1.append(result.params['center'].value)
	print(i)
	int_fit_ch1.append(dif_peaks_ch1)


for i in range(1, len(index_peaks_ch2) - 1):
	dif_peaks_ch2 = ((index_peaks_ch2[i + 1] - index_peaks_ch2[i]))
	dif_peaks_ch2_del = round(dif_peaks_ch2 / 2)

	s = df_ch2_norm[index_peaks_ch2[i] - dif_peaks_ch2_del : index_peaks_ch2[i] + dif_peaks_ch2_del ]
	x = s.index * ticks

	y_fit = s.values.flatten()
	x_fit = x.values.flatten()

	#choise model
	model = model_func

	# construction fit
	params = model.guess(y_fit, x=x_fit)
	result = model.fit(y_fit, params, x=x_fit)

	# report fit
	print(result.fit_report())

	# plot fit
	#result.plot_fit()
	#plt.show()

	# find center
	center_values_ch2.append(result.params['center'].value)
	print(i)
	int_fit_ch1.append(dif_peaks_ch2)

print('Модель функции: ', model_func)
print('Центры пиков функции ch1: ',center_values_ch1)
print('Центры пиков функции ch2: ',center_values_ch2)

#calc dif times and freq
dif = []
freq = []

for i in range(0, len(center_values_ch1)):
	dif.append((center_values_ch1[i] - center_values_ch2[i]))

for i in range(0, len(center_values_ch1)):
	freq.append((center_values_ch1[i] - center_values_ch2[i]) * velocity_sweep)

print('Разность времен: ', dif)
print('Разность частот: ', freq)
avg_freq = np.mean(freq)
print("Сдвиг частот ср.: ", avg_freq)
noise_freq = np.std(freq)
print("Шум частот : ", noise_freq )
rotation = (avg_freq / 305.43721) - (-3253.7217/305.43721)
print("Угловая скорость : ", rotation )
rotation_noise = noise_freq / 305.43721
print("Шум угловой скорости : ", rotation_noise)
#plt.scatter(peaks_time_ch1, freq, label='freqdif')
