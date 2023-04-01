from scipy.io import wavfile as wf
from scipy.fftpack import fft, fftfreq
import numpy as np
from scipy import signal

def analyse(data=None, rate=None):
    """Retourne la transformée de Fourier de -data-"""
    """FFT de l'échantillon"""
    yf = fft(data)
    xf=fftfreq(data.size, 1/rate)
    yf = np.absolute(yf)
    """Normalisation"""
    yf = yf/250000
    return(xf, yf)

def analyse_brute(data, rate):
    yf = fft(data)
    xf=fftfreq(len(data), 1/rate)
    yf = np.absolute(yf)
    yf = yf/250000
    return(xf, yf)

def find_peaks(xf, yf, number):
    peaks, _= signal.find_peaks(yf, prominence = 0.05, distance = 1)  
    peaks = peaks[:number]
    return([xf[peak] for peak in peaks], peaks)

def discret(xf, yf, intervalle, min, max) :
    """Récupère une amplitude du spectre tous les -intervalle- Hz sur la plage [min, max]"""
    intervalle_freq = xf[1] - xf[0]
    x = []
    y = []
    distance_discrete = int(intervalle/intervalle_freq)
    for abscisse in range(int(min/intervalle_freq) , int(max/intervalle_freq) , distance_discrete):
        y.append(yf[abscisse])
        x.append(xf[abscisse])
    return x,y


def pic(width, son, RATE, n):
    """Retourne le n-eme pic de -son- sur une largeur width"""
    peaks, _ = signal.find_peaks(son, prominence = 1000, distance = 2000)
    pic = son[peaks[n]-int(0.01*RATE): peaks[n] + int((width-0.01)*RATE)]
    return pic, peaks[n], peaks

