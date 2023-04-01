import tkinter as tk
from tkinter.filedialog import askdirectory
from tkinter import StringVar
from datetime import datetime as dt
import pyaudio as pyd
import numpy as np
import wave
import threading
import time
import tkinter.font as font
import matplotlib.pyplot as plt
from scipy.io import wavfile as wf
from scipy import signal
from scipy.fftpack import fftfreq,fft
from scipy.interpolate import Rbf
import os
import sys

##sys.path.append('##Add path to folder containing analyse_son.py if needed')

import analyse_son

CHUNK = 4096
RATE = 44100

duration_value = 0.04
delay_value = 0.23
frames = []

on_record = False

done = False

time_click = []

p = pyd.PyAudio()

stream = p.open(RATE, channels = 1, input = True, format = pyd.paInt16, frames_per_buffer = CHUNK, output = True)

fen = tk.Tk()
fen.title("Création d'une base de donnée")
fen.geometry("500x500")

ft = font.Font(size = 32)

keypressed = StringVar()

def define_directory() :
    directory_string.set(askdirectory())

"""Fonction d'enregistrement d'un son"""
def record_func():
    global on_record, start_time, done
    done = False
    start_time = time.perf_counter()
    on_record = True
    while on_record :
        data = stream.read(CHUNK)
        stream.write(data, CHUNK)
        frames.append(np.frombuffer(data, np.int16))
    done = True

stat_enabled = False
    

record = threading.Thread(target=record_func)
record.name = "Record"

def key_pressed(evt):
    global time_click, start_time
    """ Ajout d'un (time, key) à la liste time_click"""
    time_click.append((time.perf_counter()-start_time, evt.keysym))
    keypressed.set(str(evt.keysym))
    print((time.perf_counter()-start_time, evt.keysym))
    

fen.bind("<KeyPress>", key_pressed)

def start():
    global start_time
    record.start()

def stop():
    global on_record, done
    on_record = False
    """Lancement de l'analyse une fois le bouton "stop" pressé"""
    fen.after(100, lambda : read_part(time_click, duration_value , delay_value , directory_string.get()) )
    
"""Parcours du fichier audio et découpage des séquences d'intérêt"""
def read_part(time_list, duration, delay, path):
    global record
    sound = [x for X in frames for x in X]
    for key_time in time_list :
        extract = sound[int((key_time[0]+delay)*RATE) : int((key_time[0] + delay + duration)*RATE) ]                
        """Analyse du son par la fonction analyse_son.analyse"""
        xf,yf = analyse_son.analyse(np.array(extract),RATE)
        if stat_enabled :
            plt.plot(xf, yf)
            plt.xlim(0,6000)
            plt.clf()
            plt.plot(extract)
        """Mise en forme des données avec la fonction analyse_data"""
        analyse_data(xf, yf, key_time[1])     
    
    """Représenation graphique du son et de tous les segments prélevés"""
    plt.clf()
    plt.plot(sound)
    for a in time_list :
        plt.axvline(x = (a[0]+delay)*RATE, c="green")
        plt.axvline(x = (a[0] + delay+duration)*RATE,c = "red")
    plt.show()


def analyse_data(xf,yf, key):
    """Transformation des données brutes de la FFT en une base de données utilisable"""
    x_discret, y_discret = analyse_son.discret(xf, yf,30 , 0, 5000)

    if stat_enabled :
        plt.figure()
        plt.title(key)
        plt.plot(xf, yf)
        plt.scatter(np.array(x_discret),y_discret, c = "red", marker = '+')
        plt.xlim(0,6000)
        plt.show()

    """Ecriture de la donnée dans la base de donnée selon une norme définie"""
    line = ""
    for amplitude in y_discret:
        line += " " + str(amplitude)
    with open(directory_string.get()+"/base.txt", "a") as base_apprentissage :
        base_apprentissage.write(key + "|")
        base_apprentissage.write(line)
        base_apprentissage.write("\n")
    


"""Création de l'interface graphique"""    

directory_string = StringVar()

def_directory = tk.Button(fen, command = define_directory, text = "Définir le chemin d'enregistrement")
def_directory.pack(pady = 5)

directory = tk.Label(fen, textvariable = directory_string)
directory.pack(pady = 5)

text = tk.Entry(fen)
text.pack(pady = 5)

start_button = tk.Button(fen, command = start, text = "Créer la base d'apprentissage")
start_button.pack(pady = 5)

stop_button = tk.Button(fen, command = stop, text = "Stop")
stop_button.pack(pady = 5)

letter = tk.Label(fen, textvariable = keypressed, font = ft)
letter.pack(pady = 5)

fen.mainloop()

stream.stop_stream()
stream.close()
p.terminate()