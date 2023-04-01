import tkinter as tk
from tkinter import StringVar
import pyaudio as pyd
import numpy as np
import threading
import time
import tkinter.font as font
import matplotlib.pyplot as plt
from scipy import signal
import sys

##sys.path.append('##Enter PATH if needed')
import analyse_son

CHUNK = 4096
RATE = 44100

duration_value = 3
delay_value = 0.4
frames = []

on_record = False

done = False

time_click = []

p = pyd.PyAudio()

stream = p.open(RATE, channels = 1, input = True, format = pyd.paInt16, frames_per_buffer = CHUNK, output = True)

fen = tk.Tk()
fen.geometry("500x500")
fen.title("Analyse de séquences")

ft = font.Font(size = 32)

keypressed = StringVar()


def record_func():
    """Enregistrement d'une séquence"""
    global on_record, start_time, done, frames
    done = False
    start_time = time.perf_counter()
    on_record = True
    frames=[]
    while on_record :
        data = stream.read(CHUNK)
        stream.write(data, CHUNK)
        frames.append(np.frombuffer(data, np.int16))
    done = True


def key_pressed(evt):
    """Ajout du temps au bout duquel une touche est pressée"""
    global time_click, start_time
    try :
        time_click.append((time.perf_counter()-start_time, evt.keysym))
        keypressed.set(str(evt.keysym))
        print((time.perf_counter()-start_time, evt.keysym))
    except :
        pass

fen.bind("<KeyPress>", key_pressed)

def start():   
    """Lance l'enregistrement"""
    record = threading.Thread(target=record_func)
    record.name = "Record"
    record.start()

plt.ion()

def stop():
    global on_record, done, line_1, fig
    on_record = False
    
n=0  

def analyse():
    """Analyse du son, en fonction des paramètres"""
    read_part(time_click[0][0], float(duration_tk.get()),float(delay_tk.get()), float(prominence_tk.get()), float(distance_tk.get()))
        
def read_part(time, duration, delay, prominence, distance):
    global n, fourier, fig, time_draw, ay, begin, end, points_draw
    sound = [x for X in frames for x in X]
    part = sound[int((time+delay)*RATE):int((time+delay+duration)*RATE)]
    """Detection des pics avec paramètre -prominence- et -distance-"""
    peaks, _ = signal.find_peaks(sound, prominence = prominence, distance = distance)
    points = np.zeros(len(sound))
    for peak in peaks :
        points[peak] = sound[peak]
    
    xf, yf = analyse_son.analyse_brute(part, RATE)
    """Représentation des données"""
    if n == 0 :
        fig = plt.figure()
        ay = fig.add_subplot(211)
        ax = fig.add_subplot(212)
        time_draw, = ay.plot(sound)
        points_draw, = ay.plot(points, c = "orange")
        begin = ay.axvline(0, c="green")
        end = ay.axvline(0, c="red")
        fourier, = ax.plot(xf, yf)
        ay.set_xlim(time*RATE, (time + 1)*RATE)
        ax.set_xlim(0,6000)
        n=1

    fourier.set_xdata(xf)
    fourier.set_ydata(yf)
    time_draw.set_ydata(sound)
    points_draw.set_ydata(points)
    begin.set_xdata((time + delay)*RATE)
    end.set_xdata((time + delay+duration)*RATE)
    fig.canvas.draw()
    fig.canvas.flush_events()


start_button = tk.Button(fen, command = start, text = "Start")
start_button.pack()

stop_button = tk.Button(fen, command = stop, text = "Stop")
stop_button.pack()

analyse_button = tk.Button(fen, command = analyse, text = "Analyse")
analyse_button.pack()

delay_txt = tk.Label(fen, text = "Delay", font = ft)
delay_txt.pack()

delay_tk = tk.Entry(fen, text="delay")
delay_tk.pack()

duration_txt = tk.Label(fen, text = "Duration", font = ft)
duration_txt.pack()

duration_tk = tk.Entry(fen, text="duration")
duration_tk.pack()

prominence_txt = tk.Label(fen, text = "Prominence", font = ft)
prominence_txt.pack()

prominence_tk = tk.Entry(fen, text="Prominence")
prominence_tk.pack()

distance_txt = tk.Label(fen, text = "Distance", font = ft)
distance_txt.pack()

distance_tk = tk.Entry(fen, text="Distance")
distance_tk.pack()

fen.mainloop()

stream.stop_stream()
stream.close()
p.terminate()

