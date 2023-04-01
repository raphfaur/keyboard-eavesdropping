from scipy import signal
import pyaudio as pyd
import numpy
import matplotlib.pyplot as plt
import sys
import time
##sys.path.append('##Add path to folder containing analyse_son.py if needed')
import analyse_son

p = pyd.PyAudio()

CHUNK = 1024*4
RATE = 44100
delay = 0
duration = 0.05
NUMBER = 10
max_freq = 0

poids = []
biais = []
couches = 0
neurone_par_couche = 0
neurone_sortie = 0

frames = []

on_record = False

fig = plt.figure()

stream = p.open(RATE, channels = 1, input = True, format = pyd.paInt16, frames_per_buffer = CHUNK, output = True)
               
def analyse(frames):
    """Analyse le segment en particulier pour trouver la zone à analyser"""
    liste_touches = []
    x=0
    while x < len(frames):
        if frames[x] >=2000:
            
            touche = frames[x- int(0.1*RATE): x + int(0.4*RATE)]
            son,t, peaks = analyse_son.pic(0.05, touche, RATE, 0)
            liste_touches.append(son)
            """Affiche tous les pics détectés"""
            i=0
            for peak in peaks :
                if i == 0 :
                    legend = "Pics"
                    i = 1
                else :
                    legend = '_nolegend_'
                plt.axvline(peak, c="orange", label = legend)

            """Affiche le début et la fin du segment"""
            plt.plot(touche)
            plt.axvline(t-int(0.01*RATE), c="green", label = "Début", linestyle = ":")
            plt.axvline(t + int(0.04*RATE), c="red", label = "Fin", linestyle = ':')
            plt.legend(fontsize = 20)
            x+=int(RATE*0.2)
            plt.show()
                   
        x+= 1
       

def record():
    """Lance l'enregistrement pendant 5 secondes"""    
    global on_record,  line, delay, start_time, frames, fig, ay, graph, part
    on_record = True
    start_time = time.perf_counter()
    print("Record started")
    while on_record :
        data = stream.read(CHUNK)
        stream.write(data)
        data_format = numpy.frombuffer(data, numpy.int16)
        frames = numpy.concatenate((frames, data_format))
        if time.perf_counter() - start_time >= 5 :
            on_record = False
    analyse(frames)
    
record()

def stop_record():
    on_record = False