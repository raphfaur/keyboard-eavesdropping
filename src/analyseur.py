import threading
import testeur
import analyse_son
import pyaudio as pyd
import numpy
import threading
import time
import msvcrt
from termcolor import colored

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

"""Chargement de la configuration du réseau à partir du fichier -config.txt- """
with open('config.txt', 'r') as config:
    poids = eval(config.readline())
    biais = eval(config.readline())
    couches = eval(config.readline())
    neurone_par_couche = eval(config.readline())
    neurone_sortie = eval(config.readline())
    max_freq = eval(config.readline())
    outs = eval(config.readline())

results = []
keys = []

def log():
    global keys
    while True :
        key = msvcrt.getwch()
        print(key)
        keys.append(key)
            
threading.Thread(target = log, name = "keylog").start()

on_record = False

stream = p.open(RATE, channels = 1, input = True, format = pyd.paInt16, frames_per_buffer = CHUNK, output = True)

resultats = []
        
def analyse(frames):
    """Analyse la séquence et retourne pour chaque augmentation sonore une analyse du son pour déterminer la touche"""
    x = 0
    while x < len(frames):
        if frames[x] > 1600 :
            touche,_,_  = analyse_son.pic(0.04,frames[x-int(0.1*RATE): x+int(0.4*RATE)], RATE, n=0)
            xf,yf = analyse_son.analyse_brute(data = touche, rate = RATE)
            _, yf = analyse_son.discret(xf, yf, 30,0,5000)
            result = testeur.sortie(yf, poids, biais, neurone_sortie, couches, neurone_par_couche)
            results.append(outs[result.index(max(result))])
            fusion = [(x,y) for x,y in zip(outs, result)]
            fusion.sort(key = lambda x:x[1], reverse=True)
            resultats.append(fusion)
            x+=int(0.2*RATE)
        x+= 1
    """Caractéristiques du test"""
    a = [key == result[0][0] for (key, result) in zip(keys, resultats)]
    rate = a.count(True)/len(a)
    print("Taux de ressembance : " + str(rate)+ "\n") 

    """Représentation graphique des résultats dans la console"""
    dic = {True : 'green', False : 'red'}
    combi = combinaison(resultats)
    combi_possible = [x for x in combi if x[1] > 0.01]
    combi_possible.sort( key = lambda x:x[1], reverse=True)
    for mot in combi_possible :
        a = [key == lettre for (key, lettre) in zip(keys, mot[0])]
        rate = a.count(True)/len(a)
        lettres = [colored(lettre, dic[key == lettre]) for (key, lettre) in zip(keys, mot[0])]
        print(''.join(lettres), mot[1], "Taux de ressembance : " + str(rate))

       

recap=[]

def combinaison(liste):
    """Effectue toutes les combinaisons de lettres possibles en donnant leur probabilité"""
    if len(liste) == 2:
        l = []
        
        for X in liste[0]:
            for Y in liste[1]:
                if X[1]*Y[1] >= 0.05 :
                    l.append((X[0]+Y[0], X[1]*Y[1]))
        return l
    return combinaison([liste[0],combinaison(liste[1:])])


def record():
    """Lance un enregistrement pendant 10s"""
    frames = numpy.array([])
    on_record = True
    start_time = time.perf_counter()
    while on_record and time.perf_counter() - start_time < 10 :
        data = stream.read(CHUNK)
        stream.write(data)
        frames = numpy.concatenate((frames,numpy.frombuffer(data, numpy.int16)))
    """Analyse de l'enregistrement"""
    analyse(frames)

record()

def stop_record():
    on_record = False       
