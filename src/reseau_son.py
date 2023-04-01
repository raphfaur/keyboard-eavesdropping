from threading import Thread
import threading
import numpy as np
import random 
from matplotlib import pyplot as plt
import time
import click
import testeur
import sys


outs = []
base_ordonned = []

PATH =  "base.txt"

nombre_neurone_sortie = 0
nombre_neurone_entree = 0
neurones_par_couches = 0
couches = 0
final_base = []
max_freq = 0

"""Taille du réseau modifiable"""
neurones_par_couches = 10
couches = 2


def create_data_base(path):
    global nombre_neurone_entree, nombre_neurone_sortie, couches, neurones_par_couches, final_base, base, max_freq
    error = []
    """Transformation de la base de donnée .txt en une liste de couple (entrée, touche attendue)"""
    with open(PATH, "r") as base_file :
        file = base_file.readlines()
        print("Analysing Data Base, found {} several data".format(len(file)))
        with click.progressbar(file, label="Working on Data Base") as bar :
            for i,line in enumerate(bar) :
                data = line.split("|")
                content = data[1].split(' ')
                content.remove('')
                final_data = [float(x.strip("\n")) for x in content]
                inputs_n = len(final_data)
                final_data = [x for x in final_data]
                if max_freq < max(final_data):
                    max_freq = max(final_data)
                if len(final_data) < 1 :
                    error.append((i, data[0], 10-len(final_data)))
                else :
                    if not data[0] in outs :
                        outs.append(data[0])
                        base_ordonned.append([])
                    index = outs.index(data[0])
                    base_ordonned[index].append(final_data)
    
    with click.progressbar(outs, label = "Formatting Data Base") as bar:
        for number, out_value in enumerate(bar):
            for input in base_ordonned[number]:
                final_base.append(([x/max_freq for x in input], [0]*number + [1] + (len(outs)-number -1)* [0]))
    
    """Adaptation de la couche d'entrée à la taille des entrées de la base"""
    nombre_neurone_sortie = len(outs)
    nombre_neurone_entree = inputs_n

    return(error)
    
def graph(outs, inputs_n, layer_n, neuron_per_layer):
    """Affiche une représentation grapique du réseau, pour en observer la structure"""
    plt.figure()
    plt.title("Neural Network Overview")
    for i in range(inputs_n) :
        plt.plot(1,(inputs_n-1)/2-i,linestyle = '',marker ="o",ms = 20,color='grey',label="Peak n°" + str(i))
        plt.text(1,(inputs_n-1)/2-i, "Peak n°" + str(i) ,horizontalalignment = 'center', verticalalignment = 'center')
    for x in range(layer_n):
        for y in range(neuron_per_layer):
            plt.plot(x+2, (neuron_per_layer-1)/2 - y ,linestyle = '',marker ="o",ms = 20,color='grey')
    for y,out in enumerate(outs):
        plt.plot(layer_n + 2, (len(outs)-1)/2-y,linestyle = '',marker ="o",ms = 20,color='grey', label= out)
        plt.text(layer_n + 2, (len(outs)-1)/2-y, out ,horizontalalignment = 'center', verticalalignment = 'center')
    plt.axis("off")
    plt.show()

def sigmoid(x):
    return (1/(1+np.exp(-x)))

def sigmoid_prime(x):
    return (sigmoid(x)*(1-sigmoid(x)))

def creation_poids(inf,sup):
    """Création du tableau de poids aléatoires et de biais"""
    global poids, biais
    biais = [[0]*neurones_par_couches]*couches

    poids = []
    poids.append([])
    for a in range(nombre_neurone_entree):
        poids[0].append([])
        for i in range(neurones_par_couches):
            poids[0][a].append(random.uniform(inf,sup))

    if couches > 1:
        for a in range(couches-1):
            poids.append([])
            for i in range(neurones_par_couches) :
                poids[a+1].append([])
                for k in range(neurones_par_couches) :
                    poids[a+1][i].append(random.uniform(inf,sup))
    poids.append([])
    for a in range(neurones_par_couches):
        poids[-1].append([])
        for i in range(nombre_neurone_sortie):
            poids[-1][a].append(random.uniform(inf,sup))

entree = []
etapes = []

def propagation(entree):
    """Propagation de -entree- dans le réseau, ajoute au tableau -etapes- le vecteur correspondant aux données 
    en transit à chaque couche"""

    etapes.append(entree)
    X = []

    """Propagation en entree"""
    for x in range(neurones_par_couches) :
        valeur = 0
        for input in range(nombre_neurone_entree):
            valeur += etapes[-1][input]*poids[0][input][x]
        X.append(valeur)
    etapes.append(X[:])
    X=[]

    """Propagation dans les couches"""
    for couche in range(couches-1):
        for x in range(neurones_par_couches):
            valeur = 0
            for input in range(neurones_par_couches):
                valeur+= sigmoid(etapes[-1][input]+biais[couche+1][input])*poids[couche+1][input][x]
            X.append(valeur)
        etapes.append(X[:])
        X=[]

    """Propagation en sortie"""
    for x in range(nombre_neurone_sortie):
        valeur = 0
        for input in range(neurones_par_couches):
            valeur+= sigmoid(etapes[-1][input]+biais[-1][input])*poids[-1][input][x]
        X.append(valeur)
    etapes.append(X[:])
    etapes.append([sigmoid(x) for x in X])

pas = 3
erreur = []

def error(attente):
    """Calcul de l'erreur en sortie"""
    erreur.append([])
    for i in range(nombre_neurone_sortie):
        erreur[0].append(sigmoid_prime(etapes[-2][i])*(etapes[-1][i] - attente[i]))

def retropropagation():
    """Retropropagation dans le réseau et calcul des erreurs"""
    for i in range(couches):
        erreur.append([])
        for x in range(neurones_par_couches):
            erreur[-1].append(sigmoid_prime(etapes[-3-i][x]+biais[-i-1][x])*sum([a*b for a,b in zip(erreur[i],poids[-i-1][x])]))
    
    """erreur.append([])
    for x in range(nombre_neurone_entree) :
        erreur[-1].append(sigmoid_prime(etapes[-3-i][x])*sum([a*b for a,b in zip(erreur[-2],poids[-i-1][x])]))"""

def maj_poids():
    """Mise à jour des tableaux -poids- et -biais- à partir du tableau -erreur-"""
    for i, neurone  in enumerate(poids[0]):
        for j,value in enumerate(neurone):
            neurone[j] -= pas*erreur[-1][j]*etapes[0][i]

    for c,couche in enumerate(poids[1:]):
        for j, neurone in enumerate(couche):
            for k, value in enumerate(neurone):
                neurone[k] -= pas*erreur[-2-c][k]*sigmoid(etapes[c+1][j]+biais[c][j])
    
    for c,couche in enumerate(biais):
        for i,neurone in enumerate(couche):
            couche[i] -= pas*erreur[-1-c][i]
        
stats = [0]
epsilon = 0.01
nombre_apprentissages = 5000

def start(base):
    """Demarre -nombre_apprentissages- cycle d'apprentissage : propagation / rétropropagation / mise à jour des poids"""
    global etapes, erreur, X, stats
    start = time.perf_counter()
    print("Réalisation de {} apprentissages".format(nombre_apprentissages))
    for i in range(nombre_apprentissages):
        propagation(base[i%len(base)][0])
        error(base[i%len(base)][1])
        stats.append(sum([(s-c)**2 for (s,c) in zip(etapes[-1], base[i%len(base)][1])]))
        longueur = int(30*i/nombre_apprentissages)
        sys.stdout.write("      [ " + longueur*"#" + (30-longueur)*" " + " ]  " + str(i) + "  " + "{:.4f}".format(stats[-1]) + "\r")
        retropropagation()
        maj_poids()
        erreur=[]
        etapes=[]
        X=[]
        
    print("Dernier test : ", print(testeur.sortie(base[-1][0],poids, biais, nombre_neurone_sortie, couches, neurones_par_couches)) ) 
    print("---------------- \n")
    print("Poids : ", poids)
    print("---------------- \n")
    print("Fini en ", time.perf_counter() - start, " s")
    print("-----------------")
    print(couches, neurones_par_couches, nombre_neurone_sortie, biais)
 
started = True

"""Création de l'interface de communication avec le réseau de neurone"""
while started :
    command = input("\n >>>")

    if command =="start":
        """Lance un thread pour démarrer une série d'apprentissage"""
        threading.Thread(target = start, name= "Learning", args=(final_base,)).start()

    if command == 'create':
        """Crée le réseau à partir d'une base préalablement chargée"""
        creation_poids(-1,1)
        print("Done")

    if "base" in command :
        """Charge la base située à l'emplacement -PATH-"""
        errors = create_data_base("")
        print("Errors line(s) ", [x[0]+1 for x in errors])
    
    if command == "graph":
        """Représentation graphique du réseau"""
        graph(outs, nombre_neurone_entree, couches, neurones_par_couches)
    
    if command.find("test") != -1:
        """Permet de tester le réseau avec l'entrée -base[i]- """
        n = command.split("test")[1]
        try :
            print(final_base[int(n)][1])
            print(testeur.sortie(final_base[int(n)][0],poids, biais, nombre_neurone_sortie, couches, neurones_par_couches))
        except :
            pass
    
    if command == "save":
        """Création du fichier de configuration du réseau pour la réutiliser pendant"""
        with open('config.txt', 'w') as config :
            config.write(str(poids)+"\n")
            config.write(str(biais)+"\n")
            config.write(str(couches)+"\n")
            config.write(str(neurones_par_couches)+"\n")
            config.write(str(nombre_neurone_sortie)+"\n")
            config.write(str(max_freq)+"\n")
            config.write(str(outs))
    
    if command == "stat":
        """Affiche l'évolution de l'erreur au cours du temps"""
        plt.scatter(range(len(stats)), stats, marker = "+")
        plt.show()


