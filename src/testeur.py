import numpy as np


def sigmoid(x):
    return (1/(1+np.exp(-x)))


def sortie(entree, poids, biais, nombre_sortie, couches, neurones_par_couche):
    """Retourne la sortie pour un réseau décrit par la configuration en paramètres et l'entrée -entree-"""
    couches = len(biais)
    X = []
    
    for neurone in range(neurones_par_couche):
        valeur = 0
        for i, input in enumerate(entree) :
            valeur+= input*poids[0][i][neurone]
        X.append(valeur)

    for couche in range(couches - 1) :
        Y = X[:]
        X=[]
        for neurone in range(neurones_par_couche):
            valeur = 0
            for x in range(len(Y)):
                valeur += sigmoid(Y[x] + biais[couche][x])*poids[couche + 1][x][neurone]
            X.append(valeur)
        
    Y = X[:]
    X = []
    for neurone in range(nombre_sortie):
        valeur = 0
        for x in range(len(Y)):
            valeur += sigmoid(Y[x] + biais[-1][x]) * poids[-1][x][neurone]
        X.append(valeur)
    return [sigmoid(x) for x in X]
