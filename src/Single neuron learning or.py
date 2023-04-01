import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import math
from matplotlib import cm

"""w2 = random.random()*20-10
w1 = random.random()*20-10"""
W2_ = -5
W1_ = -2.5

"""Mettre les poids à -10 = minimum local"""
w3=-0.5
b = 1
pas = 1
dw1 = 0
dw2 = 0
db = 0
dw3 = 0


def sigmoid(x) :
    return (1/(1+np.exp(-x)))

def result(w1,w2,biais,x,y) :
    """Retourne la sortie du neurone"""
    return (sigmoid(x*w1 + y*w2 +biais ))
    """le biais fait descendre le plan horizontal"""

def erreur_quadratique(result, cible):
    """Retourne l'erreur quadratique associé à un résultat et une valeur attendue"""
    return(1/2*(result - cible)**2)

def f(w1,w2,biais, a,b,c):
    """Retourne l'erreur quadratique associée à l'entré a affectée du poids w1 , de l'entrée b affectée du poids w2 et du résultat attendu c"""
    return(erreur_quadratique(result(w1,w2,biais, a,b), c))


def diff(result, cible, x, y):
    """Calcule dErreur/dw2"""
    global dw1, dw2, db
    dw1 = (result-cible)*result * (1-result) * x
    dw2 = (result-cible)*result * (1-result) * y
    """dw3 = (result-cible)*result * (1-result) * z"""
    db = (result-cible)*result * (1-result)
    
    
def retro():
    """Rectifie les poids à partir des pentes calculées précédemment"""
    global W2_,W1_, b, w3
    W2_ -= dw2 * pas
    W1_ -= dw1 * pas
    b -= db*pas
    """w3 -= dw3*pas"""

A = [(0,0,1,0),(1,1,1,1),(1,0,1,0),(0,1,1,1)]
B = [(0,0,0),(1,1,1),(0,1,1),(1,0,1)]

E = []

fig = plt.figure()
axe = Axes3D(fig)

W1 = np.linspace(-10,15,20)
W2 = np.linspace(-10,15,20)
vW1, vW2 = np.meshgrid(W1,W2)

"""On garde une trace de tous les biais/poids calculés pour tracer le chemin après"""
W1trace =[]
W2trace = []
btrace = []

"""Répartition logarithmique des points"""
logg = np.floor(np.logspace(0,5, base = 10, num = 30))


"""On démarre la descente de gradient"""
for i in range (10000):
        if i in logg or i == 0 : 
            W1trace.append(W1_)
            W2trace.append(W2_)
            btrace.append(b)
        a = B[i%4]
        resultat = result(W1_,W2_,b,a[0],a[1])
        cible = a[2]
        erreur = erreur_quadratique(resultat, cible)
        E.append(resultat - cible)
        diff(resultat, cible, a[0], a[1])
        retro()
        

"""Création du graph 3D"""

plt.clf()
fig = plt.figure()
fig.suptitle("Descente de gradient")
axe = Axes3D(fig)

axe.set_xlabel("W1")
axe.set_ylabel("W2")
axe.set_zlabel("Erreur quadratique")
axe.grid(False)

"""Tracé des surface et des chemins à biais fixé (car le biais bouge en même temps...)"""

for i,a in enumerate(B[1:4]) :
    vz = f(vW1, vW2,b ,a[0], a[1], a[2])
    couleur = np.array(['coral', 'teal', 'olive'])
    axe.plot_surface(vW1, vW2, vz, alpha = .2, color = couleur[i])
    for X in zip(W1trace, W2trace, btrace) :
        axe.scatter(X[0],X[1],f(X[0],X[1],b,a[0], a[1], a[2]), s = 20 , marker = "x", c= couleur[i])


plt.show()
plt.savefig("Descente de gradient", transparent = True)

