# Outil de test d'une attaque de type "keyboard-eavesdropping"

## Préambule
Avant tout, je tiens à préciser que cet outil a été développé à des fins de test et d'étude. Les auteurs de cet outil déclinent toute responsabilité en cas de quelconque utilisation malveillante des programmes.

## Utilisation
- ```base_apprentissage.py``` permet de créer une base d'apprentissage ```base.txt``` à partir d'une entrée audio.
- ```reseau_son.py``` Lance une CLI permettant de créer un réseau de neurone à partir d'un fichie ```base.txt``` puis d'entraîner et visualiser le réseau.  
La commande ```save``` permet notamment une fois les différentes opérations réaliséees de sauvegarder l'état du réseau dans un fichier ```confix.txt```
- ```analyseur.py``` permet de lancer une analyse sonore pendant un temps donné et de sortir les séquences saisies au clavier les plus probables.

## Version
Cette version a été développée sous Windows et ne supporte pas nécessairement MacOs ou Linux. 
