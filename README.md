Ce projet, intitulé "Deep Siamese Neural Network for Contour 2D Registration", s'inspire de la méthode analytique ACMA (Affine Curve Matching Algorithm) présentée dans l'article "An Efficient 2D Curve Matching Algorithm under Affine Transformations" par Elghoul, S. et Ghorbel, F. (DOI: 10.5220/0006719504740480). L'ACMA a réussi à relever le défi de la reconnaissance de formes partiellement occultées sous des transformations affines, en se basant sur la longueur d'arc affine normalisée.

Le projet vise à créer un algorithme basé sur des réseaux de neurones profonds, une boîte noire capable d'estimer avec précision ces transformations affines, similaire aux formules mathématiques utilisées dans l'ACMA, et d'adapter ce modèle pour manipuler des données de contours 2D. Les points clés de cette démarche incluent :

Étape 1 : Création d'une base de données augmentée et adaptée aux réseaux de neurones :

Utilisation des bases de données MPEG-7 et MCD.
Étape 2 : Création d'une architecture capable de lire deux entrées :

Conception d'une structure capable d'assimiler les paires de formes 2D pour l'entraînement et les tests.
Étape 3 : Entraînement et test de la solution proposée :

Mise en œuvre de l'algorithme basé sur le Deep Siamese Neural Network et évaluation de ses performances.
Ce projet s'engage à relever le défi de la reconnaissance précise des contours 2D sous des transformations affines en utilisant une approche basée sur les réseaux de neurones profonds. Les bases de données sélectionnées et la méthodologie d'entraînement et de test contribueront à démontrer l'efficacité de cette approche novatrice.
