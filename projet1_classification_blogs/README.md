# Fichiers
Tout le prétraitement effectué est contenue dans le module ```code/preprocessing.py```. 
L'exploration pour la prédiction du genre est contenue dans le fichier ```code/prediction_genre.ipynb```. 
L'exploration pour la prédiction de l'âge est contenue dans le fichier ```code/prediction_age.ipynb```.  
L'exploration pour la prédiction du signe astrologique est contenue dans le fichier ```code/prediction_signe.ipynb```. 
Le rapport est contenue dans le fichier ```rapport.pdf```. 
Les prédictions faites sur l'ensemble test ```quiz.csv``` est contenue dans le fichier ```best.csv```.

Le fichier ```best.csv``` est sous la forme suivante: auteur, genre, âge, signe. Par exemple: BLIDE4B,female,0,Scorpio

# Entraînement

Chaque tâche est évaluée à l'aide d'un modèle unique. L'ensemble d'entraînement est regroupé par auteur. Le prétraitement effectué consiste à retirer les *stop-words* de la librairie *nltk* et appliqué un *stemming*  avec *SnowballStemmer* de *nltk*. Un plongement de mots est fait à l'aide de *TFDIF* de la librairie *scikit-learn*

**Pour la tâche du genre**
 Un modèle support vector machine avec *SVM* de la librairie *scikit-learn* à été entraîner sur tout l'ensemble d'entraînement.
 
 **Pour la tâche de l'âge**
 Un modèle Naïves Bayes avec *ComplementNB* de la librairie *scikit-learn* à été entraîner sur tout l'ensemble d'entraînement.
 
 **Pour la tâche du signe astrologique**
 Un modèle multilayer perceptron avec *MLPClassifier* de la librairie *scikit-learn* à été entraîner sur tout l'ensemble d'entraînement.
