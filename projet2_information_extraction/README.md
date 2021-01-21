# Fichiers
L'exploration pour les acronymes est dans le fichier ```code/acronymes.ipynb```. 
L'exploration pour les entités nommées est dans le fichier```code/entites_nommees.ipynb```.  
L'exploration pour les oie est dans le fichier ```code/oie.ipynb```. 
Le rapport est contenue dans le fichier ```rapport.pdf```. 
Les extraction faites sur l'ensemble test sont dans le dossier ```extractions_test/``` qui contient un fichier d'extactions par article.

Chaque fichier ```nom_du_fichier_json.out``` est sous la forme 

- [- Acronymes]
[{liste de tous les acronymes trouvés}]

- [- Entités nommées]
[{liste de toutes les entités nommées trouvées}]

- [-OIE]
[{liste des extractions trouvées}]


# Extraction

**Pour la tâche des acronymes**
Le système a été en mesure de trouver les deux seuls acronymes présents au travers des 5 articles du corpus test: MBAM (Musée des beaux-arts de Montréal) et SQ (Sûreté du Québec).
Ce nombre peu élevé d'acronymes est attendu due à la rareté de leur utilisation dans un corpus ordinaire (tel qu'expliquer dans le rapport). 
Cependant, la qualité de ces 2 extractions est parfaite.

**Pour la tâche des entités nommées**
Les entités nommées ont été extraite en utilisant la correction des productions de Spacy. La qualité de ces extractions au vu de la précision dans les productions est parfaite.
 
**Pour la tâche hyponymie**
Les articles de tests contiennent 121 phrases. Nous avons extrapolé que l'extracteur devrait retiré 2.33% extractions indépendantes pertinentes. 
Ce qui correspond à 2 extractions pertinentes. C'est le résultat qu'on obtient sur les 13 extractions totales.
Les deux extractions contenant de l'information pertinentes indépendantes sont:

*Ce Jardin sculptures est endroit* et *La directrice CIA Gina est rendue* (Donne le prénom de la directrice de la CIA)