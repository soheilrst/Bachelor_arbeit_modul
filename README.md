# Entity-Matching-System für Firmen- und Straßennamen
Dieses Repository stellt ein Python-Paket bereit, das für das Matching von Firmen- und Straßennamen auf Basis feinabgestimmter 
DistilBERT-Modelle entwickelt wurde. Das System wurde entwickelt, um reale Herausforderungen wie Variationen in der Schreibweise,
Abkürzungen (z. B. "GmbH" vs. "Gesellschaft mit beschränkter Haftung", "Str." vs. "Straße")
sowie die Ähnlichkeit von Hausnummern zu bewältigen. Es beinhaltet zudem fortschrittliche
Blocking- und Indexing-Methoden, um den Matching-Prozess zu optimieren.


## Präsentation
Die vollständige Projektpräsentation ist hier zu finden:

[für mehr Details, Hier klicken](Bachelor_arbeit_v4_repo.pdf)


## Module
- blocker
- fine_tuning
- feature_extraction
- housenumber
- matcher


## Installation
Das Paket kann wie folgt installiert werden:

```py 
pip install  git+https://github.com/soheilrst/Bachelor_arbeit_modul.git
imoprt Bachelor_arbeit_modul as bam
````


## Beispiel-Notebook

Das Notebook [`Test.ipynb`](Test.ipynb) enthält:
1. Training der Modelle.
2. Anwendung der Matcher-Funktion.
3. Ergebnisse 


