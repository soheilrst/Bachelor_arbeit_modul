# Entity-Matching-System für Firmen- und Straßennamen
Dieses Repository enthält ein Python-Paket, das mithilfe feinabgestimmter DistilBERT-Modelle für das Matching 
von Firmen- und Straßennamen entwickelt wurde.Es adressiert Herausforderungen wie Schreibvariationen, Abkürzungen
(z. B. „GmbH“ vs. „Gesellschaft mit beschränkter Haftung“, „Str.“ vs. „Straße“) und die Ähnlichkeit
von Hausnummern. Ergänzt wird es durch fortschrittliche Blocking- und Indexing-Methoden, 
um den Matching-Prozess effizienter zu gestalten.


## Präsentation
Die vollständige Projektpräsentation ist hier zu finden:

[für mehr Details, Hier klicken](Bachelor_arbeit_v4_repo.pptx)


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


