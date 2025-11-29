#  Analiza jakości wina oraz klasyfikacja płci kotów na podstawie cech

### Autorzy

Wiktor Rapacz, Hanna Paczoska

---
## Opis projektu

Celem projektu jest porównanie dwóch klasycznych algorytmów klasyfikacji:
Drzewa decyzyjnego (Decision Tree Classifier) oraz
Maszyny wektorów nośnych (Support Vector Machine – SVM)
na dwóch niezależnych zbiorach danych:

Wine Quality (wino czerwone)

zadanie: klasyfikacja jakości wina (klasy 3–8)

Cats Dataset (Kaggle)

link: https://www.kaggle.com/datasets/waqi786/cats-dataset

zadanie: klasyfikacja płci kota (Male / Female)

Projekt obejmuje:
- przygotowanie i czyszczenie danych,
- wizualizacje rozkładów cech (histogramy, heatmapy korelacji),
- trening modeli Decision Tree i SVM,
- porównanie wyników z wykorzystaniem różnych kernel function,
- przykładowe predykcje dla danych wejściowych,
- wnioski dotyczące wpływu dobranych parametrów na jakość klasyfikacji.
---
## Instrukcja przygotowania środowiska
1. Utworzenie środowiska wirtualnego
python -m venv .venv

2. Aktywacja środowiska
- Windows:
.\.venv\Scripts\activate
- Linux / macOS:
source .venv/bin/activate

3. Instalacja wymaganych bibliotek
pip install numpy pandas scikit-learn matplotlib seaborn

4. Uruchomienie programu
python main.py


### Program wygeneruje:

- wykresy,
- metryki jakości,
- przykładowe predykcje,
- porównanie kernel function.

---
## Zbiory danych
1. Wine Quality Dataset

Plik użyty w projekcie: resources/winequality-red.csv

Dane zawierają parametry fizykochemiczne wina:
- fixed acidity
- volatile acidity
- citric acid
- residual sugar
- chlorides
- free sulfur dioxide
- total sulfur dioxide
- density
- pH
- sulphates
- alcohol
- quality (etykieta do przewidywania)

2. Cats Dataset

Plik:
resources/cats_dataset.csv
Pobrany z:
https://www.kaggle.com/datasets/waqi786/cats-dataset?resource=download

Atrybuty:
- Breed
- Age (Years)
- Weight (kg)
- Color
- Gender (etykieta do przewidywania)

---
## Metody klasyfikacji
### Decision Tree Classifier

Model drzewa decyzyjnego buduje hierarchiczną strukturę („drzewo”), w której na podstawie wartości cech dokonywana jest decyzja o przydzieleniu próbki do danej klasy.

Zalety:
- łatwa interpretacja
- szybki trening
- działa dobrze dla danych nieliniowych
- radzi sobie z danymi kategorycznymi po zakodowaniu

### Support Vector Machine (SVM)

SVM stara się znaleźć hiper-płaszczyznę maksymalnie oddzielającą klasy.
W zależności od użytego kernela, może tworzyć zarówno liniowe, jak i bardzo złożone granice decyzyjne.

### W projekcie wykorzystano kernelle:
- linear
- rbf (z różnymi wartościami C i gamma)
- poly
- sigmoid
---
##  Wizualizacje

W projekcie wykorzystano:

Dla wina:
- histogramy rozkładów wszystkich cech (12 wykresów)
- heatmapę korelacji między cechami

Dla kotów:
- histogramy wieku oraz wagi
- scatter plot: Age vs Weight, z oznaczeniem płci

###  Predykcja przykładowych danych

Program demonstruje działanie modeli poprzez klasyfikację sztucznie wygenerowanych danych wejściowych:

dwa przykładowe wina → oba modele przewidziały klasę 6

dwa przykładowe koty → modele podały płeć (Male / Female)

To pokazuje prawidłowe działanie modeli w praktyce.

---

## Wpływ rodzaju kernel function na wyniki klasyfikacji

W SVM wybór kernel function ma kluczowe znaczenie dla jakości klasyfikacji.
Przetestowano kernelle:
- linear
- rbf (różne C i gamma)
- poly
- sigmoid

1. Kernel liniowy (linear) działa najlepiej dla danych separowalnych liniowo.
W naszych danych wyniki były przeciętne, co oznacza, że rozkład klas jest nieliniowy.

2. Kernel RBF - najbardziej uniwersalny — potrafi modelować nieliniowe zależności.
Dał najlepsze wyniki podczas klasyfikacji wina, szczególnie przy zwiększeniu:

    C → kara za błędną klasyfikację
          
    gamma → wpływ pojedynczej próbki

    RBF z C=10, gamma=0.1 uzyskał najwyższe accuracy.
3. Kernel polynomial - daje dobre wyniki, ale zazwyczaj jest gorszy niż optymalny RBF.
W danych kotów był minimalnie najlepszy (ok. 0.513).

4. Kernel sigmoid - w SVM zwykle działa najsłabiej.
Potwierdziło się to zarówno w przypadku wina, jak i kotów.

###  Wnioski:

W danych o winie istnieją nieliniowe wzorce, więc najlepsze wyniki daje kernel RBF z odpowiednimi parametrami.

W danych o kotach cechy są słabo związane z płcią, więc modele osiągają wyniki bliskie losowym (~0.50), niezależnie od kernela.

Przydatność kernela zależy od struktury danych — najlepsze modele to te, które najlepiej odzwierciedlają ukryte zależności
