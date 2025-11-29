# Podsumowanie
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


