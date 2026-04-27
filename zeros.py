import numpy as np


def wyznacz_zera_niezalezne(M):
    """
    Wyznacza maksymalny zbiór zer niezależnych w macierzy M.

    Zero niezależne = takie zero, że w jego wierszu i kolumnie
    nie ma innego zera niezależnego.

    Parametry:
        M - macierz numpy, najlepiej już po redukcji

    Zwraca:
        niezalezne - lista krotek (wiersz, kolumna) z zerami niezależnymi
        zalezne - lista krotek (wiersz, kolumna) z zerami zależnymi
    """

    n, m = M.shape

    # Dla każdego wiersza zapisujemy kolumny, w których są zera
    zera_w_wierszach = []
    for i in range(n):
        kolumny_zer = []
        for j in range(m):
            if M[i, j] == 0:
                kolumny_zer.append(j)
        zera_w_wierszach.append(kolumny_zer)

    # match_col[j] = numer wiersza, który ma przypisane zero w kolumnie j
    # -1 oznacza, że kolumna nie jest jeszcze zajęta
    match_col = [-1] * m

    def znajdz_przypisanie(i, odwiedzone_kolumny):
        """
        Próbuje przypisać wiersz i do jednej z kolumn z zerem.
        Jeśli kolumna jest zajęta, próbujemy przepchnąć poprzedni wiersz
        do innej kolumny.
        """
        for j in zera_w_wierszach[i]:
            if odwiedzone_kolumny[j]:
                continue

            odwiedzone_kolumny[j] = True

            if match_col[j] == -1 or znajdz_przypisanie(match_col[j], odwiedzone_kolumny):
                match_col[j] = i
                return True

        return False

    # Szukamy maksymalnego skojarzenia w grafie: wiersze -> kolumny z zerami
    for i in range(n):
        odwiedzone_kolumny = [False] * m
        znajdz_przypisanie(i, odwiedzone_kolumny)

    # Odczytujemy zera niezależne
    niezalezne = []
    for j in range(m):
        if match_col[j] != -1:
            niezalezne.append((match_col[j], j))

    # Pozostałe zera traktujemy jako zależne
    niezalezne_set = set(niezalezne)
    zalezne = []

    for i in range(n):
        for j in range(m):
            if M[i, j] == 0 and (i, j) not in niezalezne_set:
                zalezne.append((i, j))

    # Sortujemy, żeby wynik był czytelny
    niezalezne.sort()
    zalezne.sort()

    return niezalezne, zalezne



def zwieksz_liczbe_zer_niezaleznych(M, wiersze_linie, kolumny_linie, dolne_ograniczenie=0):
    """
    Wykonuje krok zwiększania liczby zer niezależnych w metodzie węgierskiej.

    Parametry:
        M - macierz numpy
        wiersze_linie - lista indeksów wierszy pokrytych liniami
        kolumny_linie - lista indeksów kolumn pokrytych liniami
        dolne_ograniczenie - aktualna wartość dolnego ograniczenia / redukcji

    Działanie:
        1. Znajduje najmniejszy element niepokryty liniami.
        2. Odejmuje go od wszystkich elementów niepokrytych.
        3. Dodaje go do elementów na przecięciach dwóch linii.
        4. Zwraca nową macierz i zaktualizowane dolne ograniczenie.

    Zwraca:
        nowa_M - zmodyfikowana macierz
        min_niepokryty - najmniejszy element niepokryty liniami
        nowe_dolne_ograniczenie - zaktualizowane dolne ograniczenie
    """

    nowa_M = M.copy()

    n, m = nowa_M.shape

    wiersze_linie = set(wiersze_linie)
    kolumny_linie = set(kolumny_linie)

    # Szukamy elementów niepokrytych żadną linią
    niepokryte = []

    for i in range(n):
        for j in range(m):
            if i not in wiersze_linie and j not in kolumny_linie:
                niepokryte.append(nowa_M[i, j])

    if len(niepokryte) == 0:
        raise ValueError("Brak elementów niepokrytych liniami. Nie można wykonać kroku zwiększania.")

    min_niepokryty = min(niepokryte)

    # Modyfikacja macierzy
    for i in range(n):
        for j in range(m):

            # Element niepokryty żadną linią
            if i not in wiersze_linie and j not in kolumny_linie:
                nowa_M[i, j] -= min_niepokryty

            # Element na przecięciu dwóch linii
            elif i in wiersze_linie and j in kolumny_linie:
                nowa_M[i, j] += min_niepokryty

            # Element pokryty dokładnie jedną linią zostaje bez zmian

    # W metodzie węgierskiej dolne ograniczenie rośnie o:
    # min_niepokryty * (n - liczba_linii)
    liczba_linii = len(wiersze_linie) + len(kolumny_linie)
    nowe_dolne_ograniczenie = dolne_ograniczenie + min_niepokryty * (n - liczba_linii)

    return nowa_M, min_niepokryty, nowe_dolne_ograniczenie
