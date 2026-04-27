import numpy as np

def wykresl_minimalne_linie(macierz):
    """
    Wyznacza minimalną liczbę linii (wierszy i kolumn) pokrywających wszystkie zera.
    Zwraca dwa wektory logiczne: (row_mask, col_mask).
    """
    m = np.array(macierz)
    n = m.shape[0]
    zera = (m == 0)

    # 1. Znalezienie maksymalnego skojarzenia (algorytm zachłanny + ścieżki powiększające)
    matching_w_kolumnie = np.full(n, -1)
    matching_w_wierszu = np.full(n, -1)

    def szukaj_sciezki(u, visited):
        for v in range(n):
            if zera[u, v] and not visited[v]:
                visited[v] = True
                if matching_w_kolumnie[v] < 0 or szukaj_sciezki(matching_w_kolumnie[v], visited):
                    matching_w_kolumnie[v] = u
                    matching_w_wierszu[u] = v
                    return True
        return False

    for i in range(n):
        szukaj_sciezki(i, np.zeros(n, dtype=bool))

    # 2. Procedura znakowania (Marking procedure)
    marked_rows = np.zeros(n, dtype=bool)
    marked_cols = np.zeros(n, dtype=bool)

    # Zaznacz wiersze, które nie mają przypisanego zera (brak skojarzenia)
    marked_rows[matching_w_wierszu == -1] = True

    change = True
    while change:
        change = False
        # Zaznacz kolumny, które mają zero w zaznaczonym wierszu
        for r in range(n):
            if marked_rows[r]:
                for c in range(n):
                    if zera[r, c] and not marked_cols[c]:
                        marked_cols[c] = True
                        change = True
        
        # Zaznacz wiersze, które mają przypisane zero w zaznaczonej kolumnie
        for c in range(n):
            if marked_cols[c]:
                r = matching_w_kolumnie[c]
                if r >= 0 and not marked_rows[r]:
                    marked_rows[r] = True
                    change = True

    # 3. Wyznaczenie linii pokrywających
    # Linie to: NIEoznaczone wiersze i OZNACZONE kolumny
    row_mask = ~marked_rows
    col_mask = marked_cols

    return row_mask, col_mask