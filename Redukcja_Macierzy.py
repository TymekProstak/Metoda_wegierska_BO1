import numpy as np

def redukcja_macierzy(macierz_kosztow):
    """
    Wykonuje pierwszy etap metody węgierskiej (redukcję macierzy).
    
    Argument:
        macierz_kosztow: Tablica lub lista dwuwymiarowa (n x n).
        
    Zwraca:
        Zredukowaną macierz jako ndarray, gdzie każdy wiersz i każda 
        kolumna posiadają co najmniej jedno zero.
    """
    m = np.array(macierz_kosztow, dtype=float)
    m -= m.min(axis=1, keepdims=True)
    m -= m.min(axis=0, keepdims=True)

    return m