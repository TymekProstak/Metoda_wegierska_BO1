import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Set, Optional


# ============================================================
# Typy pomocnicze
# ============================================================

Matrix = np.ndarray
Position = Tuple[int, int]   # (wiersz, kolumna), indeksowanie od 0


@dataclass
class LineCover:
    """
    Przechowywana jest informacja o tym, które wiersze i kolumny zostały wykreślone liniami.
    """
    rows: Set[int]
    cols: Set[int]


@dataclass
class IterationSnapshot:
    """
    Przechowywany jest jeden etap działania algorytmu, żeby potem łatwo wypisać go w sprawozdaniu.
    """
    iteration: int
    matrix: Matrix
    independent_zeros: List[Position]
    line_cover: Optional[LineCover]
    h: Optional[float]
    lower_bound: float
    description: str


@dataclass
class HungarianResult:
    """
    Przechowywany jest wynik końcowy metody węgierskiej.
    """
    assignment: List[Position]
    optimal_cost: float
    lower_bound: float
    trace: List[IterationSnapshot]


# ============================================================
# Dane do zadania 2
# ============================================================

def create_cost_matrix() -> Matrix:
    """
    Tworzona jest macierz kosztów 6x6 tak, aby po samej redukcji nie dało się jeszcze od razu wybrać 6 zer niezależnych.
    """
    return np.array([
        [10, 60, 62, 35, 56, 24],
        [28, 15, 35,  8, 59, 15],
        [89, 69, 46, 64, 34, 51],
        [44, 39, 30, 35, 24, 75],
        [52, 48, 11, 81, 87, 19],
        [47, 25, 22, 53, 26, 85]
    ], dtype=float)


# ============================================================
# Funkcje Michała
# ============================================================

def reduce_matrix(cost_matrix: Matrix) -> Tuple[Matrix, np.ndarray, np.ndarray, float]:
    """
    TODO: Michał
    Redukcja macierzy:
    - redukcja wierszy
    - redukcja kolumn

    Zwraca:
    - macierz po redukcji
    - minima wierszy
    - minima kolumn
    - dolne ograniczenie
    """
    m = cost_matrix.copy()

    # Redukcja wierszy
    row_min = m.min(axis=1)
    m = m - row_min[:, np.newaxis]

    # Redukcja kolumn
    col_min = m.min(axis=0)
    m = m - col_min

    # Dolne ograniczenie
    lower_bound = float(np.sum(row_min) + np.sum(col_min))

    return m, row_min, col_min, lower_bound


def cover_zeros_with_min_lines(matrix: Matrix,
                               independent_zeros: List[Position]) -> LineCover:
    n = matrix.shape[0]

    zeros = (matrix == 0)

    # ===== 1. Odtworzenie matchingu =====
    match_row = [-1] * n
    match_col = [-1] * n

    for r, c in independent_zeros:
        match_row[r] = c
        match_col[c] = r

    # ===== 2. Procedura znakowania =====
    marked_rows = [False] * n
    marked_cols = [False] * n

    # Zaznacz wiersze bez przypisanego zera
    for r in range(n):
        if match_row[r] == -1:
            marked_rows[r] = True

    changed = True
    while changed:
        changed = False

        # zaznacz kolumny z zerami w zaznaczonych wierszach
        for r in range(n):
            if marked_rows[r]:
                for c in range(n):
                    if zeros[r, c] and not marked_cols[c]:
                        marked_cols[c] = True
                        changed = True

        # zaznacz wiersze przypisane do tych kolumn
        for c in range(n):
            if marked_cols[c]:
                r = match_col[c]
                if r != -1 and not marked_rows[r]:
                    marked_rows[r] = True
                    changed = True

    # ===== 3. Wyznaczenie linii =====
    rows = {i for i in range(n) if not marked_rows[i]}
    cols = {j for j in range(n) if marked_cols[j]}

    return LineCover(rows=rows, cols=cols)



# ============================================================
# Funkcje Maksa
# ============================================================

def find_independent_zeros(matrix: Matrix) -> List[Position]:
    """
    TODO: Maks

    Funkcja znajduje maksymalny zbiór zer niezależnych.

    Zera niezależne to takie zera, że:
    - żadne dwa nie leżą w tym samym wierszu,
    - żadne dwa nie leżą w tej samej kolumnie.

    Zwracana jest lista pozycji wybranych zer:
    [(0, 0), (5, 1), (4, 2), ...]
    """
    n, m = matrix.shape

    # Dla każdego wiersza zapisujemy kolumny, w których są zera
    zera_w_wierszach = []

    for i in range(n):
        kolumny_zer = []

        for j in range(m):
            if matrix[i, j] == 0:
                kolumny_zer.append(j)

        zera_w_wierszach.append(kolumny_zer)

    # match_col[j] = numer wiersza, który ma przypisane zero w kolumnie j
    # -1 oznacza, że kolumna nie jest jeszcze zajęta
    match_col = [-1] * m

    def znajdz_przypisanie(i: int, odwiedzone_kolumny: List[bool]) -> bool:
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
    independent_zeros = []

    for j in range(m):
        if match_col[j] != -1:
            independent_zeros.append((match_col[j], j))

    # Sortujemy, żeby wynik był czytelny
    independent_zeros.sort()

    return independent_zeros
    # raise NotImplementedError("Maks uzupełnia algorytm wyznaczania zer niezależnych.")


def increase_independent_zeros_step(matrix: Matrix,
                                    line_cover: LineCover) -> Tuple[Matrix, float]:
    """
    TODO: Maks

    Funkcja wykonuje krok zwiększania liczby zer niezależnych.

    Procedura:
    1. Znajduje się najmniejszy element niepokryty liniami.
    2. Odejmowany jest on od wszystkich elementów niepokrytych.
    3. Dodawany jest on do elementów pokrytych dwiema liniami.
    4. Elementy pokryte jedną linią pozostają bez zmian.

    Zwracane są:
    - nowa macierz,
    - wartość h, czyli najmniejszy element niepokryty.
    """
    new_matrix = matrix.copy()

    n, m = new_matrix.shape

    rows_lines = set(line_cover.rows)
    cols_lines = set(line_cover.cols)

    # Szukamy elementów niepokrytych żadną linią
    uncovered = []

    for i in range(n):
        for j in range(m):
            if i not in rows_lines and j not in cols_lines:
                uncovered.append(new_matrix[i, j])

    if len(uncovered) == 0:
        raise ValueError("Brak elementów niepokrytych liniami. Nie można wykonać kroku zwiększania.")

    h = min(uncovered)

    # Modyfikacja macierzy
    for i in range(n):
        for j in range(m):

            # Element niepokryty żadną linią
            if i not in rows_lines and j not in cols_lines:
                new_matrix[i, j] -= h

            # Element na przecięciu dwóch linii
            elif i in rows_lines and j in cols_lines:
                new_matrix[i, j] += h

            # Element pokryty dokładnie jedną linią zostaje bez zmian

    return new_matrix, h
    # raise NotImplementedError("Maks uzupełnia krok zwiększania liczby zer niezależnych.")


# ============================================================
# Funkcje Tymka — schemat ogólny algorytmu
# ============================================================

def calculate_assignment_cost(original_matrix: Matrix,
                              assignment: List[Position]) -> float:
    """
    Liczony jest koszt przydziału na podstawie oryginalnej macierzy kosztów.
    """
    total_cost = 0.0

    for row, col in assignment:
        total_cost += original_matrix[row, col]

    return total_cost


def update_lower_bound_after_increase(lower_bound: float,
                                      h: float,
                                      n: int,
                                      line_cover: LineCover) -> float:
    """
    Aktualizowane jest dolne ograniczenie po kroku zwiększania liczby zer.

    Jeżeli k oznacza liczbę linii, to dolne ograniczenie rośnie o:

        h * (n - k)

    gdzie:
    - h to najmniejszy element niepokryty,
    - n to rozmiar macierzy,
    - k to liczba wykreślonych linii.
    """
    number_of_lines = len(line_cover.rows) + len(line_cover.cols)
    return lower_bound + h * (n - number_of_lines)


def solve_hungarian(cost_matrix: Matrix) -> HungarianResult:
    """
    Realizowany jest główny schemat metody węgierskiej.

    - korzysta z redukcji macierzy,
    - szuka zer niezależnych,
    - sprawdza warunek zakończenia,
    - w razie potrzeby wykreśla zera liniami,
    - wykonuje krok zwiększania liczby zer,
    - zapisuje wszystkie etapy do śladu obliczeń.
    """

    original_matrix = cost_matrix.copy()

    if cost_matrix.shape[0] != cost_matrix.shape[1]:
        raise ValueError("Macierz kosztów musi być kwadratowa.")

    n = cost_matrix.shape[0]
    trace: List[IterationSnapshot] = []

    # --------------------------------------------------------
    # 1. Redukcja całkowita macierzy
    # --------------------------------------------------------
    reduced_matrix, row_reduction, col_reduction, lower_bound = reduce_matrix(cost_matrix)

    trace.append(
        IterationSnapshot(
            iteration=0,
            matrix=reduced_matrix.copy(),
            independent_zeros=[],
            line_cover=None,
            h=None,
            lower_bound=lower_bound,
            description="Macierz po redukcji wierszy i kolumn."
        )
    )

    iteration = 1

    # --------------------------------------------------------
    # 2. Główna pętla metody węgierskiej
    # --------------------------------------------------------
    while True:

        # Szukany jest maksymalny zbiór zer niezależnych.
        independent_zeros = find_independent_zeros(reduced_matrix)

        # Jeżeli znaleziono n zer niezależnych, to uzyskano rozwiązanie optymalne.
        if len(independent_zeros) == n:
            optimal_cost = calculate_assignment_cost(original_matrix, independent_zeros)

            trace.append(
                IterationSnapshot(
                    iteration=iteration,
                    matrix=reduced_matrix.copy(),
                    independent_zeros=independent_zeros,
                    line_cover=None,
                    h=None,
                    lower_bound=lower_bound,
                    description="Znaleziono pełny zbiór zer niezależnych. Algorytm zakończony."
                )
            )

            return HungarianResult(
                assignment=independent_zeros,
                optimal_cost=optimal_cost,
                lower_bound=lower_bound,
                trace=trace
            )

        # Jeżeli nie ma jeszcze n zer niezależnych, wykreślane są wszystkie zera minimalną liczbą linii.
        line_cover = cover_zeros_with_min_lines(reduced_matrix, independent_zeros)

        trace.append(
            IterationSnapshot(
                iteration=iteration,
                matrix=reduced_matrix.copy(),
                independent_zeros=independent_zeros,
                line_cover=line_cover,
                h=None,
                lower_bound=lower_bound,
                description="Nie znaleziono pełnego przydziału. Wykreślono zera minimalną liczbą linii."
            )
        )

        # Wykonywany jest krok zwiększania liczby zer niezależnych.
        reduced_matrix, h = increase_independent_zeros_step(reduced_matrix, line_cover)

        # Aktualizowane jest dolne ograniczenie.
        lower_bound = update_lower_bound_after_increase(
            lower_bound=lower_bound,
            h=h,
            n=n,
            line_cover=line_cover
        )

        trace.append(
            IterationSnapshot(
                iteration=iteration,
                matrix=reduced_matrix.copy(),
                independent_zeros=[],
                line_cover=line_cover,
                h=h,
                lower_bound=lower_bound,
                description="Wykonano krok zwiększania liczby zer niezależnych."
            )
        )

        iteration += 1


# ============================================================
# Funkcje do wypisywania wyników
# ============================================================

def to_one_based_positions(positions: List[Position]) -> List[Tuple[int, int]]:
    """
    Zamieniane jest indeksowanie od 0 na indeksowanie od 1.
    """
    return [(row + 1, col + 1) for row, col in positions]


def print_matrix(title: str, matrix: Matrix) -> None:
    """
    Wypisywana jest macierz w czytelnej postaci.
    """
    print(title)
    print(np.array2string(matrix, precision=2, suppress_small=True))
    print()


def print_result(result: HungarianResult, original_matrix: Matrix) -> None:
    """
    Wypisywany jest końcowy wynik algorytmu.
    """
    print("=" * 60)
    print("WYNIK KOŃCOWY")
    print("=" * 60)

    print("Przydział końcowy, indeksowanie od 1:")
    print(to_one_based_positions(result.assignment))
    print()

    print("Koszty wybranych przydziałów:")
    for row, col in result.assignment:
        print(f"P{row + 1} -> Z{col + 1}: koszt = {original_matrix[row, col]:.0f}")

    print()
    print(f"Optymalny koszt przydziału: {result.optimal_cost:.0f}")
    print(f"Końcowe dolne ograniczenie: {result.lower_bound:.0f}")


def print_trace(result: HungarianResult) -> None:
    """
    Wypisywany jest ślad działania algorytmu, czyli macierze pośrednie i informacje z kolejnych etapów.
    """
    print("=" * 60)
    print("ŚLAD DZIAŁANIA ALGORYTMU")
    print("=" * 60)

    for step in result.trace:
        print(f"Iteracja: {step.iteration}")
        print(step.description)
        print(f"Dolne ograniczenie: {step.lower_bound:.0f}")

        print_matrix("Macierz:", step.matrix)

        if step.independent_zeros:
            print("Zera niezależne, indeksowanie od 1:")
            print(to_one_based_positions(step.independent_zeros))
            print()

        if step.line_cover is not None:
            rows = sorted([r + 1 for r in step.line_cover.rows])
            cols = sorted([c + 1 for c in step.line_cover.cols])
            print(f"Wykreślone wiersze: {rows}")
            print(f"Wykreślone kolumny: {cols}")
            print()

        if step.h is not None:
            print(f"Najmniejszy element niepokryty h = {step.h:.0f}")
            print()

        print("-" * 60)


# ============================================================
# Program główny
# ============================================================

def main() -> None:
    """
    Zadanie 2.
    """

    cost_matrix = create_cost_matrix()

    print_matrix("Macierz kosztów początkowa:", cost_matrix)

    result = solve_hungarian(cost_matrix)

    print_trace(result)
    print_result(result, cost_matrix)


if __name__ == "__main__":
    main()
