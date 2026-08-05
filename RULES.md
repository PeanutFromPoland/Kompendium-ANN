# RULES

1. Commity:
   - Opisuj je trzema tagami. Pierwszy tag wskazuje na numer iteracji np. [IT1]. Drugi wskazujący na rodzaj wykonanej pracy. [!] oznacza commit organizacyjny, zaś np. [S] oznacza commit związany z wykonaniem projektu System wieloagentowy. Trzeci tag [human] oznacza pracę wykonaną przez człowieka, a [AI] oznacza pracę wykonaną przez AI;
   - W nazwach commitów wskazuj na modyfikowane pliki i w krótkich słowach opisz zmiany;
   - Jeżeli commit jest na branchu `production` to ma zawierać TYLKO i wyłącznie zmergowane postępy z innych branchy opatrzone stosownym opisem np. `Task[S1][human] wykonany` . Po wdrożeniu ewentualnych poprawek oraz USUNIĘCIU plików roboczych commit ma się nazywać np. `v0.1`, gdzie pierwsza liczba odpowiada numerowi wersji stabilnej, druga liczba numerowi iteracji (przyjmijmy roboczo, że jedna iteracja jest co tydzień);
   - Przykłady:
     - `[IT1][W][human] Naprawiono trening w ann.ipynb`
     - `[IT4][S][AI] Wygenerowano prompty dla agenta dzielącego skrypt na slajdy`
2. Iteracje
   - Trwają one tydzień lub dwa. Datę końca iteracji ustala się w pliku `ITERATIONS.md` wraz z opisem wykonanych postępów i uwag;
3. Branche
   - Niech zaczynają się od taga wskazującego na mini-projekt np. [T];
   - Dalsza część nazwy niech odnosi się do zadania lub projektu.
4. Wersjonowanie
   - Wersja stabilna to wersja, która przeszła pozytywnie proces CI/CD;
   - Wersja iteracyjna to wersja, która zawiera ostateczne wersje plików odnoszących się do zadań. Nie powinna ona zawierać roboczych postępów w innych zadaniach. W ramach tej wersji POWINIEN być uzupełniony dziennik iteracji `ITERATIONS.md`
