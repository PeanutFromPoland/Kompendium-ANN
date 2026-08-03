# Sztuczne Sieci Neuronowe

## Spis treści

- [Sztuczne Sieci Neuronowe](#sztuczne-sieci-neuronowe)
  - [Spis treści](#spis-treści)
  - [1 Wprowadzenie](#1-wprowadzenie)
    - [1.1 Geneza](#11-geneza)
    - [1.2 Zastosowania](#12-zastosowania)
    - [1.3 Propagacja w przód](#13-propagacja-w-przód)
    - [1.4 Propagacja w tył (Backpropagation)](#14-propagacja-w-tył-backpropagation)
    - [1.5 Techniki zapobiegające overfittingowi](#15-techniki-zapobiegające-overfittingowi)
  - [2 Architektury ANN](#2-architektury-ann)
    - [2.1 Sieci konwolucyjne (Convolutional Neural Network)](#21-sieci-konwolucyjne-convolutional-neural-network)
    - [2.2 Generatywne Sieci Adwersalne (Generative Adversal Network)](#22-generatywne-sieci-adwersalne-generative-adversal-network)
    - [2.3 Sieci Rekurencyjne (Recurrent Neural Network)](#23-sieci-rekurencyjne-recurrent-neural-network)
    - [2.4 Długa Pamięć Krótkoterminowa LSTM (Long Short-Term Memory)](#24-długa-pamięć-krótkoterminowa-lstm-long-short-term-memory)
    - [2.5 GAT](#25-gat)
    - [2.6 Autoenkodery](#26-autoenkodery)
    - [2.7 Transformery](#27-transformery)
      - [2.7.1 Mechanizm atencji](#271-mechanizm-atencji)
      - [2.7.2 Feed forward](#272-feed-forward)
      - [2.7.3 Enkoder](#273-enkoder)
      - [2.7.4 Dekoder](#274-dekoder)
  - [3 Bibliografia](#3-bibliografia)
  
---

## 1 Wprowadzenie

Rozdział ten opowiada o istocie i zasadzie działania sztucznych sieci neuronowych. Opisuje ich genezę, budowę, zastosowania we współczesnym świecie i mechanizmy, które zachodzą zarówno podczas trenowania, jak i ewaluacji modeli opartych o sztuczne sieci neuronowe.

### 1.1 Geneza

Bezpośrednią inspiracją dla powstania sztucznych sieci neuronowych (które będę skrótowo odtąd nazywać ANN) jest budowa neuronów w ludzkim mózgu.

![image](imgs/neuron.png)

Neurony składają się z dendrytów, jądra komórkowego, ciała komórkowego, aksonu i synaps. Dendrydy otrzymują sygnały z sąsiednich neuronów i przekazują je do ciała i jądra komórkowego, które modyfikują sygnał. Akson przekazuje nowy sygnał do synaps podłączonych do dendrydów innych neuronów.

Sygnały w mózgu przechodzą między neuronami, w których poddawane są indywidualnym procesom transformacji. Siła sygnału wyjściowego neuronu zależy od siły sygnałów wejściowych.

Twórcy koncepcji ANN zaproponowali, aby siła sygnałów była reprezentowana przez liczby rzeczywiste, a procesy transformacji polegały na obliczaniu wartości funkcji liniowej zawierającej tyle samo zmiennych, co wejść do danego sztucznego neuronu i zastosowaniu na niej funkcji aktywacji, która pozwoli znormalizować wartość siły sygnału w sieci. Jest to bardzo ważne, gdyż bez tego pewne części ANN mogłyby w sposób niezamierzony (i na dodatek nieuczciwy) wpływać na wynik końcowy.

Przyjmijmy, że sieć neuronowa została wytrenowana do szacowania wartości mieszkania w zależności od metrażu, odległości od centrum i przeciętnych zarobków w tym mieście. Łatwo da się dostrzec, że dziedzina zmiennej opisującej przeciętne pensje mieści się w przedziale kilku, kilkunastu tysięcy. Gdyby nie stosować normalizacji zmiennych, to ta właśnie zmienna "przejęłaby kontrolę" nad modelem, co jest absolutnie niepożądane. Chcemy, aby każda zmienna w modelu miała wstępnie te same szanse.

### 1.2 Zastosowania

Sztuczne sieci neuronowe stosuje się m.in. do:

- szacowania przyszłych cen akcji na giełdzie;
- oceny ryzyka kredytowego;
- tłumaczenia tekstów na obce języki;
- wykrywania chorób ze zdjęć RTG;
- analizy sentymentu na podstawie wpisów w Internecie;
- generowania muzyki
- generowania filmów i obrazów
- wspomagania podejmowania decyzji w złożonych środowiskach operacyjnych

### 1.3 Propagacja w przód

Przyjmijmy, że jest sztuczna sieć neuronowa, która została wytrenowana do predykcji prawdopodobieństwa zawału serca. Jako wejście przyjmuje m.in. współczynnik spożycia alkoholu, palenie papierosów, czas aktywności fizycznej w tygodniu, BMI, ciśnienie krwi, etc.

Sieć neuronowa składa się z 3 warstw, gdzie pierwsza ma 26 neuronów, druga ma 104 neurony, a trzecia ma jeden neuron zwracający wynik od 0 do 1 oznaczający prawdopodobieństwo zawału serca.

Propagacja w przód polega na przekazywaniu sygnałów DO PRZODU warstwa po warstwie. W każdym neuronie sygnały z poprzedniej warstwy są sumaryzowane, modyfikowane parametrem bias, a następnie stosuje się funkcję aktywacji, która wprowadza element normalizacji.

$$
n(x) = a(\sum_i{w_i x_i + b})
$$

Dostępne funkcje aktywacji:

- Tanh

$$
a(x)=\frac{2}{1+e^{-2x}} - 1
$$

- Sigmoid

$$
a(x)=\frac{1}{1+e^{-x}}
$$

- ELU

$$
a(x)=\begin{cases}
x, & x>0\\
\alpha (e^x-1), & x\le0
\end{cases}
$$

- ReLU*

$$
a(x)=max(0, x)
$$

- Leaky ReLU

$$
a(x)=
\begin{cases}
x, & x>0\\
\alpha x, & x\le0
\end{cases}
$$

- SELU

$$
a(x)=\lambda \begin{cases}
x, & x>0\\
\alpha (e^x-1), & x\le0
\end{cases}
\lambda \approx 1.05
\alpha \approx 1.67
$$

- SoftPlus

$$
a(x)=log(1 + e^x)
$$

- Softmax*

$$
a(x_i)=\sigma(x_i)=\frac{e^{x_i}}{\sum_{j=1}^{n} e^{x_j}}
$$

*- funkcje te są jedynymi dozwolonymi na warstwach wyjściowych

### 1.4 Propagacja w tył (Backpropagation)

Propagacja w tył jest sposobem na wytrenowanie sztucznej sieci neuronowej. Proces treningu składa się z następujących kroków:

1. Ustaw wagi wstępne w modelu;
2. Użyj danych treningowych do przeprowadzenia propagacji w przód;
3. Wynik z propagacji porównaj z docelowym wynikiem i na jego podstawie oblicz błąd modelu;
4. Na podstawie wielkości błędu oblicz zmianę wag dla każdego neuronu warstwa po warstwie wstecz;
5. Powtórz proces od kroku 2., jeżeli to była ostatnia iteracja lub błąd stał się akceptowalny.

A więc propagacja w tył to nic innego jak przerzucanie błędu modelu od warstwy końcowej na sam początek. Błąd można obliczyć korzystając z:

- różnicy;
- błędu średniokwadratowego;
- entropii krzyżowej.

Aby można było oszacować zmianę wagi, skorzystamy z techniki gradientowej optymalizacji. Należy obliczyć gradient dla wyjścia modelu oraz przewidywanego wyjścia i odwrócić kierunek w stronę (lokalnego) optimum. Dla funkcji sigmoidalnej postaci:

$$
a(x)=\frac{1}{1+e^{-x}}
$$

pochodna funkcji to:

$$
\frac{d}{dx}a(x)=x(1-x)
$$

Dla funkcji ReLU:

$$
a(x)=max(0, x)
$$

pochodną funkcji jest:

$$
\frac{d}{dx}a(x)=\begin{cases}
1, & x>0 \\
0, & x<0
\end{cases}
$$

Dla neuronu wyjściowego (warstwy wyjściowej) wzór na zmianę wag połączeń kończących się w nim jest następujący:

$$
\delta_w=e_w\frac{d}{dx}a(x) \\

w_h = w_h + o_h^T \delta_w \lambda \\

b_w = b_w + \sum \delta_w\\
\delta_h = w_h \frac{d}{dx} a(o_h) \\

w_i = w_i + o_i^T \delta_h \\
b_h = b_h + \sum \delta_h\\

$$

Oznaczenia:

- $o_w$ - wyjście;
- $e_w$ - błąd na wyjściu;
- $o_h$ - wyjście warstwy ukrytej;
- $w_h$ - wagi połączeń między warstwą ukrytą, a wyjściem;
- $\lambda$ - współczynnik uczenia;
- $b_w$ - parametr bias w warstwie wyjściowej;
- $o_i$ - wejście;
- $w_i$ - wagi połączeń między wejściem, a warstwą ukrytą;
- $b_h$ - parametr bias w warstwie ukrytej.

Współczynnik uczenia ustawia się po to, aby ustabilizować trening. Bez tego model ma tendencję do wpadania w najbliższe optimum lokalne, które najczęściej będzie ono niesatysfakcjonujące.


### 1.5 Techniki zapobiegające overfittingowi

- Dzielenie zbioru danych na podzbiór treningowy i testowy. W praktyce często wyznacza się też osobny zbiór walidacyjny, który pozwala na bieżąco oceniać postęp treningu epoka za epoką;
- Walidacja krzyżowa - w przypadku większych modeli może okazać się zbyt kosztowna;
- Regularyzacja L1 (lasso) i L2 (ridge);
- Dropout;
- Selekcja najistotniejszych zmiennych;
- Wzbogacanie danych treningowych (data augmentation) - wprowadzanie do zbioru danych treningowych artefaktów, które w praktycznym zastosowaniu mogłyby zaburzać pracę, ale podczas treningu uodporniają model na anomalie otrzymane na wejściu. Sprowadza się to do dodawania mniej lub bardziej regularnych szumów, zakrywania części obrazu i innych manipulacji;
- Ograniczanie złożoności modelu.

## 2 Architektury ANN

### 2.1 Sieci konwolucyjne (Convolutional Neural Network)

### 2.2 Generatywne Sieci Adwersalne (Generative Adversal Network)

### 2.3 Sieci Rekurencyjne (Recurrent Neural Network)

### 2.4 Długa Pamięć Krótkoterminowa LSTM (Long Short-Term Memory)

### 2.5 GAT

### 2.6 Autoenkodery

### 2.7 Transformery

#### 2.7.1 Mechanizm atencji

#### 2.7.2 Feed forward

#### 2.7.3 Enkoder

#### 2.7.4 Dekoder

## 3 Bibliografia

1. R. Hurbans. Grokking Artificial Intelligence Algorithms. Manning Publications Co. Rok wydania 2020. ISBN: 9781617296185
2. L. Tunstall, L. von Werra, T. Wolf. Przetwarzanie języka naturalnego z wykorzystaniem transformerów. Helion S.A. 2024. ISBN: 978-83-289-0711-9
3. D. Chuan-En-Lin, 8 Simple Techniques to Prevent Overfitting, [online]. Dostęp w Internecie: <https://medium.com/data-science/8-simple-techniques-to-prevent-overfitting-4d443da2ef7d>. [dostęp: 31.07.2026]
4. GeeksforGeeks, Activation Functions in Neural Network, [online]. Dostęp w Internecie: <https://www.geeksforgeeks.org/machine-learning/activation-functions-neural-networks/>. [dostęp: 31.07.2026]
5. GeeksforGeeks, Backpropagation in Neural Network, [online]. Dostęp w Internecie: <https://www.geeksforgeeks.org/machine-learning/backpropagation-in-neural-network/>. [dostęp: 03.08.2026]