# Sztuczne Sieci Neuronowe


## Spis treści

- [Wprowadzenie](#wprowadzenie)
    - [Geneza](#geneza)
    - [Zastosowania](#zastosowania)
    - [Propagacja w przód](#propagacja-w-przód)
    - [Propagacja w tył (Backpropagation)](#propagacja-w-tył-backpropagation)
- [Architektury ANN](#architektury-ann)
    - [Sieci konwolucyjne (Convolutional Neural Network)](#sieci-konwolucyjne-convolutional-neural-network)
    - [Generatywne Sieci Adwersalne (Generative Adversal Network)](#generatywne-sieci-adwersalne-generative-adversal-network)
    - [Sieci Rekurencyjne (Recurrent Neural Network)](#sieci-rekurencyjne-recurrent-neural-network)
    - [Długa Pamięć Krótkoterminowa LSTM (Long Short-Term Memory)](#długa-pamięć-krótkoterminowa-lstm-long-short-term-memory)
    - [GAT](#gat)
    - [Autoenkodery](#autoenkodery)
    - [Transformery](#transformery)
  
---

## Wprowadzenie

Rozdział ten opowiada o istocie i zasadzie działania sztucznych sieci neuronowych. Opisuje ich genezę, budowę, zastosowania we współczesnym świecie i mechanizmy, które zachodzą zarówno podczas trenowania, jak i ewaluacji modeli opartych o sztuczne sieci neuronowe.


### Geneza

Bezpośrednią inspiracją dla powstania sztucznych sieci neuronowych (które będę skrótowo odtąd nazywać ANN) jest budowa neuronów w ludzkim mózgu. 

![image](imgs/neuron.png)

Neurony składają się z dendrytów, jądra komórkowego, ciała komórkowego, aksonu i synaps. Dendrydy otrzymują sygnały z sąsiednich neuronów i przekazują do ciała i jądra komórkowego, które modyfikują sygnał. Akson przekazuje nowy sygnał do synaps podłączonych do dendrydów innych neuronów. 

Sygnały w mózgu przechodzą między neuronami, w których poddawane są indywidualnym procesom transformacji. Siła sygnału wyjściowego neuronu zależy od siły sygnałów wejściowych. 

Twórcy koncepcji ANN zaproponowali, aby siła sygnałów była reprezentowana przez liczby rzeczywiste, a procesy transformacji polegały na obliczaniu wartości funkcji liniowej zawierającej tyle samo zmiennych, co wejść do danego sztucznego neuronu i zastosowaniu na niej funkcji aktywacji, która pozwoli znormalizować wartość siły sygnału w sieci. Jest to bardzo ważne, gdyż bez tego pewne części ANN mogłyby w sposób niezamierzony (i na dodatek nieuczciwy) wpływać na wynik końcowy. 

Przyjmijmy, że sieć neuronowa została wytrenowana do szacowania wartości mieszkania w zależności od metrażu, odległości od centrum i przeciętnych zarobków w tym mieście. Łatwo da się dostrzec, że dziedzina zmiennej opisującej przeciętne pensje mieści się w przedziale kilku, kilkunastu tysięcy. Gdyby nie stosować normalizacji zmiennych do przedziału [0, 1], to ta właśnie zmienna "przejęłaby kontrolę" nad modelem, co jest absolutnie niepożądane. Chcemy, aby każda zmienna w modelu miała wstępnie te same szanse. 

### Zastosowania

Sztuczne sieci neuronowe stosuje się m.in. do: 
- szacowania przyszłych cen akcji na giełdzie; 
- oceny ryzyka kredytowego;
- tłumaczenia tekstów na obce języki;
- wykrywania chorób ze zdjęć RTG;
- analizy sentymentu na podstawie wpisów w Internecie.


### Propagacja w przód



### Propagacja w tył (Backpropagation)

## Architektury ANN

### Sieci konwolucyjne (Convolutional Neural Network)

### Generatywne Sieci Adwersalne (Generative Adversal Network)

### Sieci Rekurencyjne (Recurrent Neural Network)

### Długa Pamięć Krótkoterminowa LSTM (Long Short-Term Memory) 

### GAT 

### Autoenkodery

### Transformery

#### Mechanizm atencji

#### Feed forward

#### Enkoder

#### Dekoder


## Źródła
1. R. Hurbans. Grokking Artificial Intelligence Algorithms. Manning Publications Co. Rok wydania 2020. ISBN: 9781617296185
2. L. Tunstall, L. von Werra, T. Wolf. Przetwarzanie języka naturalnego z wykorzystaniem transformerów. Helion S.A. 2024. ISBN: 978-83-289-0711-9