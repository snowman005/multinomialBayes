# Clasificator Multinomial Bayes pentru Clickbait

## Descriere

Acest proiect implementeaza un **clasificator multinomial Bayes** pentru detectarea titlurilor **clickbait**. Modelul analizeaza cuvintele din titluri si calculeaza probabilitatea ca un titlu sa fie clickbait sau non-clickbait, pe baza frecventei cuvintelor in setul de antrenament.

Proiectul include:

* Preprocesarea textului (lowercase, eliminare stopwords si simboluri speciale)
* Vectorizare text (bag-of-words)
* Calcul probabilitati a priori si conditionate
* Clasificarea titlurilor necunoscute
* Evaluarea performantei modelului (acuratete, precizie, sensibilitate, F1 si matrice de confuzie)
* Exemple de cuvinte care au contribuit cel mai mult la predictie

---

## Teoria folosita

**Naive Bayes Multinomial** se bazeaza pe urmatoarele formule:

1. **Probabilitatea a priori pentru clasa (c):**

```
P(c) = numar documente in clasa c / numar total documente
```

2. **Probabilitatea conditionata pentru cuvinte (w_i) in clasa (c):**

```
P(w_i | c) = (numar aparitii cuvant w_i in clasa c + α) / (total cuvinte in clasa c + α * V)
```

unde α este parametrul de Laplace smoothing si V este dimensiunea vocabularului.

3. **Probabilitatea ca un text (d) sa apartina clasei (c):**

```
P(c | d) ∝ P(c) * Π P(w_i | c)^(numar aparitii w_i in d)
```

---

## Structura codului

* `nb_clickbait.py`: fisier principal cu implementarea vectorizatorului si a modelului Naive Bayes
* Functii principale:

  * `clean_text(text)` – curata si tokenizeaza textul
  * `Vectorizer` – creeaza bag-of-words si transforma textul in matrice numerica
  * `NaiveBayes` – antreneaza si face predictii
  * `accuracy`, `precision_recall_f1`, `confusion_matrix` – metrici de evaluare

---

## Librarii necesare

* Python 3.x
* pandas
* numpy
* re
* math
* collections (Counter)

Instalare librarii:

```bash
pip install pandas numpy
```

---

## Rulare

Fisierul CSV trebuie sa contina doua coloane:

* `headline` – titlul articolului
* `clickbait` – eticheta 0 (non-clickbait) sau 1 (clickbait)

Rulare script:

```bash
python3 nb_clickbait.py --csv train1.csv
```

---

## Exemplu de output

```
Rezultate
Acuratete: 0.971875
Precizie: {0: 0.9871, 1: 0.9567}
Sensibilitate: {0: 0.9577, 1: 0.9868}
F1: {0: 0.9722, 1: 0.9715}

Matricea confuzie:
[[3148  139]
 [  41 3072]]

Exemple: (1 -> clickbait / 0 -> non-clickbait)
Titlu: German internet watchdog to remove URLs to 'Virgin Killer' from search engines
Predictie: 0, Scor: {0: -73.28, 1: -82.06}
Top cuvinte ce contribuie la raspuns: ['german', 'internet', 'search', 'killer', 'remove']

Titlu: This Mom Just Shut It Down With Her Hilarious "First Day Of School" Picture
Predictie: 1, Scor: {0: -86.13, 1: -72.71}
Top cuvinte ce contribuie la raspuns: ['just', 'her', 'first', 'day', 'hilarious']
```

---

## Referinte

1. [Naive Bayes Classifier – Wikipedia](https://en.wikipedia.org/wiki/Naive_Bayes_classifier)
2. [Bag of Words Model](https://en.wikipedia.org/wiki/Bag-of-words_model)
3. [Laplace Smoothing](https://en.wikipedia.org/wiki/Additive_smoothing)
