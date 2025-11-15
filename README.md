# Clasificator Multinomial Bayes pentru Clickbait

## Descriere

Acest proiect implementează un **clasificator multinomial Bayes** pentru detectarea titlurilor **clickbait**. Modelul analizează cuvintele din titluri și calculează probabilitatea ca un titlu să fie clickbait sau non-clickbait, pe baza frecvenței cuvintelor în setul de antrenament.

Proiectul include:

* Preprocesarea textului (lowercase, eliminare stopwords și simboluri speciale)
* Vectorizare text (bag-of-words)
* Calcul probabilități a priori și condiționate
* Clasificarea titlurilor necunoscute
* Evaluarea performanței modelului (acuratețe, precizie, sensibilitate, F1 și matrice de confuzie)
* Exemple de cuvinte care au contribuit cel mai mult la predicție

---

## Teoria folosită

**Naive Bayes Multinomial** se bazează pe următoarele formule:

1. Probabilitatea a priori pentru clasa (c):
   [
   P(c) = \frac{\text{număr documente în clasa } c}{\text{număr total documente}}
   ]

2. Probabilitatea condiționată pentru cuvinte (w_i) în clasa (c):
   [
   P(w_i|c) = \frac{\text{număr apariții cuvânt } w_i \text{ în clasa } c + \alpha}{\text{total cuvinte în clasa } c + \alpha \cdot V}
   ]
   unde (\alpha) este parametru Laplace smoothing și (V) este dimensiunea vocabularului.

3. Probabilitatea ca un text (d) să aparțină clasei (c):
   [
   P(c|d) \propto P(c) \prod_{i} P(w_i|c)^{\text{număr apariții } w_i \text{ în } d}
   ]

---

## Structura codului

* `nb_clickbait.py`: fișier principal cu implementarea vectorizatorului și a modelului Naive Bayes
* Funcții principale:

  * `clean_text(text)` – curăță și tokenizează textul
  * `Vectorizer` – creează bag-of-words și transformă textul în matrice numerică
  * `NaiveBayes` – antrenează și face predicții
  * `accuracy`, `precision_recall_f1`, `confusion_matrix` – metrici de evaluare

---

## Librării necesare

* Python 3.x
* pandas
* numpy
* re
* math
* collections (Counter)

Instalare librării:

```bash
pip install pandas numpy
```

---

## Rulare

Fișierul CSV trebuie să conțină două coloane:

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

## Exemple de utilizare

```python
# Predicție pentru titlu nou
new_title = "10 Secrets You Never Knew About Cats"
x_vec = vectorizer.transform([new_title])[0]
pred = nb.predict_one(x_vec)
print("Titlu:", new_title)
print("Predictie:", pred)
```

---

## Referințe

1. [Naive Bayes Classifier – Wikipedia](https://en.wikipedia.org/wiki/Naive_Bayes_classifier)
2. [Bag of Words Model](https://en.wikipedia.org/wiki/Bag-of-words_model)
3. [Laplace Smoothing](https://en.wikipedia.org/wiki/Additive_smoothing)
