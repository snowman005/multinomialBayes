import pandas as pd
import numpy as np
import re
import math
from collections import Counter

STOPWORDS = set([
    "a","an","the","in","on","at","to","of","for","and","or","but","with","from","by","is","it","this","that","as","be"
])

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens


class Vectorizer:
    def __init__(self):
        self.vocab = {}
        self.inv_vocab = []

    def build_vocab(self, texts):

        vocab_set = set()
        for text in texts:
            tokens = clean_text(text)
            vocab_set.update(tokens)
        self.inv_vocab = sorted(list(vocab_set))
        self.vocab = {word: i for i, word in enumerate(self.inv_vocab)}

    def transform(self, texts):

        matrix = np.zeros((len(texts), len(self.vocab)), dtype=int)
        for idx, text in enumerate(texts):
            tokens = clean_text(text)
            for t in tokens:
                if t in self.vocab:
                    matrix[idx][self.vocab[t]] += 1
        return matrix

    def fit_transform(self, texts):
        self.build_vocab(texts)
        return self.transform(texts)

class NaiveBayes:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.prior = {}
        self.likelihood = {}
        self.classes = []

    def fit(self, X, y):
        self.classes = np.unique(y)
        total_docs = len(y)

        count_classes = Counter(y)
        for c in self.classes:
            self.prior[c] = math.log(count_classes[c] / total_docs)

        vocab_size = X.shape[1]

        for c in self.classes:
            Xc = X[y == c]
            word_counts = np.sum(Xc, axis=0)
            total_count = np.sum(word_counts)
            prob = (word_counts + self.alpha) / (total_count + self.alpha * vocab_size)
            self.likelihood[c] = np.log(prob)

    def predict_one(self, x):
        scores = {}
        for c in self.classes:
            scores[c] = self.prior[c] + np.sum(x * self.likelihood[c])
        return max(scores, key=scores.get)

    def predict(self, X):
        return np.array([self.predict_one(x) for x in X])

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def precision_recall_f1(y_true, y_pred):
    tp = Counter()
    fp = Counter()
    fn = Counter()
    classes = set(y_true)

    for t, p in zip(y_true, y_pred):
        if t == p:
            tp[t] += 1
        else:
            fp[p] += 1
            fn[t] += 1

    prec = {}
    rec = {}
    f1 = {}
    for c in classes:
        p = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) > 0 else 0
        r = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) > 0 else 0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0
        prec[c] = p
        rec[c] = r
        f1[c] = f
    return prec, rec, f1

def confusion_matrix(y_true, y_pred):
    classes = sorted(list(set(y_true)))
    mat = np.zeros((len(classes), len(classes)), dtype=int)
    index = {c: i for i, c in enumerate(classes)}
    for t, p in zip(y_true, y_pred):
        mat[index[t]][index[p]] += 1
    return mat

df = pd.read_csv("train1.csv")  
df = df.sample(frac=1, random_state=42)

train_size = int(0.8 * len(df))
train_df = df[:train_size]
test_df = df[train_size:]

X_train_text = train_df["headline"].tolist()
y_train = np.array(train_df["clickbait"].tolist())
X_test_text = test_df["headline"].tolist()
y_test = np.array(test_df["clickbait"].tolist())

vectorizer = Vectorizer()
X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)

nb = NaiveBayes(alpha=1.0)
nb.fit(X_train, y_train)

y_pred = nb.predict(X_test)

print("\nRezultate")
print("Acuratete:", accuracy(y_test, y_pred))
prec, rec, f1 = precision_recall_f1(y_test, y_pred)
print("Precizie:", prec)
print("Sensibilitate:", rec)
print("F1:", f1)
print("\nMatricea confuzie:")
print(confusion_matrix(y_test, y_pred))

print("\nExemple: (1 -> clickbait/ 0 -> non-clickbait)")
top_n_words_per_title = 5


for i in range(7):
  x_vec = X_test[i]
  headline = X_test_text[i]
  scores = {c: nb.prior[c] + np.sum(x_vec * nb.likelihood[c]) for c in nb.classes}
  scores_trunc = {c: round(float(s), 2) for c, s in scores.items()}
  pred = y_pred[i]
  word_scores = x_vec * nb.likelihood[pred] 
  present_indices = np.where(x_vec > 0)[0]
  sorted_indices = present_indices[np.argsort(word_scores[present_indices])[::-1]]
  top_words = [vectorizer.inv_vocab[idx] for idx in sorted_indices[:top_n_words_per_title]]
  print(f"Titlu: {headline}")
  print(f"Predictie: {pred}, Scor: {scores_trunc}")
  print(f"Top cuvinte ce contribue la raspuns: {top_words}\n")

