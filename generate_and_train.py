# generate_and_train.py

import sympy as sp
import random
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump

x = sp.Symbol('x')

# ----- Data Generators with Controlled Phrasing -----
def gen_by_parts(n):
    funcs = [sp.log(x), sp.exp(x), sp.sin(x), sp.cos(x)]
    samples = []
    for _ in range(n):
        f = random.choice(funcs)
        a = random.randint(1, 5)
        phrases = [
            f"Find the integral of {a}*x*{f}",
            f"Evaluate ∫ {a}x{f}",
            f"Integration of {a}*x*{f}",
            f"Compute ∫ {a}*x*{f}"
        ]
        samples.append((random.choice(phrases), "by_parts"))
    return samples

def gen_by_sub(n):
    samples = []
    for _ in range(n):
        a, b = random.randint(1, 5), random.randint(1, 5)
        inner = a*x**2 + b
        options = [sp.cos(inner)*2*x, sp.sin(inner)*2*x, sp.exp(inner)*2*x]
        f = random.choice(options)
        phrases = [
            f"Evaluate integral of {f}",
            f"Find ∫ {f}",
            f"Integration of expression {f}",
            f"Compute integral of {f}"
        ]
        samples.append((random.choice(phrases), "by_substitution"))
    return samples

def gen_pf(n):
    samples = []
    for _ in range(n):
        num = random.randint(1, 5)*x + random.randint(0, 5)
        den = (x + random.randint(1, 5)) * (x + random.randint(1, 5))
        f = sp.simplify(num / den)
        phrases = [
            f"Integrate using partial fractions: {f}",
            f"∫ {f} using decomposition",
            f"Find the integral of rational expr: {f}",
            f"Evaluate integral: {f}"
        ]
        samples.append((random.choice(phrases), "partial_fractions"))
    return samples

def gen_others(n):
    extras = [sp.sqrt(x), sp.log(x)**2, sp.sin(x**3)]
    samples = []
    for _ in range(n):
        f = random.choice(extras)
        phrases = [
            f"Find the integral of {f}",
            f"Evaluate ∫ {f}",
            f"Integration of {f}",
            f"Compute integral of {f}"
        ]
        samples.append((random.choice(phrases), "others"))
    return samples

# ----- Preprocessing -----
def preprocess(t):
    t = t.lower()
    return re.sub(r'[^a-z0-9*/+()^ .-]', ' ', t)

# ----- Generate Dataset -----
all_data = (
    gen_by_parts(2000) +
    gen_by_sub(1500) +
    gen_pf(1500) +
    gen_others(1000)
)

df = pd.DataFrame(all_data, columns=["question_text", "label"])
df.to_csv("dataset.csv", index=False)

# ----- Vectorization -----
df["clean"] = df["question_text"].apply(preprocess)
vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english', min_df=2)
X = vectorizer.fit_transform(df["clean"])
y = df["label"]

# ----- Train-Test Split -----
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# ----- Train Model -----
model = MultinomialNB()
model.fit(X_tr, y_tr)

# ----- Save Artifacts -----
dump(vectorizer, "vectorizer.pkl")
dump(model, "model.pkl")

# ----- Evaluation -----
print("Train accuracy:", accuracy_score(y_tr, model.predict(X_tr)))
print("Test accuracy:", accuracy_score(y_te, model.predict(X_te)))
print(classification_report(y_te, model.predict(X_te), zero_division=1))
