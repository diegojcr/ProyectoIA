import os
import math
import random
from collections import defaultdict, Counter
import re


# Cargar y etiquetar los datos
def load_dataset(path):
    dataset = []
    for category in os.listdir(path):
        category_path = os.path.join(path, category)
        if os.path.isdir(category_path):
            for file in os.listdir(category_path):
                file_path = os.path.join(category_path, file)
                with open(file_path, 'r', encoding='latin1') as f:
                    text = f.read().lower()
                    dataset.append((text, category))
    return dataset


# Preprocesar el texto
def preprocess(text):
    # Eliminar caracteres especiales y números
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenizar el texto
    tokens = text.lower().split()
    return tokens


# Separar el dataset en entrenamiento y prueba
def split_dataset(dataset, train_size=0.8):
    random.shuffle(dataset)
    split_index = int(len(dataset) * train_size)
    train_set = dataset[:split_index]
    test_set = dataset[split_index:]
    return train_set, test_set


# clase del clasificador Naive Bayes
class NaiveBayesClassifier:
    def __init__(self):
        self.class_word_counts = defaultdict(Counter)
        self.class_counts = Counter()
        self.vocab = set()
        self.total_docs = 0

    def train(self, train_set):
        self.total_docs = len(train_set)
        for text, label in train_set:
            tokens = preprocess(text)
            self.class_counts[label] += 1
            self.class_word_counts[label].update(tokens)
            self.vocab.update(tokens)
        self.word_totals = {cls: sum(tokens.values()) for cls, tokens in self.class_word_counts.items()}

    def predict(self, text):
        tokens = preprocess(text)
        class_scores = {}
        for cls in self.class_counts:
            log_prob = math.log(self.class_counts[cls] / self.total_docs)
            for token in tokens:
                word_frequency = self.class_word_counts[cls][token] + 1  # Laplace smoothing
                word_prob = word_frequency / (self.word_totals[cls] + len(self.vocab))
                log_prob += math.log(word_prob)
            class_scores[cls] = log_prob
        return max(class_scores, key=class_scores.get)


# evaluar el modelo
def evaluate(classifier, test_set):
    correct = 0
    for text, label in test_set:
        prediction = classifier.predict(text)
        if prediction == label:
            correct += 1
    accuracy = correct / len(test_set)
    return accuracy


_classifier = None

def load_classifier():
    global _classifier
    if _classifier is None:
        dataset_path = "C:/Users/diego/OneDrive - Universidad Rafael Landivar/Escritorio/naivebayes_web/classifyapp/archive/BBC News Summary/News Articles"
        dataset = load_dataset(dataset_path)
        train_data, _ = split_dataset(dataset)
        nb = NaiveBayesClassifier()
        nb.train(train_data)
        _classifier = nb
    return _classifier

def classify_text(text):
    clf = load_classifier()
    return clf.predict(text)