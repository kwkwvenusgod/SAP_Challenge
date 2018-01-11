import gensim.models.word2vec as word2vec
import numpy as np


def load_data(file_path):
    with open(file_path) as f:
        content = f.readlins()

    content = [x.strip() for x in content]
    sentences = [list(sentence) for sentence in content];
    return sentences


def train_character_embedding(sentences):
    embedding_model = word2vec(sentences)
    return embedding_model


def save_embedding():
    return


if __name__ == "__main__":
    training_file_path = "C:\\SAP_Challenge\\Offline-Challenge\\xtrain_obfuscated.txt"
    training_sentences = load_data(training_file_path)
    embeddings = train_character_embedding(training_sentences)
