import json
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from src.preprocess import clean_text
from src.model import build_model



DATA_PATH = "data/emotions.csv"
MODEL_PATH = "models/emotion_model.keras"
TOKENIZER_PATH = "models/tokenizer.json"
LABELS_PATH = "models/labels.json"

VOCAB_SIZE = 20000
MAX_LEN = 40
EPOCHS = 8
BATCH_SIZE = 32

def main():
    df = pd.read_csv(DATA_PATH)
    df["text"] = df["text"].astype(str).apply(clean_text)
    labels = df["label"].astype(str).values

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(labels)

    # Tokenize
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
    tokenizer.fit_on_texts(df["text"].tolist())

    X = tokenizer.texts_to_sequences(df["text"].tolist())
    X = pad_sequences(X, maxlen=MAX_LEN, padding="post", truncating="post")

    # Train-test split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    num_classes = len(le.classes_)
    model = build_model(VOCAB_SIZE, MAX_LEN, num_classes)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)
    ]

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks
    )

    # Save model + tokenizer + labels
    model.save(MODEL_PATH)

    with open(TOKENIZER_PATH, "w", encoding="utf-8") as f:
        f.write(tokenizer.to_json())

    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump(le.classes_.tolist(), f)

    print("✅ Training complete.")
    print(f"Saved: {MODEL_PATH}, {TOKENIZER_PATH}, {LABELS_PATH}")

if __name__ == "__main__":
    main()
