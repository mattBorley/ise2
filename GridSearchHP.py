import pandas as pd
import numpy as np
import re

from keras.src.layers import average
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight
from itertools import product

# ===============================
# Hyperparameters and Configuration
# ===============================

# Define the dataset
project = 'tensorflow'  # Choose the project (options: 'pytorch', 'tensorflow', 'keras', 'incubator-mxnet', 'caffe')
path = f'datasets/{project}.csv'

# Define sequence length and other text processing parameters
max_sequence_length = 100  # Max sequence length for padding (adjustable)

# Tokenizer and Embedding parameters
vocab_size = 10000  # Vocabulary size (number of unique words)
embedding_dim = 100  # GloVe embedding dimension (must match the GloVe file used)
embedding_trainable = True  # Use pre-trained embeddings without training (True/False)

# Convolutional Layer parameters (testing these parameters)
num_filters_options = [32, 64, 128, 256, 512]  # Vary number of filters
kernel_size_options = [3, 5, 7]  # Vary kernel size

# Dense Layer parameters (testing these parameters)
dense_units_options = [32, 64, 128]  # Vary dense layer units
dropout_rate_options = [0.3, 0.5, 0.7]  # Vary dropout rate

# Model Training parameters (testing these parameters)
epochs_options = [8, 10, 20, 30, 50]  # Vary epochs
batch_size_options = [16, 32, 64]  # Vary batch size

# Learning Rate (testing these parameters)
learning_rate_options = [0.0001, 0.0005, 0.001, 0.005, 0.01]  # Vary learning rate

# Classification Threshold (testing these parameters)
classification_threshold_options = [0.5, 0.6, 0.7]  # Vary classification threshold

# ===============================
# Libraries and Dependencies
# ===============================
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


def remove_html(text):
    return re.sub(r'<.*?>', '', text)


def remove_emoji(text):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"
                               u"\U0001F300-\U0001F5FF"
                               u"\U0001F680-\U0001F6FF"
                               u"\U0001F1E0-\U0001F1FF"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


def remove_stopwords(text):
    return " ".join([word for word in text.split() if word.lower() not in stop_words])


def clean_str(text):
    text = re.sub(r"[^A-Za-z0-9(),.!?\'\`]", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip().lower()


# ===============================
# Data Loading and Preprocessing
# ===============================
pd_all = pd.read_csv(path)
pd_all = pd_all.sample(frac=1, random_state=999)

pd_all['Title+Body'] = pd_all.apply(
    lambda row: row['Title'] + '. ' + row['Body'] if pd.notna(row['Body']) else row['Title'],
    axis=1
)

data = pd_all.rename(columns={"class": "sentiment", "Title+Body": "text"})
data = data[['sentiment', 'text']].fillna('')

train_text, test_text, y_train, y_test = train_test_split(
    data['text'], data['sentiment'], test_size=0.3, random_state=999
)

X_train, X_val, y_train, y_val = train_test_split(train_text, y_train, test_size=0.2, random_state=999)

# Preprocess text
X_train = X_train.apply(remove_html).apply(remove_emoji).apply(remove_stopwords).apply(clean_str)
X_val = X_val.apply(remove_html).apply(remove_emoji).apply(remove_stopwords).apply(clean_str)
X_test = test_text.apply(remove_html).apply(remove_emoji).apply(remove_stopwords).apply(clean_str)

# Tokenization and padding
tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(X_train)

X_train = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=max_sequence_length)
X_val = pad_sequences(tokenizer.texts_to_sequences(X_val), maxlen=max_sequence_length)
X_test = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=max_sequence_length)

# Convert labels to integer values (one-hot encode for softmax)
y_train = pd.get_dummies(y_train).values
y_val = pd.get_dummies(y_val).values
y_test = pd.get_dummies(y_test).values

# ===============================
# Compute class weights
# ===============================
class_weights = compute_class_weight('balanced', classes=np.unique(y_train.argmax(axis=1)), y=y_train.argmax(axis=1))
class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}

# ===============================
# Grid Search: Train and evaluate for each parameter combination
# ===============================
results = []

for num_filters, kernel_size, dense_units, dropout_rate, epochs, batch_size, learning_rate, classification_threshold in product(
        num_filters_options, kernel_size_options, dense_units_options, dropout_rate_options,
        epochs_options, batch_size_options, learning_rate_options, classification_threshold_options):
    # Create model
    model = Sequential([
        Embedding(input_dim=vocab_size,
                  output_dim=embedding_dim,
                  input_length=max_sequence_length,
                  trainable=embedding_trainable),
        Conv1D(filters=num_filters, kernel_size=kernel_size, activation='relu'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(dense_units, activation='relu'),
        Dropout(dropout_rate),
        Dense(2, activation='softmax')
    ])

    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size,
                        class_weight=class_weight_dict, verbose=0)

    # Evaluate the model
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    f1 = f1_score(y_test.argmax(axis=1), y_pred_classes, average='macro')

    # Save results for this configuration
    results.append({
        'num_filters': num_filters,
        'kernel_size': kernel_size,
        'dense_units': dense_units,
        'dropout_rate': dropout_rate,
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'classification_threshold': classification_threshold,
        'f1_score': f1
    })

    # Print the current configuration and F1 score
    print(f"Tested combination: num_filters={num_filters}, kernel_size={kernel_size}, dense_units={dense_units}, "
          f"dropout_rate={dropout_rate}, epochs={epochs}, batch_size={batch_size}, learning_rate={learning_rate}, "
          f"classification_threshold={classification_threshold} => F1 Score: {f1:.4f}")

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Sort by F1 score and get top 10 combinations
top_results = results_df.sort_values(by='f1_score', ascending=False).head(10)

# Save top results to CSV
top_results.to_csv('top_10_combinations.csv', index=False)

# Print top 10 combinations
print("\nTop 10 combinations based on F1 score:")
print(top_results)
