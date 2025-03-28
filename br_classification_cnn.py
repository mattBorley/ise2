# ===============================
# Hyperparameters and Configuration
# ===============================

# Define the dataset
project = 'caffe'  # Choose the project (options: 'pytorch', 'tensorflow', 'keras', 'incubator-mxnet', 'caffe')
path = f'datasets/{project}.csv'

# Define sequence length and other text processing parameters
max_sequence_length = 100  # Max sequence length for padding (adjustable)

# Tokenizer and Embedding parameters
vocab_size = 10000  # Vocabulary size (number of unique words)
embedding_dim = 100  # GloVe embedding dimension (must match the GloVe file used)
# Set False to use pre-trained embeddings without training
embedding_trainable = True                  #known

# Convolutional Layer parameters
num_filters = 128                       #test
kernel_size = 3                        #test
activation_conv = 'relu'

3819402756#known

# MaxPooling Layer parameters
pool_size = 2                               #known

# Dense Layer parameters
dense_units = 32                        #test
dropout_rate = 0.5                     #test

# Output Layer parameters
num_classes = 2  # Now two classes (bug vs non-bug)
output_activation = 'softmax'  # Changed to softmax for multi-class classification

# Model compilation parameters
learning_rate = 0.0005                   #test
optimizer = 'adam'                          #known
loss_function = 'binary_crossentropy'       #known

# Training parameters
epochs = 20                             #test
batch_size = 16                         #test

pre_embedding = False  # Set this to True if you want to use pre-trained GloVe embeddings, False for training from scratch


# Threshold for classification
classification_threshold = 0.5          #test

# =====================0.6040
# ==========
# Libraries and Dependencies
# ===============================
import pandas as pd
import numpy as np
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight

nltk.download('stopwords')

# ===============================
# Text Preprocessing Functions
# ===============================
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

stop_words = set(stopwords.words('english'))

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
# Load GloVe Embeddings (if pre_embedding is True)
# ===============================

if pre_embedding:
    glove_path = 'glove.6B.100d.txt'  # Update this if using a different dimension
    embeddings_index = {}

    with open(glove_path, encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    print(f"Loaded {len(embeddings_index)} word vectors from GloVe.")

    # Create an embedding matrix
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    for word, i in tokenizer.word_index.items():
        if i < vocab_size:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
else:
    # If not using pre-trained embeddings, initialize embedding matrix randomly
    embedding_matrix = None  # No pre-trained weights, will train from scratch

# ===============================
# CNN Model Architecture
# ===============================
model = Sequential([
    Embedding(input_dim=vocab_size,
              output_dim=embedding_dim,
              weights=[embedding_matrix] if pre_embedding else None,  # Use pre-trained GloVe if pre_embedding is True
              input_length=max_sequence_length,
              trainable=embedding_trainable if pre_embedding else True),  # Train the embeddings if learning from scratch
    Conv1D(filters=num_filters, kernel_size=kernel_size, activation=activation_conv),
    MaxPooling1D(pool_size=pool_size),
    Flatten(),
    Dense(dense_units, activation=activation_conv),
    Dropout(dropout_rate),
    Dense(num_classes, activation=output_activation)  # Using softmax for multi-class classification
])

model.compile(optimizer=Adam(learning_rate=learning_rate), loss=loss_function, metrics=['accuracy'])

model.summary()

# ===============================
# Compute class weights
# ===============================
class_weights = compute_class_weight('balanced', classes=np.unique(y_train.argmax(axis=1)), y=y_train.argmax(axis=1))
class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}

# ===============================
# Train the CNN Model
# ===============================
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, class_weight=class_weight_dict, verbose=1)

# ===============================
# Evaluate the Model
# ===============================
results = model.evaluate(X_test, y_test, verbose=1)
print(f"Test Results: {dict(zip(model.metrics_names, results))}")

# Get predictions and apply threshold
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)  # Get the class with the highest probability

# Compute classification metrics
accuracy = accuracy_score(y_test.argmax(axis=1), y_pred_classes)
precision = precision_score(y_test.argmax(axis=1), y_pred_classes, average='macro')
recall = recall_score(y_test.argmax(axis=1), y_pred_classes, average='macro')
f1 = f1_score(y_test.argmax(axis=1), y_pred_classes, average='macro')

# Print results
print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")