import numpy as np
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForMaskedLM
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import torch

# Load the tokenizer and the quantized model
tokenizer = AutoTokenizer.from_pretrained("InstaDeepAI/nucleotide-transformer-500m-human-ref")
quantized_model = torch.quantization.quantize_dynamic(
    AutoModelForMaskedLM.from_pretrained("InstaDeepAI/nucleotide-transformer-500m-human-ref"),
    {torch.nn.Linear},
    dtype=torch.qint8
)

# Load the dataset
dataset_path = 'data/modified_dataset'
dataset = load_from_disk(dataset_path)

# Access the 'train' and 'test' splits from the dataset
train_sequences = dataset["train"]["sequence"]
train_labels = dataset["train"]["label"]
test_sequences = dataset["test"]["sequence"]
test_labels = dataset["test"]["label"]

# Determine the number of samples for 0.6% of the data
train_samples = int(len(train_sequences) * 0.003)
test_samples = int(len(test_sequences) * 0.003)

# Randomly select 0.6% of the data for train and test splits
train_indices = np.random.choice(len(train_sequences), train_samples, replace=False)
test_indices = np.random.choice(len(test_sequences), test_samples, replace=False)

# Select the samples using the indices
train_sequences = [train_sequences[i] for i in train_indices]
train_labels = [train_labels[i] for i in train_indices]

test_sequences = [test_sequences[i] for i in test_indices]
test_labels = [test_labels[i] for i in test_indices]

def get_embeddings(sequences, batch_size, tokenizer, quantized_model):
    max_length = tokenizer.model_max_length
    num_sequences = len(sequences)
    num_batches = (num_sequences + batch_size - 1) // batch_size

    all_embeddings = []
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_sequences)
        batch_sequences = sequences[start_idx:end_idx]

        tokens_ids = tokenizer.batch_encode_plus(
            batch_sequences,
            return_tensors="pt",
            padding="max_length",
            max_length=max_length
        )["input_ids"]

        # Embeddings generation using quantized model
        attention_mask = tokens_ids != tokenizer.pad_token_id
        torch_outs = quantized_model(
            tokens_ids,
            attention_mask=attention_mask,
            encoder_attention_mask=attention_mask,
            output_hidden_states=True
        )
        embeddings = torch_outs['hidden_states'][-1].detach().numpy()

        # Mean sequence embeddings calculation
        attention_mask = torch.unsqueeze(attention_mask, dim=-1)
        mean_sequence_embeddings = torch.sum(attention_mask * embeddings, axis=-2) / torch.sum(attention_mask, axis=1)

        all_embeddings.append(mean_sequence_embeddings)

    # Concatenate embeddings from all batches
    mean_sequence_embeddings = np.concatenate(all_embeddings, axis=0)

    return mean_sequence_embeddings

# Define batch size
batch_size = 6

# Tokenization and Embedding generation for train and test sequences
train_embeddings = get_embeddings(train_sequences, batch_size, tokenizer, quantized_model)
test_embeddings = get_embeddings(test_sequences, batch_size, tokenizer, quantized_model)

# Convert tensors to numpy arrays
X_train, y_train = train_embeddings, train_labels
X_test, y_test = test_embeddings, test_labels

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Training the classifier with increased max_iter
classifier = LogisticRegression(random_state=42, max_iter=1000)  # Increased max_iter
classifier.fit(X_train, y_train)

# Predicting on the evaluation set
y_pred = classifier.predict(X_test)

# Evaluating the classifier
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, zero_division=0)  # Set zero_division to 0

def classify_sequence(sequence):
    embedding = get_embeddings([sequence], 1, tokenizer, quantized_model)
    embedding = scaler.transform(embedding)
    prediction = classifier.predict(embedding)
    return prediction[0], accuracy

def get_model_accuracy():
    return accuracy, report
