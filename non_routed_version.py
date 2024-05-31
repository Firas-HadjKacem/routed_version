import numpy as np
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForMaskedLM
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import torch

# Define the get_embeddings function
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

# Load the tokenizer and the quantized model
tokenizer = AutoTokenizer.from_pretrained("InstaDeepAI/nucleotide-transformer-500m-human-ref")
quantized_model = torch.quantization.quantize_dynamic(
    AutoModelForMaskedLM.from_pretrained("InstaDeepAI/nucleotide-transformer-500m-human-ref"),
    {torch.nn.Linear},
    dtype=torch.qint8
)

# Load your dataset
dataset_path = r"C:\Users\USER\Downloads\InstaDeep Models\modified_dataset"
dataset = load_from_disk(dataset_path)

# Access the 'train' and 'test' splits from the dataset
train_sequences = dataset["train"]["sequence"]
train_labels = dataset["train"]["label"]

test_sequences = dataset["test"]["sequence"]
test_labels = dataset["test"]["label"]

# Determine the number of samples for 5% of the data
train_samples = int(len(train_sequences) * 0.006)
test_samples = int(len(test_sequences) * 0.006)

# Randomly select 5% of the data for train and test splits
train_indices = np.random.choice(len(train_sequences), train_samples, replace=False)
test_indices = np.random.choice(len(test_sequences), test_samples, replace=False)

# Select the samples using the indices
train_sequences = [train_sequences[i] for i in train_indices]
train_labels = [train_labels[i] for i in train_indices]

test_sequences = [test_sequences[i] for i in test_indices]
test_labels = [test_labels[i] for i in test_indices]

# Display the number of rows in the train and test splits
print("Number of rows in train split:", len(train_sequences))
print("Number of rows in test split:", len(test_sequences))

# print(train_sequences[:5])
# Define batch size
batch_size = 6

# Tokenization and Embedding generation for train sequences
train_embeddings = get_embeddings(train_sequences, batch_size, tokenizer, quantized_model)

# Tokenization and Embedding generation for test sequences
test_embeddings = get_embeddings(test_sequences, batch_size, tokenizer, quantized_model)

# Convert tensors to numpy arrays
X_train, y_train = train_embeddings, train_labels
X_test, y_test = test_embeddings, test_labels

# Training the classifier
classifier = LogisticRegression(random_state=42)
classifier.fit(X_train, y_train)

# Predicting on the evaluation set
y_pred = classifier.predict(X_test)

print(X_test,y_pred)
# Evaluating the classifier
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
