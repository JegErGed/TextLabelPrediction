import os
import pandas as pd
import json
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import matplotlib.pyplot as plt
import datetime
from sklearn.model_selection import train_test_split



# Load data function
def load_data(data_path):
    annotations_path = os.path.join(data_path, 'annotations')
    
    data = []
    for annotation_file in os.listdir(annotations_path):
        if annotation_file.endswith('.json'):
            with open(os.path.join(annotations_path, annotation_file), 'r', encoding='utf-8') as f:
                annotation_data = json.load(f)
                
                for form_entry in annotation_data.get('form', []):
                    text = form_entry.get('text', '')
                    label = form_entry.get('label', '')
                    data.append({'text': text, 'label': label, 'image_name': annotation_file.replace('.json', '.png')})
    
    return pd.DataFrame(data)

def update_optimizer(optimizer, model, learning_rate):
    # Update optimizer with newly unfrozen layers
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    return optimizer

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def preprocess_text(dataframe):
    tokens = tokenizer(list(dataframe['text']), padding=True, truncation=True, return_tensors='pt')
    return tokens

# Model Definition
class BertClassifier(nn.Module):
    def __init__(self, num_labels, freeze_bert=True):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(self.bert.config.hidden_size, num_labels)
        
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        logits = self.fc(pooled_output)
        return logits

def freeze_bert_layers(model, unfreeze_last_n_layers=2):
    # Freeze all layers first
    for param in model.bert.parameters():
        param.requires_grad = False

    # Unfreeze the last n layers
    for layer in model.bert.encoder.layer[-unfreeze_last_n_layers:]:
        for param in layer.parameters():
            param.requires_grad = True

# Main training function with gradual unfreezing
def train_model(train_loader, val_loader, model, optimizer, criterion, epochs, patience, device):
    best_val_accuracy = 0.0
    unfrozen_layers_num = 2
    patience_counter = 0
    max_layers = 12  # Max BERT layers to unfreeze
    metrics = []
    val_metrics = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        metrics.append(avg_train_loss)

        # Validation step
        model.eval()
        total_val_loss = 0
        correct_predictions = 0

        with torch.no_grad():
            for val_batch in val_loader:
                val_input_ids = val_batch[0].to(device)
                val_attention_mask = val_batch[1].to(device)
                val_labels = val_batch[2].to(device)

                val_outputs = model(val_input_ids, attention_mask=val_attention_mask)
                val_loss = criterion(val_outputs, val_labels)
                total_val_loss += val_loss.item()

                _, val_predicted = torch.max(val_outputs, dim=1)
                correct_predictions += (val_predicted == val_labels).sum().item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_accuracy = correct_predictions / len(val_loader.dataset)
        val_metrics.append((avg_val_loss, val_accuracy))

        # Check for improvement in validation accuracy
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            patience_counter = 0  # Reset patience
        else:
            patience_counter += 1  # Increase patience counter if no improvement

        # Unfreeze more layers if patience runs out
        if patience_counter >= patience and unfrozen_layers_num < max_layers:
            unfrozen_layers_num += 2
            freeze_bert_layers(model, unfreeze_last_n_layers=unfrozen_layers_num)
            optimizer = update_optimizer(optimizer, model, learning_rate)  # Update optimizer to include new layers
            patience_counter = 0
            print(f"Unfroze additional layers. Now {unfrozen_layers_num} layers are unfrozen.")

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Accuracy: {val_accuracy:.4f}")

    return metrics, val_metrics


# Set up model, optimizer, and loss
learning_rate = 1.6e-5
epochs = 16
patience = 3

# Load training and testing data
train_data = load_data('dataset/training_data')
train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)
test_data = load_data('dataset/testing_data')

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def preprocess_text(dataframe):
    # Tokenizing the text data and generating attention masks
    tokens = tokenizer(list(dataframe['text']), padding=True, truncation=True, return_tensors='pt')
    return tokens

train_tokens = preprocess_text(train_data)
val_tokens = preprocess_text(val_data)
test_tokens = preprocess_text(test_data)

# Initialize LabelEncoder and encode the labels
label_encoder = LabelEncoder()
train_labels_encoded = label_encoder.fit_transform(train_data['label'].values)
val_labels_encoded = label_encoder.transform(val_data['label'].values)
test_labels_encoded = label_encoder.transform(test_data['label'].values)

# Convert the encoded labels to torch tensors
train_labels_tensor = torch.tensor(train_labels_encoded, dtype=torch.long)
val_labels_tensor = torch.tensor(val_labels_encoded, dtype=torch.long)
test_labels_tensor = torch.tensor(test_labels_encoded, dtype=torch.long)

# Create TensorDataset including attention masks
train_dataset = TensorDataset(train_tokens['input_ids'], train_tokens['attention_mask'], train_labels_tensor)
val_dataset = TensorDataset(val_tokens['input_ids'], val_tokens['attention_mask'], val_labels_tensor)
test_dataset = TensorDataset(test_tokens['input_ids'], test_tokens['attention_mask'], test_labels_tensor)


train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, pin_memory=True)


# Define the number of unique labels
num_labels = len(label_encoder.classes_)

model = BertClassifier(num_labels=len(label_encoder.classes_))
freeze_bert_layers(model=model, unfreeze_last_n_layers=2)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# Device check
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Train the model
metrics, val_metrics = train_model(train_loader, val_loader, model, optimizer, criterion, epochs, patience, device)

# Save model
cur_datetime = str(datetime.datetime.now()).replace(" ","_").replace(":", "-")[0:19]

# Plotting Loss over each epoch
plt.figure(figsize=(10,5))
plt.plot(range(1, epochs + 1), metrics, marker='o', label='Train Loss')
plt.plot(range(1, epochs + 1), [x[0] for x in val_metrics], marker='o', label='Validation Loss')
plt.title(f'Loss over each epoch. Layers are unfrozen gradually')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig(f'Metrics_{f"{cur_datetime}"}.png')
plt.show()



model.to(device)

# Ensure the model is in evaluation mode
model.eval()

# Disable gradient calculations for validation
with torch.no_grad():
    # Create lists to store true labels and predictions
    all_labels = []
    all_predictions = []

    # Iterate through the test data in batches
    for batch in test_loader:
        input_ids = batch[0].to(device)  # Move input_ids to the correct device (GPU or CPU)
        attention_mask = batch[1].to(device)  # Move attention_mask to the correct device
        labels = batch[2].to(device)  # Move labels to the correct device

        # Get model predictions, pass both input_ids and attention_mask
        outputs = model(input_ids, attention_mask=attention_mask)
        _, predicted = torch.max(outputs, dim=1)  # Get the predicted classes

        # Append true labels and predictions to the lists
        all_labels.extend(labels.cpu().numpy())  # Move back to CPU for evaluation
        all_predictions.extend(predicted.cpu().numpy())  # Move back to CPU for evaluation



# Convert lists to numpy arrays for evaluation
all_labels = np.array(all_labels)
all_predictions = np.array(all_predictions)

# Calculate and print metrics
accuracy = accuracy_score(all_labels, all_predictions)
print(f'Test Accuracy: {accuracy:.4f}')

# Print a classification report for more detailed metrics
print(classification_report(all_labels, all_predictions, target_names=label_encoder.classes_))



# Visualizing a few predictions
num_samples = 10
sample_indices = np.random.choice(len(all_labels), num_samples, replace=False)

plt.figure(figsize=(15, 5))
for i, idx in enumerate(sample_indices):
    plt.subplot(2, num_samples // 2, i + 1)
    plt.title(f'Pred: {label_encoder.inverse_transform([all_predictions[idx]])[0]}\n'
          f'True: {label_encoder.inverse_transform([all_labels[idx]])[0]}')
    plt.axis('off')
plt.tight_layout()
plt.show()