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
from torch.optim.lr_scheduler import ReduceLROnPlateau

torch.manual_seed(1729)
# Load data function
def load_data(data_path):
    annotations_path = os.path.join(data_path, 'annotations')
    
    data = []
    for annotation_file in os.listdir(annotations_path):
        if annotation_file.endswith('.json'):  # Ensure we're only reading JSON files
            with open(os.path.join(annotations_path, annotation_file), 'r', encoding='utf-8') as f:
                annotation_data = json.load(f)  # Load the JSON file
                
                # Iterate over each form element in the JSON
                for form_entry in annotation_data.get('form', []):
                    text = form_entry.get('text', '')  # Extract the 'text' field
                    label = form_entry.get('label', '')  # Extract the 'label' field
                    
                    # Add to data list (we can use image_name if relevant for analysis)
                    data.append({'text': text, 'label': label, 'image_name': annotation_file.replace('.json', '.png')})
    
    return pd.DataFrame(data)

# Load training and testing data
train_data = load_data('dataset/training_data')
train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)
test_data = load_data('dataset/testing_data')

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')

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

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)


# Define the number of unique labels
num_labels = len(label_encoder.classes_)

# Model Definition
class BertClassifier(nn.Module):
    def __init__(self, num_labels):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-large-uncased')
        self.fc = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # Get the pooled output for classification
        logits = self.fc(pooled_output)
        return logits

# Set up model, optimizer, and loss
learning_rate=2e-5
weight_decay = 2e-4
model = BertClassifier(num_labels)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
criterion = nn.CrossEntropyLoss()
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=2)
early_stopping_patience = 2


# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"We are currently using {device} for training.")
model.to(device)

metrics = []
val_metrics = []
auto_stopping = False  # This is for making graphs
best_val_loss = 10
epochs_without_improvement = 0 
# Training Loop
epochs = 6
for epoch in range(epochs):
    model.train()  # Set model to training mode
    total_loss = 0
    for batch in train_loader:
        input_ids = batch[0].to(device)  # input_ids
        attention_mask = batch[1].to(device)  # attention_mask
        labels = batch[2].to(device)  # labels
        # Zero gradients
        optimizer.zero_grad()
        # Forward pass
        logits = model(input_ids, attention_mask=attention_mask)
        # Compute loss
        loss = criterion(logits, labels)
        # Backward pass
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_train_loss = total_loss / len(train_loader)
    metrics.append(avg_train_loss)
    # Validation step
    model.eval()  # Set model to evaluation mode
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
    val_accuracy = correct_predictions / len(val_data)
    val_metrics.append((avg_val_loss, val_accuracy))
    print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
    scheduler.step(avg_val_loss)
    
    # Early Stopping Logic
    if avg_val_loss < best_val_loss:
        best_val_loss= avg_val_loss
        epochs_without_improvement = 0  # Reset the counter
        
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= early_stopping_patience and auto_stopping:
            print(f'Early stopping triggered. Stopping training at epoch {epoch + 1}.')
            break  # Exit the training loop

    # Save model
    cur_datetime = str(datetime.datetime.now()).replace(" ","_").replace(":", "-")[0:19]

# Plotting Loss over each epoch
plt.figure(figsize=(10,5))
plt.plot(range(1, len(metrics) + 1), metrics, marker='o', label='Train Loss')
plt.plot(range(1, len(metrics) + 1), [x[0] for x in val_metrics], marker='o', label='Validation Loss')
plt.title(f'Loss over each epoch batch_size=32, weight_decay={weight_decay}')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
if not auto_stopping:
    plt.savefig(f'Metrics_{f"{cur_datetime}"}.png')
plt.show()

# Ensure the model is on the correct device (GPU or CPU)
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