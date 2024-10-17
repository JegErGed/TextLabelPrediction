import os
import pandas as pd
import json
import torch
from transformers import MarianMTModel, MarianTokenizer
from tqdm import tqdm

# Load the dataset from the specified path
def load_data(data_path):
    data = []
    annotations_path = os.path.join(data_path, 'annotations')
    
    with tqdm(total=len(os.listdir(annotations_path)), desc='Loading Data', unit='file') as pbar:
        for annotation_file in os.listdir(annotations_path):
            if annotation_file.endswith('.json'):
                with open(os.path.join(annotations_path, annotation_file), 'r', encoding='utf-8') as f:
                    annotation_data = json.load(f)

                    for form_entry in annotation_data.get('form', []):
                        text = form_entry.get('text', '')
                        label = form_entry.get('label', '')
                        data.append({'text': text, 'label': label, 'image_name': annotation_file.replace('.json', '.png')})
                    
                    pbar.update(1)

    return pd.DataFrame(data)

# Translation model (English <-> French)
def get_translation_models(device):
    en_fr_model_name = 'Helsinki-NLP/opus-mt-en-fr'
    fr_en_model_name = 'Helsinki-NLP/opus-mt-fr-en'

    en_fr_tokenizer = MarianTokenizer.from_pretrained(en_fr_model_name)
    en_fr_model = MarianMTModel.from_pretrained(en_fr_model_name).to(device)

    fr_en_tokenizer = MarianTokenizer.from_pretrained(fr_en_model_name)
    fr_en_model = MarianMTModel.from_pretrained(fr_en_model_name).to(device)

    return en_fr_tokenizer, en_fr_model, fr_en_tokenizer, fr_en_model

# Back-translation function
def back_translate(text, en_fr_tokenizer, en_fr_model, fr_en_tokenizer, fr_en_model, device):
    # Move input to the device
    fr_tokens = en_fr_tokenizer(text, return_tensors="pt", padding=True).to(device)
    fr_translation = en_fr_model.generate(**fr_tokens)
    fr_text = en_fr_tokenizer.batch_decode(fr_translation, skip_special_tokens=True)[0]

    en_tokens = fr_en_tokenizer(fr_text, return_tensors="pt", padding=True).to(device)
    en_translation = fr_en_model.generate(**en_tokens)
    back_translated_text = fr_en_tokenizer.batch_decode(en_translation, skip_special_tokens=True)[0]

    return back_translated_text

# Data loading with augmentation
def load_data_with_augmentation(data_path, en_fr_tokenizer, en_fr_model, fr_en_tokenizer, fr_en_model, augment_classes=[], augment_factor=1.5, output_csv='processed_dataset/augmented_data.csv'):
    annotations_path = os.path.join(data_path, 'annotations')
    data = []
    
    with tqdm(total=len(os.listdir(annotations_path)), desc='Processing and Augmenting', unit='file') as pbar:
        for annotation_file in os.listdir(annotations_path):
            if annotation_file.endswith('.json'):
                with open(os.path.join(annotations_path, annotation_file), 'r', encoding='utf-8') as f:
                    annotation_data = json.load(f)

                    for form_entry in annotation_data.get('form', []):
                        text = form_entry.get('text', '')
                        label = form_entry.get('label', '')
                        data.append({'text': text, 'label': label, 'image_name': annotation_file.replace('.json', '.png')})
                        
                        if label in augment_classes:
                            for _ in range(int(augment_factor)):
                                augmented_text = back_translate(text, en_fr_tokenizer, en_fr_model, fr_en_tokenizer, fr_en_model, device)
                                data.append({'text': augmented_text, 'label': label, 'image_name': annotation_file.replace('.json', '.png')})

                    pbar.update(1)

    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False, encoding='utf-8')
    print(f"Augmented data saved to {output_csv}")

    return df

# Main function to run the augmentation
if __name__ == "__main__":
    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"We are using {device} for augmenting.")

    # Load translation models
    en_fr_tokenizer, en_fr_model, fr_en_tokenizer, fr_en_model = get_translation_models(device)

    # Load training data
    train_data = load_data('dataset/training_data')
    class_counts = train_data['label'].value_counts()
    median_count = class_counts.median()
    augment_classes = class_counts[class_counts < median_count].index.tolist()

    # Augment the training data
    train_data_augmented = load_data_with_augmentation('dataset/training_data', en_fr_tokenizer, en_fr_model, fr_en_tokenizer, fr_en_model, augment_classes=augment_classes)

    # Save test data to CSV
    test_data = load_data('dataset/testing_data')
    test_data.to_csv('processed_dataset/test_data.csv', index=False, encoding='utf-8')
    print("Test data saved to processed_dataset/test_data.csv")
