model_name = "title_BERT"
num_epochs = 5

import itertools
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader
import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import recall_score

class TitlesDataset(Dataset):
    def __init__(self, examples, tokenizer):
        self.examples = []
        for id in examples:
            example = examples[id]
            if example["answer"] == "A":
                example["answer"] = "B"
            self.examples.append(example)
        
        self.tokenizer = tokenizer
        self.label_map = {"B": 0, "C": 1}

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        prompt = example["title"]
        label = example["answer"]

        # Tokenize the title and label
        inputs = self.tokenizer.encode_plus(
            prompt,
            add_special_tokens=True,
            padding="max_length",
            max_length=128,
            truncation=True,
            return_tensors="pt"
        )

        input_ids = inputs["input_ids"].squeeze()
        attention_mask = inputs["attention_mask"].squeeze()
        label_id = self.label_map[label]

        return input_ids, attention_mask, torch.tensor(label_id)

def collate_fn(batch):
    input_ids, attention_masks, labels = zip(*batch)
    input_ids = pad_sequence(input_ids, batch_first=True)
    attention_masks = pad_sequence(attention_masks, batch_first=True)
    labels = torch.stack(labels)

    return input_ids, attention_masks, labels

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(F.softmax(inputs, dim=-1), targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        return focal_loss.mean()

def grid_search():
#load datasets
    with open("train_dict.json", 'r', encoding='utf8') as file:
        train_dict = json.load(file)
    with open("test_dict.json", 'r', encoding='utf8') as file:
        test_dict = json.load(file)

    # Define the hyperparameter values to search
    alpha_values = [1, 10, 20, 30, 40]
    gamma_values = [0.5, 0.75, 1, 2, 3]

    # Generate all combinations of hyperparameter values
    hyperparameter_combinations = list(itertools.product(alpha_values, gamma_values))

    # Perform grid search
    max_recall = 0
    best_hyperparameters = None

    for alpha, gamma in hyperparameter_combinations:
        print(f"Testing hyperparameters: alpha={alpha}, gamma={gamma}")

        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

        dataset = TitlesDataset(train_dict, tokenizer)
        data_loader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        criterion = FocalLoss(alpha=alpha, gamma=gamma)

        validation_dataset = TitlesDataset(test_dict, tokenizer)
        validation_data_loader = DataLoader(validation_dataset, batch_size=16, collate_fn=collate_fn)

        patience = 2  # Number of epochs to wait for improvement in recall
        early_stopping_counter = 0  # Counter to keep track of epochs without improvement
        prev_recall = 0
        for epoch in range(num_epochs):
            total_loss = 0

            for inputs, attention_mask, labels in tqdm(data_loader):
                inputs = inputs.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)

                outputs = model(inputs, attention_mask=attention_mask)
                loss = criterion(outputs.logits, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(data_loader)
            print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")

            # Calculate recall on validation data
            model.eval()
            true_labels = []
            predicted_labels = []

            for inputs, attention_mask, labels in tqdm(validation_data_loader):
                inputs = inputs.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)

                with torch.no_grad():
                    outputs = model(inputs, attention_mask=attention_mask)
                    _, predicted = torch.max(outputs.logits, dim=1)

                    true_labels.extend(labels.tolist())
                    predicted_labels.extend(predicted.tolist())

            recall = recall_score(true_labels, predicted_labels, pos_label=0)
            print(f"Epoch {epoch+1} - Validation Recall: {recall:.4f}")

            # Check if recall improved
            if recall > prev_recall:
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1

            # Check if early stopping criteria is met
            if early_stopping_counter >= patience:
                print("Early stopping triggered. No improvement in recall.")
                break
            prev_recall = recall
        # Update best hyperparameters if current recall is better
        if max(recall, prev_recall) > max_recall:
            max_recall = max(recall, prev_recall)
            best_hyperparameters = (alpha, gamma)
            best_model_name = f"best_{model_name}"
            model.save_pretrained(best_model_name)


    print("Grid search completed.")
    print(f"Best hyperparameters: alpha={best_hyperparameters[0]}, gamma={best_hyperparameters[1]}")
    print(f"Best Recall: {max_recall:.4f}")


if __name__ == "__main__":
    # Main function to classify titles
    grid_search()

