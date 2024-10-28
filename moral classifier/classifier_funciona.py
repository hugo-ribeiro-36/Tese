import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import torch
import seaborn as sns
import transformers
import json
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaModel, RobertaTokenizer
from transformers import BertTokenizer, BertModel

import logging
import matplotlib.pyplot as plt
import seaborn as sns
logging.basicConfig(level=logging.ERROR)

from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'
import gc
torch.cuda.empty_cache()
gc.collect()


# Load multi-hot encoded data from CSV
data = pd.read_csv('.csv')

MAX_LEN = 256
TRAIN_BATCH_SIZE = 4
VALID_BATCH_SIZE = 8
LEARNING_RATE = 1e-05
# tokenizer =RobertaTokenizer.from_pretrained('roberta-base', truncation=True, do_lower_case=True)
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
# Define dataset class
class MoralFoundationData(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe['text']  # Assuming 'Tweet' is the column name for tweets
        # print(self.text)
        self.labels = dataframe.drop(columns=['text'])  # Drop the 'Tweet' column to get labels
        # print(self.labels)
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = str(self.text[index])
        labels = torch.tensor(self.labels.iloc[index].values, dtype=torch.float)

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True,
            truncation=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        return {
            'text': text,
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'labels': labels
        }

# Split data into train and test sets
train_data, test_data = train_test_split(data, test_size=0.1, random_state=42)

train_data.reset_index(drop=True, inplace=True)
test_data.reset_index(drop=True, inplace=True)

training_set = MoralFoundationData(train_data, tokenizer, MAX_LEN)
testing_set = MoralFoundationData(test_data, tokenizer, MAX_LEN)

# DataLoaders
train_params = {'batch_size': TRAIN_BATCH_SIZE, 'shuffle': True, 'num_workers': 0}
test_params = {'batch_size': VALID_BATCH_SIZE, 'shuffle': False, 'num_workers': 0}

training_loader = DataLoader(training_set, **train_params)
testing_loader = DataLoader(testing_set, **test_params)

# Define the model
class RobertaClass(torch.nn.Module):
    def __init__(self):
        super(RobertaClass, self).__init__()
        self.l1 = BertModel.from_pretrained("bert-large-uncased")
        # self.l1 = RobertaModel.from_pretrained("roberta-base")
        self.pre_classifier = torch.nn.Linear(1024, 1024)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(1024, len(train_data.columns) - 1)  # Output size based on number of labels

    def forward(self, input_ids, attention_mask, token_type_ids):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output

model = RobertaClass()
model.to(device)

loss_function = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

# Training function
def train(epoch):
    model.train()
    tr_loss = 0
    for _, data in tqdm(enumerate(training_loader, 0)):
        ids = data['ids'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
        labels = data['labels'].to(device, dtype=torch.float)

        optimizer.zero_grad()
        outputs = model(ids, mask, token_type_ids)

        loss = loss_function(outputs, labels)
        tr_loss += loss.item()

        loss.backward()
        optimizer.step()

    print(f'Training Loss Epoch: {tr_loss/len(training_loader)}')

# Evaluation function
def evaluate(model, testing_loader):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for _, data in tqdm(enumerate(testing_loader, 0)):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            labels = data['labels'].to(device, dtype=torch.float)

            outputs = model(ids, mask, token_type_ids)
            loss = loss_function(outputs, labels)
            val_loss += loss.item()

    print(f'Validation Loss: {val_loss/len(testing_loader)}')


def evaluate2(model, testing_loader):
    model.eval()
    val_loss = 0
    all_pred = []
    all_labels = []
    with torch.no_grad():
        for _, data in tqdm(enumerate(testing_loader, 0)):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            labels = data['labels'].to(device, dtype=torch.float)

            outputs = model(ids, mask, token_type_ids)
            loss = loss_function(outputs, labels)
            val_loss += loss.item()

            # Convert predictions and labels to numpy arrays
            preds = torch.sigmoid(outputs).cpu().numpy()
            preds[preds >= 0.5] = 1
            preds[preds < 0.5] = 0
            all_pred.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    all_pred = np.array(all_pred)
    all_labels = np.array(all_labels)

    # Calculate evaluation metrics
    accuracy = accuracy_score(all_labels, all_pred)
    hamming_loss = np.mean(np.not_equal(all_labels, all_pred))
    classification_rep = classification_report(all_labels, all_pred)
    confusion_mat = confusion_matrix(all_labels.flatten(), all_pred.flatten())

    print(f'Validation Loss: {val_loss/len(testing_loader)}')
    print(f'Accuracy: {accuracy}')
    print(f'Hamming Loss: {hamming_loss}')
    print(f'Classification Report:\n{classification_rep}')
    print(f'Confusion Matrix:\n{confusion_mat}')
    return val_loss/len(testing_loader)

# Training loop
EPOCHS = 3
best_val_loss = float('inf')
for epoch in range(EPOCHS):
    train(epoch)
    val_loss = evaluate2(model, testing_loader)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss
        }, 'bert_model.pth')

    

output_model_file = 'pytorch_berta_sentiment'
output_vocab_file = './'

model_to_save = model
torch.save(model_to_save, output_model_file)
tokenizer.save_vocabulary(output_vocab_file)


print('All files saved')

print("Evaluation: \n")

evaluate2(model, testing_loader)

print("FINITO")