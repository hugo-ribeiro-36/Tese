import subprocess
import pandas as pd
import numpy as np
import torch
import tensorflow as tf
from sklearn.model_selection import GroupKFold
from tqdm import tqdm
import tensorflow.keras.backend as K
from transformers import DistilBertTokenizer, TFDistilBertModel, DistilBertConfig
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import csv

def run(command):
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    out, err = process.communicate()
    print(out.decode('utf-8').strip())

print('# CPU')
run('cat /proc/cpuinfo | egrep -m 1 "^model name"')
run('cat /proc/cpuinfo | egrep -m 1 "^cpu MHz"')
run('cat /proc/cpuinfo | egrep -m 1 "^cpu cores"')

print('# RAM')
run('cat /proc/meminfo | egrep "^MemTotal"')

print('# GPU')
run('lspci | grep VGA')

print('# OS')
run('uname -a')

print(tf.__version__)

MODEL_TYPE = 'distilbert-base-uncased'  # Using a smaller model to save memory
MAX_SIZE = 300
BATCH_SIZE = 8  # Further reduced batch size
MAX_SEQUENCE_LENGTH = 256 # Further reduced sequence length

tokenizer = DistilBertTokenizer.from_pretrained(MODEL_TYPE)

# Read and prepare datasets
df_train = pd.read_csv('path_to_updated_csv_file.csv')
df_train = df_train[['text', 'humor_rating']]  # Assuming humor_rating is the column for regression
X = df_train['text'].values
Y = df_train['humor_rating'].values

df_test = pd.read_csv("Codalab-test-dataset.csv")
df_test = df_test[['text']]

def _convert_to_transformer_inputs(text, tokenizer, max_sequence_length):
    """Converts tokenized input to ids, masks, and segments for transformer (including BERT)"""
    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_sequence_length,
        padding='max_length',
        truncation=True
    )
    input_ids = inputs['input_ids']
    input_masks = inputs['attention_mask']
    return [input_ids, input_masks]

def compute_input_arrays(df, tokenizer, max_sequence_length):
    input_ids, input_masks = [], []
    for text in tqdm(df['text']):
        ids, masks = _convert_to_transformer_inputs(text, tokenizer, max_sequence_length)
        input_ids.append(ids)
        input_masks.append(masks)
    return [np.asarray(input_ids, dtype=np.int32),
            np.asarray(input_masks, dtype=np.int32)]

def compute_output_arrays(df, column):
    return np.asarray(df[column])

inputs = compute_input_arrays(df_train, tokenizer, MAX_SEQUENCE_LENGTH)
outputs = compute_output_arrays(df_train, 'humor_rating')
test_inputs = compute_input_arrays(df_test, tokenizer, MAX_SEQUENCE_LENGTH)

def create_model():
    input_ids = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
    input_masks = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)

    config = DistilBertConfig.from_pretrained(MODEL_TYPE)
    config.output_hidden_states = False
    bert_model = TFDistilBertModel.from_pretrained(MODEL_TYPE, config=config)

    sequence_output = bert_model(input_ids, attention_mask=input_masks)[0]
    pooled_output = tf.keras.layers.GlobalAveragePooling1D()(sequence_output)
    
    x = tf.keras.layers.Dropout(0.2)(pooled_output)
    x = tf.keras.layers.Dense(1, activation='linear')(x)

    model = tf.keras.models.Model(inputs=[input_ids, input_masks], outputs=x)
    return model

def print_evaluation_metrics(y_true, y_pred):
    print('mean_absolute_error:', mean_absolute_error(y_true, y_pred))
    print('mean_squared_error:', mean_squared_error(y_true, y_pred))
    print('r2_score:', r2_score(y_true, y_pred))

min_mse = float('inf')
best_model = None

for LR in [1e-5]:
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    print('LR=', LR)
    gkf = GroupKFold(n_splits=5).split(X=df_train.text, groups=df_train.text)

    for fold, (train_idx, valid_idx) in enumerate(gkf):
        if fold not in range(1):  # Run only one fold for simplicity
            continue
        train_inputs = [inputs[i][train_idx] for i in range(len(inputs))]
        train_outputs = outputs[train_idx]

        valid_inputs = [inputs[i][valid_idx] for i in range(len(inputs))]
        valid_outputs = outputs[valid_idx]

        K.clear_session()
        model = create_model()
        optimizer = tf.keras.optimizers.Adam(learning_rate=LR)
        model.compile(loss='mean_squared_error', optimizer=optimizer)
        
        model.fit(train_inputs, train_outputs, epochs=3, batch_size=BATCH_SIZE, verbose=1)  # Reduced epochs

        valid_preds = model.predict(valid_inputs)
        mse = mean_squared_error(valid_outputs, valid_preds)
        print_evaluation_metrics(valid_outputs, valid_preds)
        if mse < min_mse:
            min_mse = mse
            best_model = model
    
        

best_model.save('best_model.h5')

print('Best model MSE:', min_mse)
best_model.summary()


test_preds = best_model.predict(test_inputs)

df_sub = df_test.copy()
df_sub['humor_rating'] = test_preds
df_sub.to_csv('colbert_test_predictions.csv', index=False)

print(df_sub.head())

input_json_file = "Joke_Classifier.csv"
jokes = []

with open(input_json_file, 'r') as csvfile:
    datareader = csv.reader(csvfile)
    for row in datareader:
        jokes.append(row[0])
        jokes.append(row[1])

jokes_df = pd.DataFrame({'text': jokes})

# Tokenize jokes
joke_inputs = compute_input_arrays(jokes_df, tokenizer, MAX_SEQUENCE_LENGTH)

# Predict humor ratings for jokes
joke_preds = best_model.predict(joke_inputs)

# Print predictions
for i, joke in enumerate(jokes):
    print(f"Joke: {joke}")
    print(f"Humor Rating: {joke_preds[i][0]}")
    print('-----------------')