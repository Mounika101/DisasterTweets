!pip install transformers
!pip install keras
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import BertTokenizer
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, LSTM, Dense, GlobalMaxPooling1D
data = pd.read_csv('dataset1.csv')
print(data)
data['text'] = data['text'].str.replace('[^\w\s]','') 
data['text'] = data['text'].str.replace('\d+','')  
data['text'] = data['text'].str.strip()
train_data, test_data = train_test_split(data, test_size=0.1, random_state=42)
print("Training Data:")
print(train_data)

print("Testing Data:")
print(test_data)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
!pip install torch
import torch
def bert_encode(texts, tokenizer, max_len=512):
    input_ids = []
    attention_masks = []

    for text in texts:
        encoded = tokenizer.encode_plus(
            text, 
            add_special_tokens=True,
            max_length=max_len,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])
    
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    
    return input_ids, attention_masks
    
train_input_ids, train_attention_masks = bert_encode(train_data['text'].values, tokenizer)
test_input_ids, test_attention_masks = bert_encode(test_data['text'].values, tokenizer)
model = Sequential()
model.add(Embedding(input_dim=tokenizer.vocab_size, output_dim=100, input_length=1024))
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(LSTM(units=64, return_sequences=True))
model.add(GlobalMaxPooling1D())
model.add(Dense(units=1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
import tensorflow as tf
from tensorflow.keras.layers import Concatenate

train_input_ids_tf = tf.convert_to_tensor(train_input_ids.numpy())
train_attention_masks_tf = tf.convert_to_tensor(train_attention_masks.numpy())

concat_layer = Concatenate()([train_input_ids_tf, train_attention_masks_tf])
model.fit(
    concat_layer.numpy(),
    train_data['target'].values,
    epochs=5,
    batch_size=32
)
test_input_ids_tf = tf.convert_to_tensor(test_input_ids.numpy())
test_attention_masks_tf = tf.convert_to_tensor(test_attention_masks.numpy())

concat_layer = Concatenate()([test_input_ids_tf, test_attention_masks_tf])
test_pred = model.predict(concat_layer.numpy())
test_pred = np.round(test_pred).flatten()
test_true = test_data['target'].values
accuracy = accuracy_score(test_true, test_pred)
precision = precision_score(test_true, test_pred)
recall = recall_score(test_true, test_pred)
f1 = f1_score(test_true, test_pred)
print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
print('F1 Score:', f1)
