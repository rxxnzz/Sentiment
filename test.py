
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import nltk
import tensorflow as tf
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
from transformers import BertTokenizer
from transformers import TFBertForSequenceClassification


df = pd.read_csv("dataset_tweet_sentiment_pilkada_DKI_2017.csv")

df.rename(columns={
    'Sentiment': 'sentiment',
    'Pasangan Calon': 'calon',
    'Text Tweet': 'text'
}, inplace=True)

df.dropna(inplace=True)

#preprocessing
def clean_text(text):
  text = re.sub(r"https?://\S+|www\.\S+", "", text) #hapus url
  text = re.sub(r"@\S+", "", text) #hapus mention
  text = re.sub(r"#\S+", "", text) #hapus hastag
  text = re.sub(r"\d+", "", text) #hapus nomor
  text = re.sub(r"[^\w\s]", "", text) #hapus tanda baca
  text = re.sub(r"(.)\1{2,}", r"\1", text) #hapus double karakter
  text = text.strip() #hapus spasi di depan dan di belakang
  text = text.lower() #ubah menjadi huruf kecil
  return text

stopword_pilkada = pd.read_csv("stopword_tweet_pilkada_DKI_2017.csv", header=None)
stopword_pilkada.columns = ['stopword']

stop_words = set(stopwords.words('indonesian'))
additional_sw = set(stopword_pilkada.stopword.values)
stop_words = stop_words.union(additional_sw)

def remove_stopwords(text):
  word_tokens = word_tokenize(text)
  filtered_sentence = [w for w in word_tokens if not w in stop_words]
  return " ".join(filtered_sentence)

def preprocess_text(text):
  text = clean_text(text)
  text = remove_stopwords(text)
  return(text)

text_to_process = "sangat gak bagus pak ahok"
processed_text = preprocess_text(text_to_process)
print(processed_text)

df_train, df_test = train_test_split(df, test_size=0.3, random_state=42)
df_val, df_test = train_test_split(df_test, test_size=0.5, random_state=42)

print("Train Data Size: ", len(df_train)) #70%
print("Validation Data Size: ", len(df_val)) #15%
print("Test Data Size: ", len(df_test)) #15%

PRETRAINED_MODEL = "indobenchmark/indobert-base-p2"
tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL)
vocab = tokenizer.get_vocab()

#mengecek distrubusi data untuk mengetahui panjang maksimal untuk token
token_lens = []

for txt in df["text"]:
  tokens = tokenizer.encode(txt)
  token_lens.append(len(tokens))

MAX_LEN = 60

df_train['sentiment'] = df_train['sentiment'].map({'positive': 1, 'negative': 0})
df_val['sentiment'] = df_val['sentiment'].map({'positive': 1, 'negative': 0})

def encode_sentence(sent):
  return tokenizer.encode_plus(
      sent,
      add_special_tokens =True,
      padding = 'max_length',
      truncation = 'longest_first',
      max_length = MAX_LEN,
      return_attention_mask =True,
      return_token_type_ids=True
  )

def map_example_to_dict(input_ids, attention_masks, token_type_ids, label):
  return{
      "input_ids": input_ids,
      "attention_mask": attention_masks,
      "token_type_ids": token_type_ids,
  }, label

def encode_dataset(ds, limit=-1):
  input_ids_list = []
  attention_mask_list = []
  token_type_ids_list = []
  label_list = []

  for index, row in ds.iterrows():
    if limit > 0 and index >= limit:
      break

    input_ids, attention_mask, token_type_ids =encode_sentence(row["text"])["input_ids"],\
    encode_sentence(row["text"])["attention_mask"],\
    encode_sentence(row["text"])["token_type_ids"]
    label = row["sentiment"]

    input_ids_list.append(input_ids)
    attention_mask_list.append(attention_mask)
    token_type_ids_list.append(token_type_ids)
    label_list.append(label)

  return tf.data.Dataset.from_tensor_slices((
      input_ids_list,
      attention_mask_list,
      token_type_ids_list,
      label_list
  )).map(map_example_to_dict)

EPOCH = 5
BATCH_SIZE = 42
LEARNING_RATE = 1e-5

df_train_shuffled = df_train.sample(frac=1, random_state=42)
train_data = encode_dataset(df_train_shuffled).batch(BATCH_SIZE)
val_data = encode_dataset(df_val).batch(BATCH_SIZE)
test_data = encode_dataset(df_test).batch(BATCH_SIZE)

model = TFBertForSequenceClassification.from_pretrained(PRETRAINED_MODEL, num_labels=2)

model.summary()

optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

model.compile(optimizer, loss=loss, metrics=[metric])

history = model.fit(
    train_data,
    epochs=EPOCH,
    batch_size=BATCH_SIZE,
    validation_data=val_data
)


# Convert string labels to numeric format for the test dataset
df_test['sentiment'] = df_test['sentiment'].map({'positive': 1, 'negative': 0})

# Create the test_data with the updated DataFrame
test_data = encode_dataset(df_test).batch(BATCH_SIZE)

# Evaluate the model
model.evaluate(test_data)

y_pred = model.predict(test_data)
y_actual = np.concatenate([y for x, y in test_data], axis=0)

labels = ["negative", "positive"]

def predict(text):
  input_ids, attention_mask, token_type_ids = encode_sentence(text)["input_ids"],\
  encode_sentence(text)["attention_mask"],\
  encode_sentence(text)["token_type_ids"]
  input_ids = tf.expand_dims(input_ids, 0)
  attention_mask = tf.expand_dims(attention_mask, 0)
  token_type_ids = tf.expand_dims(token_type_ids, 0)

  outputs = model([input_ids, attention_mask, token_type_ids])
  return labels[np.argmax(tf.nn.softmax(outputs[0], axis=1).numpy()[0])]