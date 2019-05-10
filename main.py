from sklearn.metrics import classification_report
from keras.layers import Embedding, Dropout
from keras.metrics import categorical_accuracy
from keras_contrib.layers import CRF
import matplotlib.pyplot as plt


# Load sentences and pad the data
sentences = word_embs.get_sentences(train_data)
padded_sentences = word_embs.apply_padding(sentences)

# Create a word index from the train vocabulary
vocab = train_data['word'].str.lower().drop_duplicates().values
word_index = {v: k for k, v in enumerate(vocab, 2)}
word_index["<padding>"] = 0

# Create encoded docs for training
encoded_docs = []
for sentence in padded_sentences:
    encoded_sentence = []
    for word in sentence:
        encoded_sentence.append(word_index.get(word.lower(), 1))
        
    encoded_docs.append(encoded_sentence)
encoded_docs = np.array(encoded_docs)


# create a weight matrix for words in training docs
embedding_matrix = np.zeros((len(vocab), 300))
last_i = 0
for word, i in word_index.items():
    if word in word_embs.word_embeddings:
        embedding_vector = word_embs.word_embeddings[word]
        embedding_matrix[i] = embedding_vector
        
    last_i = i
embedding_matrix[1] = np.zeros((1, 300)) # For the unknown words


# Create a target y for training
target = Target()
y = target.get_target_ner(input_data=train_data)


# Define model
model = Sequential()
model.add(Embedding(len(vocab), output_dim=300, weights=[embedding_matrix], trainable=False, input_length=50, mask_zero=True))
model.add(Bidirectional(LSTM(50, return_sequences=True)))
model.add(TimeDistributed(Dense(50, activation='sigmoid')))
crf = CRF(y.shape[2])
model.add(crf)
model.compile(loss=crf.loss_function, optimizer='rmsprop', metrics=[crf.accuracy])

# Fit and show history
history = model.fit(encoded_docs, y, epochs=20, validation_split=0.20, batch_size=32)
hist = pd.DataFrame(history.history)
plt.style.use("ggplot")
plt.figure(figsize=(12,12))
plt.plot(hist["loss"])
plt.plot(hist["val_loss"])
plt.show()


# TESTING
# Create sentence
sentences_test = word_embs.get_sentences(test_data)
padded_sentences_test = word_embs.apply_padding(sentences_test)

# Encode test sentences
encoded_docs_test = []
for sentence in padded_sentences_test:
    encoded_sentence = []
    for word in sentence:
        encoded_sentence.append(word_index.get(word.lower(), 1))
        
    encoded_docs_test.append(encoded_sentence)
encoded_docs_test = np.array(encoded_docs_test)

# Get predictions
predictions = model.predict(encoded_docs_test)
y_test = target.get_target_ner(input_data=test_data)

# Get test DF and show
prediction_df = {'prediction': [], 'actual': [], 'id': []}
for i in range(len(y_test)):
    for j in np.argmax(predictions[i], axis=1):
        prediction_df['prediction'].append(target.encoded_classes[j])
        prediction_df['id'].append(i)
        
    for j in np.argmax(y_test[i], axis=1):
        prediction_df['actual'].append(target.encoded_classes[j])
        
result_df_test = pd.DataFrame(prediction_df)
print(classification_report(result_df_test['actual'], result_df_test['prediction']))
