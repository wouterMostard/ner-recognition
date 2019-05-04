import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential
from keras.layers import LSTM, TimeDistributed, Dense

all_data = pd.read_csv('./annotated_entities.csv')

def threshold_ner(target, threshold=0.30):
    ner_count = 0
    
    for entity in target:
        if entity != 'O':
            ner_count += 1
    return ner_count / len(target) >= threshold


MAX_SENTENCE_SIZE = 5


def apply_padding(sentences, entities):
    # Apply padding or cut of the sentences if needed
    for index, sentence in enumerate(sentences):
        if len(sentence) > MAX_SENTENCE_SIZE:
            sentences[index] = sentence[:MAX_SENTENCE_SIZE]
            entities[index] = entities[index][:MAX_SENTENCE_SIZE]
        else:
            num_fill = MAX_SENTENCE_SIZE - len(sentence)
            for i in range(num_fill):
                sentences[index].append('<padding>')
                entities[index].append('<padding>')


def get_input_and_target(all_data):
    sentences = []
    entities = []
    sentence = []
    entity = []
    # Split the sentences and remove sentences with no Named Entity
    for index in range(all_data.shape[0]):
        sentence.append(all_data.iloc[index]['pos_tag'])
        entity.append(all_data.iloc[index]['entity'])

        if all_data.iloc[index]['EOS'] == True:           
            # Only add if there is a NER found in the sentence (not all 'O')
            if threshold_ner(entity):
                sentences.append(sentence)
                entities.append(entity)
            
            sentence = []
            entity = []
            
    return sentences, entities


def encode_dummy(target_colum, target_name):
    encoded = []
    encoder = LabelBinarizer()
    encoder.fit(np.array(target_colum).flatten())
    print(encoder.classes_)
    
    for index, item in enumerate(target_colum):
        encoded.append(encoder.transform(item))
        
    return encoded, len(encoder.classes_)


def generate_LSTM_dataset(encoded, encoder_length):
    dataset = np.zeros((len(encoded), MAX_SENTENCE_SIZE, encoder_length))
    for item in range(len(encoded)):
        for tag in range(len(encoded[item])):
            dataset[item, tag, :] = encoded[item][tag]
    
    return dataset

def create_dataset(data, input_label, target_label):
    
    # Get the sentences 
    input_, target = get_input_and_target(data)
    
    # Apply padding or cut of the sentences if needed
    apply_padding(input_, target)
    
    # Encode the input_ (x) and target (y) into dummy variables
    input_encoded, input_encoder_size = encode_dummy(input_, input_label)
    target_encoded, target_encoder_size = encode_dummy(target, target_label)

    # Conver into a correct formate for keras (# num instances (sentences), # num items (words), # num input_ tags)
    input_array = generate_LSTM_dataset(input_encoded, input_encoder_size)
    target_array = generate_LSTM_dataset(target_encoded, target_encoder_size)
            
    return input_array, target_array

input_array, target_array = create_dataset(all_data, 'pos_tag', 'entity')


from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(input_array, target_array)



model = Sequential()
model.add(LSTM(500, input_shape=(MAX_SENTENCE_SIZE, input_array.shape[2]), return_sequences=True))
model.add(TimeDistributed(Dense(target_array.shape[2], activation='sigmoid')))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])


model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

predictions = model.predict(X_test)


for i in range(len(y_test)):
    if np.sum(np.argmax(predictions[i], axis=1)) != 12*len(np.argmax(predictions[i], axis=1)):
        print('predicted: ')
        print(np.argmax(predictions[i], axis=1))
        print('actual:')
        print(np.argmax(y_test[i], axis=1))



