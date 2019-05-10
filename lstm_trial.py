import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential
from keras.layers import LSTM, TimeDistributed, Dense, Bidirectional
from sklearn.model_selection import train_test_split

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


X_train, X_test, y_train, y_test = train_test_split(input_array, target_array)



model = Sequential()
model.add(Bidirectional(LSTM(500, input_shape=(MAX_SENTENCE_SIZE, input_array.shape[2]), return_sequences=True)))
model.add(TimeDistributed(Dense(target_array.shape[2], activation='sigmoid')))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])


model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))

predictions = model.predict(X_test)

predictions = model.predict(X_test)

ner_tag = ['<padding>','B-Company name', 'B-Employee' ,'B-Function', 'B-Person'
 ,'B-Related company', 'B-Title' ,'I-Company name' ,'I-Employee', 'I-Function'
 ,'I-Person', 'I-Related company', 'I-Title', 'O']

ner_idx_to_name = {}
for i in range(len(ner_tag)):
    ner_idx_to_name[i] = ner_tag[i]

prediction_df = {'prediction': [], 'actual': [], 'id': []}
for i in range(len(y_test)):
    for j in np.argmax(predictions[i], axis=1):
        prediction_df['prediction'].append(ner_idx_to_name[j])
        prediction_df['id'].append(i)
        
    for j in np.argmax(y_test[i], axis=1):
        prediction_df['actual'].append(ner_idx_to_name[j])
        
result_df = pd.DataFrame(prediction_df)

result_df.to_csv('./results', index=0)

