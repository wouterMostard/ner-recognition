class Target():
    def __init__(self, max_sentence_size=50):
        self.max_sentence_size = max_sentence_size
        self.encoded_classes = None
        
        
    def apply_padding(self, entities):
        for index, entity in enumerate(entities):
            if len(entity) > self.max_sentence_size:
                entities[index] = entity[:self.max_sentence_size]
            else:
                num_fill = self.max_sentence_size - len(entity)
                for i in range(num_fill):
                    entities[index].append('<padding>')
                    
        return entities


    def get_entities(self, input_data):
        entities = []
        entity = []
        for index in range(input_data.shape[0]):
            entity.append(input_data.iloc[index]['entity'])

            if input_data.iloc[index]['EOS'] == True:           
                entities.append(entity)
                entity = []

        return entities
    
    
    def generate_LSTM_dataset(self, entities):
        encoder = LabelBinarizer()
        encoder.fit(np.array(entities).flatten())
        
        self.encoded_classes = encoder.classes_
        
        dataset = np.zeros((len(entities), self.max_sentence_size, len(self.encoded_classes)))
        for sentence_index in range(len(entities)): # 9542
            dataset[sentence_index, :, :] = encoder.transform(entities[sentence_index])
    
        return dataset
    
    def get_target_pos(self, input_data):
        entities = self.get_entities(input_data)
        
        self.apply_padding(entities)
        
        return self.generate_LSTM_dataset(entities)
