class WordEmbeddings():
    def __init__(self, filepath, max_sentence_size=50):
        self.word_embeddings = gensim.models.KeyedVectors.load_word2vec_format(filepath, binary=True)
        self.max_sentence_size = max_sentence_size
        self.word_embeddings_length = 300
        
        
    def apply_padding(self, sentences):
        for index, sentence in enumerate(sentences):
            if len(sentence) > self.max_sentence_size:
                sentences[index] = sentence[:self.max_sentence_size]
            else:
                num_fill = self.max_sentence_size - len(sentence)
                for i in range(num_fill):
                    sentences[index].append('<padding>') # A float is required for masking
                    
        return sentences

    def get_sentences(self, input_data):
        sentences = []
        sentence = []
        for index in range(input_data.shape[0]):
            sentence.append(input_data.iloc[index]['word'])

            if input_data.iloc[index]['EOS'] == True:           
                sentences.append(sentence)
                sentence = []

        return sentences
