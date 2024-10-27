""" Imports """
import mlflow
from convokit import Corpus, download
from convokit import text_processing
from spacy.lang.en import stop_words
from cleantext import clean
from speechbrain.inference.text import GraphemeToPhoneme
from nltk.corpus import cmudict
import re
from ipa import IPA
import keras
from keras.layers import TextVectorization
import pickle
import tensorflow as tf
import numpy as np

# log with mlflow
mlflow.tensorflow.autolog()

class Prepare_Train_Data:
    def __init__(self):

        # CMU Movie Corpus
        self.corpus = Corpus(filename=download("movie-corpus"))

        # Clean Corpus
        sentences = self.text_preprocess(self.corpus)

        # Convert all sentences in corpus to phonemic representations
        transcriptions = self.graphemes_to_phonemes(sentences)

        self.ipa_converter = IPA()

        self.ipa_text = self.phoneme_to_ipa(transcriptions)

    def text_preprocess(self, corpus):

        # Define Cleaner Params
        clean_str = lambda s: clean(
            s,
            fix_unicode=True,  # fix various unicode errors
            to_ascii=True,  # transliterate to closest ASCII representation
            lower=True,  # lowercase text
            no_line_breaks=True,  # fully strip line breaks as opposed to only normalizing them
            no_urls=True,  # replace all URLs with a special token
            no_emails=True,  # replace all email addresses with a special token
            no_phone_numbers=True,  # replace all phone numbers with a special token
            no_numbers=True,  # replace all numbers with a special token
            no_digits=False,  # replace all digits with a special token
            no_currency_symbols=True,  # replace all currency symbols with a special token
            no_punct=True,  # fully remove punctuation
            replace_with_url="",
            replace_with_email="",
            replace_with_phone_number="",
            replace_with_number="",
            replace_with_digit="0",
            replace_with_currency_symbol="",
            lang="en",
        )

        # Clean Data
        tc = text_processing.textCleaner.TextCleaner(clean_str)
        corpus = tc.transform(corpus)
        sentences = []
        uids = corpus.get_utterance_ids()

        for uid in uids:
            parsed_data = corpus.get_utterance(uid).text
            sentences.append(parsed_data)

        return(sentences)
    
    def graphemes_to_phonemes(self, sentences):
        # Convert graphemes to phonemes
        g2p = GraphemeToPhoneme.from_hparams("speechbrain/soundchoice-g2p")

        transcriptions = []
        for sentence in sentences:
            try:
                transcriptions.append(g2p(sentence)[:-1])
            except:
                pass
        return transcriptions
    
    def reformat_list(self,l):
        result = []
        sublist = []

        for item in l:
            if item == ' ':
                if sublist:  # add current sublist to result if it's not empty
                    result.append(sublist)
                    sublist = []  # reset sublist
            else:
                sublist.append(item)

        # Add any remaining items in sublist
        if sublist:
            result.append(sublist)
        
        return result
    
    def phoneme_to_ipa(self, transcriptions):
        ipa_text = []
        for l in transcriptions:
            l = self.reformat_list(l)
            print(l)
            ipa_text_part = self.ipa_converter.phoneme_to_ipa(l)
            ipa_text.append(ipa_text_part)
        return ipa_text

class Build_Model:
    def __init__(self, ipa_text, model_is_saved):

        if model_is_saved == False:
            self.max_seq_length = 20
            self.embedding_dim = 100
            self.lstm_units = 128

            self.model = self.build_and_test(ipa_text)

            mlflow.tensorflow.save_model(self.model, 'model')
            # Save vector layer
            # Pickle the config and weights
            pickle.dump({'config': self.vectorize_layer.get_config(),
                        'weights': self.vectorize_layer.get_weights()}
                        , open("vector.pickle", "wb"))

        else:
            self.model = mlflow.tensorflow.load_model('model')
            with open(r"vector.pickle", "rb") as input_file:

                self.vectorize_layer = pickle.load(input_file)

            vocab = pickle.load(open("vocab.pickle", "rb"))
            from_disk = pickle.load(open("vector.pickle", "rb"))
            new_v = TextVectorization.from_config(from_disk['config'])
            # You have to call `adapt` with some dummy data (BUG in Keras)
            new_v.adapt(tf.data.Dataset.from_tensor_slices(["xyz"]))
            new_v.set_weights(from_disk['weights'])

            self.vectorize_layer = new_v
            self.vectorize_layer.set_vocabulary(vocab)

    def build_and_test(self, ipa_text):

        # Plaintext
        ipa_plaintext = [] 
        for sentence in ipa_text:
            ipa_plaintext.append(' '.join(sentence))

        '''
        Instance of TextVectorization to tokenize 
        input text and output a vector of integers
        '''
        self.vectorize_layer = TextVectorization(standardize=None, # type: ignore
                                                    split="whitespace",
                                                    output_mode='int')

        # Adapt vectorizer to the input text
        self.vectorize_layer.adapt(ipa_plaintext)

        # Get vocabulary
        vocab = self.vectorize_layer.get_vocabulary()

        # Save vocabulary
        with open('vocab.pickle', 'wb') as handle:
            pickle.dump(vocab, 
                        handle, 
                        protocol=pickle.HIGHEST_PROTOCOL)

        # Vector of integers
        sequences = self.vectorize_layer(ipa_plaintext)

        # Generate data: X is the sequence, y is the next word
        X = sequences[:, :-1] # type: ignore
        y = sequences[:, 1:] # type: ignore
        vocab_size = len(self.vectorize_layer.get_vocabulary())     

        # Define layers of model
        model = keras.Sequential([
            keras.layers.Embedding(input_dim=vocab_size,
                                        output_dim=self.embedding_dim),
            keras.layers.LSTM(self.lstm_units,return_sequences=True),
            keras.layers.Dense(vocab_size,activation='softmax')
        ])

        # Prepare model for training
        model.compile(loss='sparse_categorical_crossentropy', 
                        optimizer='adam', 
                        metrics=['accuracy'])

        # Train the model
        model.fit(X, y, epochs=10)

        return model

class Predictor:
    def __init__(self, model, vectorize_layer):
        self.model = model
        self.vectorize_layer = vectorize_layer
    def predict_user_input_with_prefix(self, input_text, top_k=5):
        """
        Predict the next phoneme or word based on partial user input, using a prefix match.

        Args:
            input_text (str): The text input from the user.
            top_k (int): Number of top predictions to return.
        
        Returns:
            List[str]: A list of the top-k predicted phonemes or words matching the input prefix.
        """
        # Tokenize and get the last entered word (prefix)
        words = input_text.split()
        prefix = words[-1] if words and input_text[-1] != " " else ""

        # Vectorize the input sequence (without the prefix if one exists)
        input_sequence = self.vectorize_layer([" ".join(words[:-1]) if prefix else input_text])

        # Check if input_sequence is non-empty
        if tf.size(input_sequence) == 0:
            return []  # No prediction if input sequence is empty

        # Predict the next word probabilities
        predictions = self.model.predict(input_sequence, verbose=0)

        # Get the vocabulary and filter by prefix
        vocab = self.vectorize_layer.get_vocabulary()
        top_k_indices = np.argsort(predictions[0][-1])[::-1]  # Sorted in descending order

        if prefix:
            # If there is a prefix, filter predictions by it
            prefix_pattern = re.compile(f"^{re.escape(prefix)}")
            matching_predictions = [vocab[idx] for idx in top_k_indices if prefix_pattern.match(vocab[idx])]
        else:
            # If no prefix, return the top-k predictions directly
            matching_predictions = [vocab[idx] for idx in top_k_indices]
        
        # Return up to top_k matching predictions
        return matching_predictions[:top_k]

    
#prepare_data = Prepare_Train_Data()
#model_build = Build_Model(ipa_text = None, model_is_saved = True)
#live_predict = Predictor(model_build.model, model_build.vectorize_layer)
# Example of using the new prediction function
#user_input = "ʃʊɹ h"
#predictions = live_predict.predict_user_input_with_prefix(user_input, top_k=6)
#print("Top predictions:", predictions)