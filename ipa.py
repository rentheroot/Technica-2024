from speechbrain.inference.text import GraphemeToPhoneme
import spacy
from nltk.corpus import cmudict
import re
from spacy.lang.en import stop_words

'''
Methods for converting between CMU Arpabet and 
International Phonetic Alphabet (IPA) symbols
'''
class IPA:

    # init class and set translation table (tt)
    def __init__(self):

        # phoneme to ipa tt
        self.phoneme_ipa_dict = {
            'AA':'ɑ', 'AE':'æ', 'AH':'ʌ', 'AO':'ɔ', 'AW':'ə', 'AY':'ī', 'EH':'ɛ',
            'ER':'ɝ', 'EY':'ā', 'IH':'ɪ', 'IY':'i', 'OW':'ō', 'OY':'ʉ', 'UH':'ʊ',
            'UW':'u', 'B':'b', 'CH':'ʧ', 'D':'d', 'DH':'ð', 'F':'f', 'G':'ɡ',
            'HH':'h', 'JH':'ʤ', 'K':'k', 'L':'l', 'M':'m', 'N':'n', 'NG':'ŋ',
            'P':'p', 'R':'ɹ', 'S':'s', 'SH':'ʃ', 'T':'t', 'TH':'θ', 'V':'v',
            'W':'w', 'Y':'j', 'Z':'z', 'ZH':'ʒ', ' ':''
        }

        # ipa symbol to phoneme tt
        self.ipa_phoneme_dict = {v: k for k, v in self.phoneme_ipa_dict.items()}

        # english to phoneme tt
        self.cmudict_tuple = self.cmudict_data()

    # Convert CMUdict phonemes to IPA symbols
    def phoneme_to_ipa(self, text):
        ipa_text = []
        for sentence in text:
            separated_word = []
            for word in sentence:
                print(word)
                separated_word.append(self.phoneme_ipa_dict[word])

            ipa_text.append(separated_word)

        ipa_text = [''.join(i) for i in ipa_text]
        return ipa_text
    
    # Convert IPA symbols to CMUdict Phonemes
    def ipa_to_phoneme(self, word):
        phoneme_list = [self.ipa_phoneme_dict[char] for char in word]
        return phoneme_list
    
    # Create cmudict lookup tuple
    def cmudict_data(self):
        cmudict_tuple = cmudict.entries()
        cmudict_tuple = [([re.sub(r'[0-9]+', '', s) \
                           for s in t[1]], t[0]) \
                           for t in cmudict_tuple]

        return cmudict_tuple

    ''' 
    Convert a list of phonemes to a
    list of unique words with 
    matching pronunciations
    '''
    def phoneme_to_word(self, word):
        potential_words = []
        for i in self.cmudict_tuple:
            if i[0] == word and i[1] not in potential_words:
                potential_words.append(i[1])
        return(potential_words)
    
    # Get list of stop words in IPA symbols
    def get_common_words(self):
        word_list = stop_words.STOP_WORDS

        # Convert to CMUdict style phonemes
        phonemes = []
        for i in word_list:
            try:
                phonemes.append(g2p(i)) # type: ignore
            except:
                print(f"Failed to add '{i}' to word list.")

        # Convert to custom phonemic character set
        ipa_text = self.phoneme_to_ipa(phonemes)

        # Plaintext
        ipa_plaintext = [] 
        for word in ipa_text:
            if word[0] not in ipa_plaintext:
                ipa_plaintext.append(word[0])

        return(ipa_plaintext)