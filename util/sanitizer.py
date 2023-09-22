from util.source_reader import Source
from os import path
from string import punctuation
from gensim import utils

stop_words_path = path.join(path.dirname(__file__), "filter", "stop-words.csv")
S_stop_words = Source(name="Filter Lists", the_path=stop_words_path)
S_stop_words.fetch()

stop_words = S_stop_words.dataframes[0]["KATA"].values.tolist()

punctuation_len = len(punctuation)

class Sanitizer():
    @staticmethod
    def _stop_word_filter(word: str):
        return word if word not in stop_words else " "


    @staticmethod
    def _special_char_filter(word: str):
        case1 = word.startswith("<") or word.endswith(">")

        return word if not case1 else " "


    @staticmethod
    def remove_punctuation(sentence: str):
        return sentence.translate(str.maketrans(punctuation, " "*punctuation_len))


    @staticmethod
    def remove_unnecessary_words(sentence: str):
        sentence = " ".join(list(filter(Sanitizer._stop_word_filter, sentence.split())))
        return " ".join(utils.simple_preprocess(sentence, deacc=True))
    

    @staticmethod
    def remove_specific(sentence: str):
        return " ".join(list(filter(Sanitizer._special_char_filter, sentence.split())))


    @staticmethod
    def sanitize(sentence: str):
        sentence = sentence.lower()
        sentence = Sanitizer.remove_specific(sentence)
        sentence = Sanitizer.remove_punctuation(sentence)
        sentence = Sanitizer.remove_unnecessary_words(sentence)

        return sentence
    
    
    @staticmethod
    def sanitize_paragraph(paragraph: list):
        return map(Sanitizer.sanitize, paragraph)

print()
