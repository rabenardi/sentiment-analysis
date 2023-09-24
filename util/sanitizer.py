from util.source_reader import Source
from os import path
from gensim import utils

# Mempersiapkan daftar tanda baca
from typing import List, Callable

# Mempersiapkan daftar tanda baca
from string import punctuation
punctuation_len = len(punctuation)

# Mempersiapkan daftar kata untuk difilter
filter_path = path.join(path.dirname(__file__), "filter")
words_col = "Kata"

filter_lits = Source(name="Filter Lists", the_path=filter_path)
filter_lits = filter_lits.fetch().flatten().dataframes[words_col]
filter_lits = filter_lits.values.tolist()

# Menyiapkan kasus khusus
f_class_list = [
    lambda word: word.startswith("<") or word.endswith(">")
]


class Sanitizer():
    @staticmethod
    def _list_filter(word: str, filter_list: List[str]):
        """
        Menyaring stop-words, bekerja pada level kata
        """

        return word if word not in filter_list else " "


    @staticmethod
    def _special_cases_filter(word: str, f_class_list: List[Callable]):
        """
        Menyaring kata dengan kasus-kasus khusus, bekerja pada level kata
        """

        # Membuat list yang berisi nilai keluaran fungsi-fungsi pada f_class_list
        cases = f_class_list
        cases = map(lambda func: func(word), cases)

        # Memeriksa apakah kata telah melewati semua filter
        return word if not any(cases) else ""


    @staticmethod
    def remove_punctuation(sentence: str):
        """
        Menghapus tanda baca pada kalimat dan menggantinya dengan spasi
        """
        return sentence.translate(str.maketrans(punctuation, " "*punctuation_len))


    @staticmethod
    def remove_unnecessary_words(sentence: str, filter_list: List[str]):
        """
        Menghapus kata-kata yang tidak perlu dan menghapus aksen pada huruf vokal
        """
        # Menyaring stop-words pada kalimat
        f_map = lambda x: Sanitizer._list_filter(x, filter_list=filter_list)
        sentence = " ".join(list(filter(f_map, sentence.split())))

        # Menghapus aksen pada huruf vokal setiap kata dalam kalimat
        return " ".join(utils.simple_preprocess(sentence, deacc=True))
    

    @staticmethod
    def remove_specific(sentence: str, f_class_list: List[Callable]):
        """
        Menghapus kata-kata dalam kalimat dengan kasus khusus
        """

        f_map = lambda x: Sanitizer._special_cases_filter(x, f_class_list=f_class_list)
        return " ".join(list(filter(f_map, sentence.split())))


    @staticmethod
    def sanitize(sentence: str, f_class_list: List[Callable]=f_class_list, filter_list: List[str]=filter_lits):
        """
        Menyanitasi kalimat dengan (1) mengubah semua kata menjadi lowercase, (2) menghapus kata-kata dengan kasus khusus, (3) menghapus tanda baca, dan (4) menghapus kata-kata yang tidak perlu
        """

        # Memanggil 3 fungsi Sanitizer
        sentence = sentence.lower()
        sentence = Sanitizer.remove_specific(sentence, f_class_list=f_class_list)
        sentence = Sanitizer.remove_punctuation(sentence)
        sentence = Sanitizer.remove_unnecessary_words(sentence, filter_list=filter_list)

        return sentence
    
    
    @staticmethod
    def sanitize_paragraph(paragraph: list):
        """
        Memanggil fungsi sanitize untuk kumpulan kalimat
        """

        return map(Sanitizer.sanitize, paragraph)

# print(Sanitizer.sanitize("<USERNAME> ahhh katanya jaga ucapan, lahh ucapan dia ngk dijaga sendiri.. dasar munafik.. dasar betina zaman now ellu kali yangg kelebihan micin..loe sendiri otak loe kan di selangkangan mana paham.. semoga Tuhan ampunin kemunafikan loe cun"))