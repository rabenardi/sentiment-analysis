import pandas as pd
from os import path, listdir, getcwd
from itertools import chain
from numpy import array as np_array, concatenate as np_concat

from typing import Callable

class Source():
    @staticmethod
    def _is_csv(the_path: str):
        return path.splitext(the_path)[1] == ".csv"


    def __init__(self, name: str, the_path: str, debug: bool=False, log=True):
        self.is_folder = path.isdir(the_path)
        if not self.is_folder and not Source._is_csv(the_path):
            raise Exception("Source.__init__: File bukanlah sebuah csv!")

        self.name = name
        self.path = path.normpath(the_path)
        self.is_folder = path.isdir(the_path)
        self.dataframes = []
        self.cumulative_len = 0

        self.debug = debug
        self.use_log = log
        self._has_fetched_once = False
        self._sentimental = False


    def _log(self, msg):
        if self.use_log:
            print(msg)


    def _read(self, the_path: str):
        if self.debug:
            print("Memuat " + the_path)

        if path.isdir(the_path):
            raise Exception("Source._read: Bukan sebuah file!")
        
        if not Source._is_csv(the_path):
            return
            
        return pd.read_csv(the_path)

    def _is_fetched(self):
        return self._has_fetched_once and len(self.dataframes)!=0


    def fetch(self, the_path: str="", depth: int=1):
        if not self.is_folder or len(self.dataframes)==1:
            if self.debug:
                print("\nPath berupa file")
            
            self.dataframes.append(self._read(self.path))
            self.cumulative_len = len(self.dataframes[0])
            return True

        cur_path = self.path
        if the_path:
            cur_path = path.normpath(the_path)
        
        if self.debug:
            print("\nPath sekarang adalah " + cur_path)
            print("Path berupa folder")
            print("Depth sekarang adalah " + str(depth) + "\n")

        for content in listdir(cur_path):
            content_path = path.join(cur_path, content)
            if path.isdir(content_path) and depth > 1:
                self.fetch(the_path=content_path, depth=depth-1)
                continue
            
            if path.isfile(content_path):
                file_content = self._read(content_path)
                self.dataframes.append(file_content)
        
        self._has_fetched_once = True 
        
        for dataframe in self.dataframes:
            self.cumulative_len += len(dataframe)

        return True
    

    def normalize_for_sentiment_analysis(self, sentiment_col: str, text_col: str):
        if not self._is_fetched():
            raise Exception("Source.normalize_for_sentiment_analysis: File csv kosong atau belum pernah memanggil fetch()!")
        
        self._log(f"{self.name} mendapatkan {self.cumulative_len} entri")

        self.dataframes = pd.concat(self.dataframes)
        self.dataframes = self.dataframes[ [sentiment_col, text_col] ]
        if self.dataframes[sentiment_col].isna().sum() or self.dataframes[text_col].isna().sum():
            self.dataframes = self.dataframes.dropna()

        duplicates = self.dataframes.duplicated([text_col], keep="first")
        self.dataframes = self.dataframes.drop_duplicates()
        if duplicates.sum():
            self._log(f"Sebanyak {duplicates.sum()} duplikat ditemukan!")

        self.sentiment_values = list(self.dataframes[sentiment_col].unique())
        f_class = lambda item: self.sentiment_values.index(item)
        self.dataframes = [self.dataframes[sentiment_col], self.dataframes[text_col]]
        self.dataframes[0] = np_array(self.dataframes[0].map(f_class))

        self._log(f"Menghapus {self.cumulative_len-len(self.dataframes[1])} entri dalam proses normalisasi")

        self.cumulative_len = len(self.dataframes[1])
        self._sentimental = True

        return True


    def prepare(self, sentiment_col: str, text_col: str):
        self.fetch()
        self.normalize(sentiment_col=sentiment_col, text_col=text_col)
        return True

    
    def map(self, index: int, f_class: Callable):
        if index < 0 or index > self.cumulative_len:
            raise Exception("Source.map: indeks melampaui batas")
        
        self.dataframes[index] = self.dataframes[index].map(f_class)
        return True

    
    def join_for_sentimental_analysis(self, source):
        if not source._sentimental:
            raise Exception("Source.join_for_sentimental_analysis: Bukan source sentimental!")
        
        self.dataframes[0] = np_concat((self.dataframes[0], source.dataframes[0]))
        self.dataframes[1] = pd.concat(self.dataframes[1], source.dataframes[1])