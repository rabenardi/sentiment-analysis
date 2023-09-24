import pandas as pd
from os import path, listdir, getcwd
from itertools import chain
from numpy import array as np_array, concatenate as np_concat

# Untuk type-hinting fungsi
from typing import Callable

# def f_map_labels(labels):
#     if labels == "neutral":
#         return 0
#     if labels == "negative":
#         return 1
#     if labels == "positive":
#         return 2

f_pass = lambda x: x

class Source():
    @staticmethod
    def _is_csv(the_path: str):
        """
        Menguji apakah suatu file berformat csv
        """
        
        # Membelah string path menjadi dua yakni [path, format_file]
        return path.splitext(the_path)[1] == ".csv"


    def __init__(self, name: str, the_path: str, debug: bool=False, log=True):
        """
        Membuat instansi Source
        """

        self.is_folder = path.isdir(the_path)

        # Memeriksa apakah file berformat csv (jika the_path menunjukk sebuah file)
        if not self.is_folder and not Source._is_csv(the_path):
            raise Exception("Source.__init__: File bukanlah sebuah csv!")

        # inisialisasi atribut
        self.name = name
        self.path = path.normpath(the_path)
        self.dataframes = []
        self.sentiment_values = []
        self.cumulative_len = 0

        # inisialisasi atribut boolean
        self.is_folder = path.isdir(the_path)
        self.debug = debug
        self.use_log = log
        self._has_fetched_once = False
        self._sentimental = False


    def _log(self, msg: str):
        """
        Mencetak pesan. Dapat dimatikan atau dinyalakan
        """
        if self.use_log:
            print(msg)


    def _read(self, the_path: str):
        """
        Membaca file csv
        """
        if self.debug:
            print("Memuat " + the_path)

        # Periksa apakah the_path sebuah folder
        if path.isdir(the_path):
            raise Exception("Source._read: Bukan sebuah file!")
        
        # Jika the_path sebuah file, periksa apakah berkasnya berformat csv
        if not Source._is_csv(the_path):
            return
        
        # Baca file csv
        return pd.read_csv(the_path)

    def _is_fetched(self):
        """
        Memeriksa apakah fungsi Source.fetch() sudah pernah dipanggil!
        """

        return self._has_fetched_once and len(self.dataframes)!=0


    def fetch(self, the_path: str="", depth: int=1):
        """
        Mencari dan membaca berkas csv di lokasi yang telah diberikan
        """

        # Jika sumber berupa sebuah berkas
        if not self.is_folder:
            if self.debug:
                print("\nPath berupa file")
            
            # Menambahkan data yang telah dibaca ke self.dataframes
            self.dataframes.append(self._read(self.path))

            # Menentukan panjang (karena hanya 1, maka panjang dataframe pertama, atau indeks 0)
            self.cumulative_len = len(self.dataframes[0])

            # Mengatur nilainya menjadi True agar tahu bahwa fungsi self.fetch sudah pernah dipanggil
            self._has_fetched_once = True 
            return self

        # Meneriksa apakah parameter the_path diberikan
        cur_path = self.path
        if the_path:
            cur_path = path.normpath(the_path)
        
        if self.debug:
            print("\nPath sekarang adalah " + cur_path)
            print("Path berupa folder")
            print("Depth sekarang adalah " + str(depth) + "\n")

        # Untuk setiap isi (baik folder maupun berkas) dalam the_path
        for content in listdir(cur_path):

            # Listdir hanya mendaftarkan isinya saja, tidak lokasinya juga
            # Oleh karena itu, buat lokasi dengan menggabung  cur_path dan content
            content_path = path.join(cur_path, content)

            # Jika sebuah folder dan harga depth > 1, maka jalankan fetch lagi (rekursi)
            if path.isdir(content_path) and depth > 1:
                self.fetch(the_path=content_path, depth=depth-1)
                continue
            
            # Jika baca file (dan cek apakah file tersebut csv) dan tambahkan ke self.dataframse
            if path.isfile(content_path):
                file_content = self._read(content_path)
                self.dataframes.append(file_content)
        
        # Mengatur nilainya menjadi True agar tahu bahwa fungsi self.fetch sudah pernah dipanggil
        self._has_fetched_once = True 
        
        # Mencari panjang total dataframe dengan melakukan iterasi pada self.dataframes
        for dataframe in self.dataframes:
            self.cumulative_len += len(dataframe)

        return self
    

    def map(self, index: int, f_class: Callable):
        """
        Memetakan dataframe pada indeks ke [index] dengan fungsi yang telah diberikan
        """
        # Periksa apakah index bukan negatif atau tidak melebihi 
        if index < 0 or index > self.cumulative_len:
            raise Exception("Source.map: indeks melampaui batas")
        
        # Memetakan dataframe pada indeks index dengan fungsi f_class
        self.dataframes[index] = self.dataframes[index].map(f_class)
        return self


    def flatten(self, skip_eror: bool=False):
        """
        Meratakan kumpulan dataframes menjadi satu dataframe
        """
        # Memeriksa apakah self.dataframes sudah pernah diratakan
        dataframes_type = type(self.dataframes)
        if dataframes_type == pd.DataFrame:
            if skip_eror: 
                return
            
            raise Exception("Source.flatten: sumber sudah pernah diratakan!")
        
        self.dataframes = pd.concat(self.dataframes)
        return self


    def normalize_for_sentiment_analysis(self, sentiment_col: str, text_col: str, f_sanitize: Callable=f_pass):
        """
        Normalisasi instansi ini agar data digunakan sebagai sampel pelatihan
        """

        # Jika fungsi self.fetch belum pernah dipanggil
        if not self._is_fetched():
            raise Exception("Source.normalize_for_sentiment_analysis: File csv kosong atau belum pernah memanggil fetch()!")
        
        self._log(f"{self.name} mendapatkan {self.cumulative_len} entri")

        # Gabungkan kumpulan dataframe dalam self.dataframes menjadi satu
        self.flatten(skip_eror=True)

        # Mengambil dua kolom dalam dataframe, yakni sentiment_col dan text_col
        self.dataframes = self.dataframes[ [sentiment_col, text_col] ]

        # Jika ada nilai yang null di salah satu kolom, maka hapus
        if self.dataframes[sentiment_col].isna().sum() or self.dataframes[text_col].isna().sum():
            self.dataframes = self.dataframes.dropna()
            # self.dataframes.fillna("tanpa kontent", inplace=True) # bukan di hapus tapi di isi

        # Memeriksa apakah ada duplikat
        # duplicates = self.dataframes.duplicated([text_col], keep="first")
        # self.dataframes = self.dataframes.drop_duplicates()
        # if duplicates.sum():
        #     self._log(f"Sebanyak {duplicates.sum()} duplikat ditemukan!")

        # Mencari nilai yang unik dalam kolom sentimen dan memperbarui self.sentiment_values
        self.sentiment_values = list(self.dataframes[sentiment_col].unique())

        # Memperbarui self.dataframes menjadi list dengan dua elemen, dataframe sentimen dan dataframe sumber teks
        f_class = lambda item: self.sentiment_values.index(item)
        self.dataframes = [self.dataframes[sentiment_col], self.dataframes[text_col]]

        # Memetakan dataframe dengan f_class: sentimen menjadi sebuah angka sesuai dengan indeks pada self.sentimen_values
        # Lalu ubah menjadi numpy array
        self.dataframes[0] = np_array(self.dataframes[0].map(f_class).values.tolist())

        # Mengubah kata menjadi angka
        self.dataframes[1] = self.dataframes[1].map(f_sanitize)
        self.dataframes[1] = np_array(self.dataframes[1].values.tolist())

        # Laporkan berapa banyak data yang dihapus setelah menghapus duplikat dan nilai kosong
        self._log(f"Menghapus {self.cumulative_len-len(self.dataframes[1])} entri dalam proses normalisasi")

        # Memperbarui panjang dataframe dan memberitahu bahwa sumber digunakan untuk sampel pelatihan
        self.cumulative_len = len(self.dataframes[1])
        self._sentimental = True

        return self


    def prepare_for_sentiment_analysis(self, sentiment_col: str, text_col: str, f_sanitize: Callable=f_pass, depth: int=1):
        """
        Fungsi yang membungkus self.fetch dan self.normalize_for_sentiment_analysis
        """

        # Memanggil fungsi self.fetch dan self.normalize_for_sentiment_analysis
        self.fetch(depth=depth)
        
        # normalisasi data untuk sampel pelatihan
        self.normalize_for_sentiment_analysis(sentiment_col=sentiment_col, text_col=text_col, f_sanitize=f_sanitize)

        return self


    
    def join_for_sentimental_analysis(self, source: "Source"):
        """
        Menggabungkan dataframe pada instans Source sentimenal lain
        """

        # Periksa apakah source digunakan sebagai sampel pelatihan
        if not source._sentimental:
            raise Exception("Source.join_for_sentimental_analysis: Bukan source sentimental!")
        
        # Menyelaraskan nilai sentimen
        l_label = source.dataframes[0]
        l_sentimen = source.sentiment_values

        for index in range(len(l_label)):
            label_to_text = l_sentimen[l_label[index]]
            l_label[index] = self.sentiment_values.index(label_to_text)

        # Panggil fungsi concat untuk np dan untuk dataframe
        self.dataframes[0] = np_concat((self.dataframes[0], source.dataframes[0]))
        self.dataframes[1] = np_concat([self.dataframes[1], source.dataframes[1]])
        
        self._log(f"Telah menambahkan {len(source.dataframes[1])} entri baru")

        # Memperbarui panjang
        self.cumulative_len = len(self.dataframes[1])
        
        return True

# from os import getcwd
# sample1 = Source(name="sample1", the_path=path.join(getcwd(), "samples", "s1.csv"))
# sample2 = Source(name="sample1", the_path=path.join(getcwd(), "samples", "s2.csv"))
# sample1.fetch().normalize_for_sentiment_analysis(sentiment_col="Sentiment",text_col="Text")
# sample2.prepare_for_sentiment_analysis(sentiment_col="Sentiment", text_col="Text")


# print(sample1.cumulative_len, sample2.cumulative_len)
# sample1.join_for_sentimental_analysis(sample2)
# print(sample1.cumulative_len, sample2.cumulative_len)
# print(sample1.dataframes[1])