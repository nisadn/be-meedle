from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
import json
# from bsbi.views import BSBIIndex, VBEPostings, IdMap
# from helpers import BSBIIndex, VBEPostings, IdMap
from django.core.files import File
from django.contrib.staticfiles.storage import staticfiles_storage

from django.shortcuts import render
import array
import pickle
import os

import contextlib
import heapq
import time
import math
import re

from tqdm import tqdm
import nltk
# nltk.download('stopwords')
from nltk.stem import PorterStemmer
# from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

# from django.contrib.staticfiles.storage import staticfiles_storage
from django.core.files import File

class IdMap:
    """
    Ingat kembali di kuliah, bahwa secara praktis, sebuah dokumen dan
    sebuah term akan direpresentasikan sebagai sebuah integer. Oleh
    karena itu, kita perlu maintain mapping antara string term (atau
    dokumen) ke integer yang bersesuaian, dan sebaliknya. Kelas IdMap ini
    akan melakukan hal tersebut.
    """

    def __init__(self, str_to_id = {}, id_to_str = []):
        """
        Mapping dari string (term atau nama dokumen) ke id disimpan dalam
        python's dictionary; cukup efisien. Mapping sebaliknya disimpan dalam
        python's list.

        contoh:
            str_to_id["halo"] ---> 8
            str_to_id["/collection/dir0/gamma.txt"] ---> 54

            id_to_str[8] ---> "halo"
            id_to_str[54] ---> "/collection/dir0/gamma.txt"
        """
        self.str_to_id = str_to_id
        self.id_to_str = id_to_str

    def __len__(self):
        """Mengembalikan banyaknya term (atau dokumen) yang disimpan di IdMap."""
        return len(self.id_to_str)

    def __get_str(self, i):
        """Mengembalikan string yang terasosiasi dengan index i."""
        # TODO
        return self.id_to_str[i]

    def __get_id(self, s):
        """
        Mengembalikan integer id i yang berkorespondensi dengan sebuah string s.
        Jika s tidak ada pada IdMap, lalu assign sebuah integer id baru dan kembalikan
        integer id baru tersebut.
        """
        # TODO
        if s not in self.str_to_id:
            self.str_to_id[s] = len(self.id_to_str)
            self.id_to_str.append(s)
        return self.str_to_id[s]

    def __getitem__(self, key):
        """
        __getitem__(...) adalah special method di Python, yang mengizinkan sebuah
        collection class (seperti IdMap ini) mempunyai mekanisme akses atau
        modifikasi elemen dengan syntax [..] seperti pada list dan dictionary di Python.

        Silakan search informasi ini di Web search engine favorit Anda. Saya mendapatkan
        link berikut:

        https://stackoverflow.com/questions/43627405/understanding-getitem-method

        Jika key adalah integer, gunakan __get_str;
        jika key adalah string, gunakan __get_id
        """
        if type(key) is int:
            return self.__get_str(key)
        elif type(key) is str:
            return self.__get_id(key)
        else:
            raise TypeError

def sorted_merge_posts_and_tfs(posts_tfs1, posts_tfs2):
    """
    Menggabung (merge) dua lists of tuples (doc id, tf) dan mengembalikan
    hasil penggabungan keduanya (TF perlu diakumulasikan untuk semua tuple
    dengn doc id yang sama), dengan aturan berikut:

    contoh: posts_tfs1 = [(1, 34), (3, 2), (4, 23)]
            posts_tfs2 = [(1, 11), (2, 4), (4, 3 ), (6, 13)]

            return   [(1, 34+11), (2, 4), (3, 2), (4, 23+3), (6, 13)]
                   = [(1, 45), (2, 4), (3, 2), (4, 26), (6, 13)]

    Parameters
    ----------
    list1: List[(Comparable, int)]
    list2: List[(Comparable, int]
        Dua buah sorted list of tuples yang akan di-merge.

    Returns
    -------
    List[(Comparablem, int)]
        Penggabungan yang sudah terurut
    """
    # TODO
    posts_tfs1.sort()
    posts_tfs2.sort()
    result = []
    i,j = 0,0
    while i < len(posts_tfs1) and j < len(posts_tfs2):
        if posts_tfs1[i][0] == posts_tfs2[j][0]:
            result.append((posts_tfs1[i][0], posts_tfs1[i][1] + posts_tfs2[j][1]))
            i += 1
            j += 1
        elif posts_tfs1[i][0] < posts_tfs2[j][0]:
            result.append(posts_tfs1[i])
            i += 1
        else:
            result.append(posts_tfs2[j])
            j += 1
    while i < len(posts_tfs1):
        result.append(posts_tfs1[i])
        i += 1
    while j < len(posts_tfs2):
        result.append(posts_tfs2[j])
        j += 1
    return result

class VBEPostings:
    """ 
    Berbeda dengan StandardPostings, dimana untuk suatu postings list,
    yang disimpan di disk adalah sequence of integers asli dari postings
    list tersebut apa adanya.

    Pada VBEPostings, kali ini, yang disimpan adalah gap-nya, kecuali
    posting yang pertama. Barulah setelah itu di-encode dengan Variable-Byte
    Enconding algorithm ke bytestream.

    Contoh:
    postings list [34, 67, 89, 454] akan diubah dulu menjadi gap-based,
    yaitu [34, 33, 22, 365]. Barulah setelah itu di-encode dengan algoritma
    compression Variable-Byte Encoding, dan kemudian diubah ke bytesream.

    ASUMSI: postings_list untuk sebuah term MUAT di memori!

    """

    @staticmethod
    def vb_decode(encoded_bytestream):
        """
        Decoding sebuah bytestream yang sebelumnya di-encode dengan
        variable-byte encoding.
        """
        # TODO
        # slide 5 - page 36
        numbers = []
        n = 0
        for i in range(len(encoded_bytestream)):
            byte = encoded_bytestream[i]
            if byte < 128:
                n = 128*n + byte
            else:
                n = 128*n + byte-128
                numbers.append(n)
                n = 0
        return numbers

    @staticmethod
    def decode(encoded_postings_list):
        """
        Decodes postings_list dari sebuah stream of bytes. JANGAN LUPA
        bytestream yang di-decode dari encoded_postings_list masih berupa
        gap-based list.

        Parameters
        ----------
        encoded_postings_list: bytes
            bytearray merepresentasikan encoded postings list sebagai keluaran
            dari static method encode di atas.

        Returns
        -------
        List[int]
            list of docIDs yang merupakan hasil decoding dari encoded_postings_list
        """
        # TODO
        # ref: https://notebooks.githubusercontent.com/view/ipynb?azure_maps_enabled=true&browser=chrome&color_mode=auto&commit=4912ce487c4562da6b73447366a5fd421df7ac07&device=unknown&enc_url=68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f5a68656e67787572752f43533237362d7061312d736b656c65746f6e2d323031392f343931326365343837633435363264613662373334343733363661356664343231646637616330372f5041312d736b656c65746f6e2e6970796e62&logged_in=false&nwo=Zhengxuru%2FCS276-pa1-skeleton-2019&path=PA1-skeleton.ipynb&platform=android&repository_id=299051363&repository_type=Repository&version=103
        numbers = VBEPostings.vb_decode(encoded_postings_list)
        prefix_sum = 0
        result = []
        for num in numbers:
            prefix_sum += num
            result.append(prefix_sum)
        return result

    @staticmethod
    def decode_tf(encoded_tf_list):
        """
        Decodes list of term frequencies dari sebuah stream of bytes

        Parameters
        ----------
        encoded_tf_list: bytes
            bytearray merepresentasikan encoded term frequencies list sebagai keluaran
            dari static method encode_tf di atas.

        Returns
        -------
        List[int]
            List of term frequencies yang merupakan hasil decoding dari encoded_tf_list
        """
        return VBEPostings.vb_decode(encoded_tf_list)


class InvertedIndex:
    """
    Class yang mengimplementasikan bagaimana caranya scan atau membaca secara
    efisien Inverted Index yang disimpan di sebuah file; dan juga menyediakan
    mekanisme untuk menulis Inverted Index ke file (storage) saat melakukan indexing.

    Attributes
    ----------
    postings_dict: Dictionary mapping:

            termID -> (start_position_in_index_file,
                       number_of_postings_in_list,
                       length_in_bytes_of_postings_list,
                       length_in_bytes_of_tf_list)

        postings_dict adalah konsep "Dictionary" yang merupakan bagian dari
        Inverted Index. postings_dict ini diasumsikan dapat dimuat semuanya
        di memori.

        Seperti namanya, "Dictionary" diimplementasikan sebagai python's Dictionary
        yang memetakan term ID (integer) ke 4-tuple:
           1. start_position_in_index_file : (dalam satuan bytes) posisi dimana
              postings yang bersesuaian berada di file (storage). Kita bisa
              menggunakan operasi "seek" untuk mencapainya.
           2. number_of_postings_in_list : berapa banyak docID yang ada pada
              postings (Document Frequency)
           3. length_in_bytes_of_postings_list : panjang postings list dalam
              satuan byte.
           4. length_in_bytes_of_tf_list : panjang list of term frequencies dari
              postings list terkait dalam satuan byte

    terms: List[int]
        List of terms IDs, untuk mengingat urutan terms yang dimasukan ke
        dalam Inverted Index.

    """
    def __init__(self, index_name, postings_encoding, directory=''):
        """
        Parameters
        ----------
        index_name (str): Nama yang digunakan untuk menyimpan files yang berisi index
        postings_encoding : Lihat di compression.py, kandidatnya adalah StandardPostings,
                        GapBasedPostings, dsb.
        directory (str): directory dimana file index berada
        """

        self.index_file_path = staticfiles_storage.url(f'{directory}/{index_name}.index')[1:]
        self.metadata_file_path = staticfiles_storage.url(f'{directory}/{index_name}.dict')[1:]

        self.postings_encoding = postings_encoding
        self.directory = directory

        self.postings_dict = {}
        self.terms = []         # Untuk keep track urutan term yang dimasukkan ke index
        self.doc_length = {}    # key: doc ID (int), value: document length (number of tokens)
                                # Ini nantinya akan berguna untuk normalisasi Score terhadap panjang
                                # dokumen saat menghitung score dengan TF-IDF atau BM25

    def __enter__(self):
        """
        Memuat semua metadata ketika memasuki context.
        Metadata:
            1. Dictionary ---> postings_dict
            2. iterator untuk List yang berisi urutan term yang masuk ke
                index saat konstruksi. ---> term_iter
            3. doc_length, sebuah python's dictionary yang berisi key = doc id, dan
                value berupa banyaknya token dalam dokumen tersebut (panjang dokumen).
                Berguna untuk normalisasi panjang saat menggunakan TF-IDF atau BM25
                scoring regime; berguna untuk untuk mengetahui nilai N saat hitung IDF,
                dimana N adalah banyaknya dokumen di koleksi

        Metadata disimpan ke file dengan bantuan library "pickle"

        Perlu memahani juga special method __enter__(..) pada Python dan juga
        konsep Context Manager di Python. Silakan pelajari link berikut:

        https://docs.python.org/3/reference/datamodel.html#object.__enter__
        """
        # Membuka index file
        idx_file = open(self.index_file_path, 'rb')
        self.index_file = File(idx_file)

        # Kita muat postings dict dan terms iterator dari file metadata
        with open(self.metadata_file_path, 'rb') as f:
            self.postings_dict, self.terms, self.doc_length = pickle.load(File(f))
            self.term_iter = self.terms.__iter__()

        return self

    def __exit__(self, exception_type, exception_value, traceback):
        """Menutup index_file dan menyimpan postings_dict dan terms ketika keluar context"""
        # Menutup index file
        self.index_file.close()

        # Menyimpan metadata (postings dict dan terms) ke file metadata dengan bantuan pickle
        # with open(self.metadata_file_path, 'wb') as f:
        #     pickle.dump([self.postings_dict, self.terms, self.doc_length], File(f))


class InvertedIndexReader(InvertedIndex):
    """
    Class yang mengimplementasikan bagaimana caranya scan atau membaca secara
    efisien Inverted Index yang disimpan di sebuah file.
    """
    def __iter__(self):
        return self

    def __next__(self): 
        """
        Class InvertedIndexReader juga bersifat iterable (mempunyai iterator).
        Silakan pelajari:
        https://stackoverflow.com/questions/19151/how-to-build-a-basic-iterator

        Ketika instance dari kelas InvertedIndexReader ini digunakan
        sebagai iterator pada sebuah loop scheme, special method __next__(...)
        bertugas untuk mengembalikan pasangan (term, postings_list, tf_list) berikutnya
        pada inverted index.

        PERHATIAN! method ini harus mengembalikan sebagian kecil data dari
        file index yang besar. Mengapa hanya sebagian kecil? karena agar muat
        diproses di memori. JANGAN MEMUAT SEMUA INDEX DI MEMORI!
        """
        curr_term = next(self.term_iter)
        pos, number_of_postings, len_in_bytes_of_postings, len_in_bytes_of_tf = self.postings_dict[curr_term]
        postings_list = self.postings_encoding.decode(self.index_file.read(len_in_bytes_of_postings))
        tf_list = self.postings_encoding.decode_tf(self.index_file.read(len_in_bytes_of_tf))
        return (curr_term, postings_list, tf_list)

    def get_postings_list(self, term):
        """
        Kembalikan sebuah postings list (list of docIDs) beserta list
        of term frequencies terkait untuk sebuah term (disimpan dalam
        bentuk tuple (postings_list, tf_list)).

        PERHATIAN! method tidak boleh iterasi di keseluruhan index
        dari awal hingga akhir. Method ini harus langsung loncat ke posisi
        byte tertentu pada file (index file) dimana postings list (dan juga
        list of TF) dari term disimpan.
        """
        # TODO
        start, num, length_post, length_tf = self.postings_dict[term]
        self.index_file.seek(start)
        postings_list = self.postings_encoding.decode(self.index_file.read(length_post))
        tf_list = self.postings_encoding.decode_tf(self.index_file.read(length_tf))
        return (postings_list, tf_list)

class BSBIIndex:
    """
    Attributes
    ----------
    term_id_map(IdMap): Untuk mapping terms ke termIDs
    doc_id_map(IdMap): Untuk mapping relative paths dari dokumen (misal,
                    /collection/0/gamma.txt) to docIDs
    data_dir(str): Path ke data
    output_dir(str): Path ke output index files
    postings_encoding: Lihat di compression.py, kandidatnya adalah StandardPostings,
                    VBEPostings, dsb.
    index_name(str): Nama dari file yang berisi inverted index
    """
    def __init__(self, data_dir, output_dir, postings_encoding, index_name = "main_index"):
        self.term_id_map = IdMap()
        self.doc_id_map = IdMap()
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.index_name = index_name
        self.postings_encoding = postings_encoding
        # self.module_dir = os.path.dirname(__file__)
        # Untuk menyimpan nama-nama file dari semua intermediate inverted index
        self.intermediate_indices = []

    def load(self):
        """Memuat doc_id_map and term_id_map dari output directory"""

        term_str_to_id_path = staticfiles_storage.url(f'{self.output_dir}/terms_str_to_id.dict')[1:]
        with open(term_str_to_id_path, 'rb') as f:
            self.term_id_map.str_to_id = pickle.load(File(f))
        term_id_to_str_path = staticfiles_storage.url(f'{self.output_dir}/terms_id_to_str.dict')[1:]
        with open(term_id_to_str_path, 'rb') as f:
            self.term_id_map.id_to_str = pickle.load(File(f))

        doc_str_to_id_path = staticfiles_storage.url(f'{self.output_dir}/docs_str_to_id.dict')[1:]
        with open(doc_str_to_id_path, 'rb') as f:
            self.doc_id_map.str_to_id = pickle.load(File(f))
        doc_id_to_str_path = staticfiles_storage.url(f'{self.output_dir}/docs_id_to_str.dict')[1:]
        with open(doc_id_to_str_path, 'rb') as f:
            self.doc_id_map.id_to_str = pickle.load(File(f))

    
    def retrieve_bm25(self, query, k = 10, k1 = 10, b = 0.5):
        """
        Melakukan Ranked Retrieval dengan skema BM25 dan TaaT (Term-at-a-Time).
        Method akan mengembalikan top-K retrieval results.

        w(t, D) = ((k + 1) * tf(t, D)) / (k * ((1 - b) + b * dl/avdl) + tf(t, D))

        w(t, Q) = IDF = log (N / df(t))

        Score = untuk setiap term di query, akumulasikan w(t, Q) * w(t, D).

        catatan: 
            1. informasi DF(t) ada di dictionary postings_dict pada merged index
            2. informasi TF(t, D) ada di tf_li
            3. informasi N bisa didapat dari doc_length pada merged index, len(doc_length)
            4. dl = sigma(tf(t, D)); for t in V
            5. avdl = average dl

        Parameters
        ----------
        query: str
            Query tokens yang dipisahkan oleh spasi

            contoh: Query "universitas indonesia depok" artinya ada
            tiga terms: universitas, indonesia, dan depok

        Result
        ------
        List[(int, str)]
            List of tuple: elemen pertama adalah score similarity, dan yang
            kedua adalah nama dokumen.
            Daftar Top-K dokumen terurut mengecil BERDASARKAN SKOR.

        JANGAN LEMPAR ERROR/EXCEPTION untuk terms yang TIDAK ADA di collection.

        """
        # TODO
        if len(self.term_id_map) == 0 or len(self.doc_id_map) == 0:
            self.load()
        
        # create stemmer
        stemmer = PorterStemmer()

        # list of stopwords
        # stop_words = set(stopwords.words('english'))
        with open(staticfiles_storage.url('stopwords/english')[1:]) as f:
            stop_words = f.read().split()
        stop_words = set(stop_words)

        # tokenize
        tokenizer = RegexpTokenizer(r'\w+')
        rem_num = re.sub('[0-9]+', '', query)
        query_term = tokenizer.tokenize(rem_num)
        filtered = [stemmer.stem(t) for t in query_term if not t.lower() in stop_words]
        heap = []

        with InvertedIndexReader(self.index_name, directory=self.output_dir, postings_encoding=
                                 self.postings_encoding) as mapper:
            
            N = len(mapper.doc_length)
            avdl = sum(mapper.doc_length.values()) / N

            for term in filtered:
                # handle term yg tidak ada di collection
                if self.term_id_map[term] not in mapper.postings_dict:
                    continue

                scores_per_doc = []
                df = mapper.postings_dict[self.term_id_map[term]][1]
                wtq = math.log(N / df, 10)
                postings_list, tf_list = mapper.get_postings_list(self.term_id_map[term])
                for i in range(df):
                    dl = mapper.doc_length[postings_list[i]]
                    wtd = ((k1 + 1) * tf_list[i]) / (k1 * ((1 - b) + b * dl/avdl) + tf_list[i])
                    score = wtq * wtd
                    scores_per_doc.append((self.doc_id_map[postings_list[i]], score))
                heapq.heappush(heap, scores_per_doc)
                
            # calculate cumulative score for each doc
            while len(heap)>1:
                list1 = heapq.heappop(heap)
                list2 = heapq.heappop(heap)
                merged_scores = sorted_merge_posts_and_tfs(list1, list2)
                heapq.heappush(heap, merged_scores)

        heap_res = sorted(heap[0], key=lambda t: t[1])      # sort based on score
        result = heap_res[-1:-k-1:-1]                       # retrieve k-top
        result = [r[::-1] for r in result]                  # reverse tuple element to (score, doc)
        return result

# Create your views here.
def meedle_view(request):
    return HttpResponse("<h1> Django Deployed meedle</h1>")

def search(request, keyword):
    data = {
        "resp": "ok",
        "keyword": keyword,
    }
    return JsonResponse(data, safe=False)

def search_with_body(request):
    body = json.loads(request.body)
    if "query" not in body:
        return HttpResponse(status=400)
    return JsonResponse(body, safe=False)

def search_query(request):
    body = json.loads(request.body)
    if "query" not in body:
        return HttpResponse(status=400)

    query = body["query"]

    BSBI_instance = BSBIIndex(data_dir = 'collection', \
        postings_encoding = VBEPostings, \
        output_dir = 'index')

    docs = []
    for (_, doc) in BSBI_instance.retrieve_bm25(query, k = 10):
        docs.append(doc)
    
    response = {
        "query": query,
        "docs_id": docs,
    }

    return JsonResponse(response, safe=False)

def get_docs(request):

    body = json.loads(request.body)
    if "docs_id" not in body:
        return HttpResponse(status=400)

    result = {}
    for doc_id in body["docs_id"]:
        url = staticfiles_storage.url(f'collection/{str(doc_id)}')
        with open(url[1:], 'r') as f:
            result[doc_id] = File(f).read()

    return JsonResponse(result, safe=False)
