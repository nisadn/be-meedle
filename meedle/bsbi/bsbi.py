import os
import pickle
import contextlib
import heapq
import time
import math
import re
from poll.settings import STATIC_URL, STATIC_ROOT

from meedle.bsbi.index import InvertedIndexReader, InvertedIndexWriter
from meedle.bsbi.util import IdMap, sorted_merge_posts_and_tfs
from meedle.bsbi.compression import StandardPostings, VBEPostings
from tqdm import tqdm
import nltk
nltk.download('stopwords')
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

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
        self.module_dir = os.path.dirname(__file__)
        # Untuk menyimpan nama-nama file dari semua intermediate inverted index
        self.intermediate_indices = []

    def save(self):
        """Menyimpan doc_id_map and term_id_map ke output directory via pickle"""

        with open(os.path.join(STATIC_ROOT, self.output_dir, 'terms.dict'), 'wb') as f:
            pickle.dump(self.term_id_map, f)
        with open(os.path.join(STATIC_ROOT, self.output_dir, 'docs.dict'), 'wb') as f:
            pickle.dump(self.doc_id_map, f)

    def load(self):
        """Memuat doc_id_map and term_id_map dari output directory"""

        with open(os.path.join(STATIC_ROOT, self.output_dir, 'terms.dict'), 'rb') as f:
            self.term_id_map = pickle.load(f)
        with open(os.path.join(STATIC_ROOT, self.output_dir, 'docs.dict'), 'rb') as f:
            self.doc_id_map = pickle.load(f)

    def parse_block(self, block_dir_relative):
        """
        Lakukan parsing terhadap text file sehingga menjadi sequence of
        <termID, docID> pairs.

        Gunakan tools available untuk Stemming Bahasa Inggris

        JANGAN LUPA BUANG STOPWORDS!

        Untuk "sentence segmentation" dan "tokenization", bisa menggunakan
        regex atau boleh juga menggunakan tools lain yang berbasis machine
        learning.

        Parameters
        ----------
        block_dir_relative : str
            Relative Path ke directory yang mengandung text files untuk sebuah block.

            CATAT bahwa satu folder di collection dianggap merepresentasikan satu block.
            Konsep block di soal tugas ini berbeda dengan konsep block yang terkait
            dengan operating systems.

        Returns
        -------
        List[Tuple[Int, Int]]
            Returns all the td_pairs extracted from the block
            Mengembalikan semua pasangan <termID, docID> dari sebuah block (dalam hal
            ini sebuah sub-direktori di dalam folder collection)

        Harus menggunakan self.term_id_map dan self.doc_id_map untuk mendapatkan
        termIDs dan docIDs. Dua variable ini harus 'persist' untuk semua pemanggilan
        parse_block(...).
        """
        # TODO
        # refs:
        #   https://ksnugroho.medium.com/dasar-text-preprocessing-dengan-python-a4fa52608ffe
        #   https://notebooks.githubusercontent.com/view/ipynb?azure_maps_enabled=true&browser=chrome&color_mode=auto&commit=4912ce487c4562da6b73447366a5fd421df7ac07&device=unknown&enc_url=68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f5a68656e67787572752f43533237362d7061312d736b656c65746f6e2d323031392f343931326365343837633435363264613662373334343733363661356664343231646637616330372f5041312d736b656c65746f6e2e6970796e62&logged_in=false&nwo=Zhengxuru%2FCS276-pa1-skeleton-2019&path=PA1-skeleton.ipynb&platform=android&repository_id=299051363&repository_type=Repository&version=103
        
        # create stemmer
        stemmer = PorterStemmer()

        # list of stopwords
        stop_words = set(stopwords.words('english'))

        # tokenizer
        tokenizer = RegexpTokenizer(r'\w+')
        
        dir_path = os.path.join(STATIC_ROOT, self.data_dir, block_dir_relative)
        td_pairs = []
        for doc_name in os.listdir(dir_path):
            with open(os.path.join(STATIC_ROOT, dir_path, doc_name), 'r') as f:
                for line in f.readlines():
                    # tokenization
                    rem_num = re.sub('[0-9]+', '', line)
                    word_tokens = tokenizer.tokenize(rem_num)
                    for token in word_tokens:
                        # stemming, remove stopwords, and append to td_pairs
                        if not token.lower() in stop_words:
                            stem = stemmer.stem(token.strip())
                            term_id = self.term_id_map[stem]
                            doc_id = self.doc_id_map[os.path.join(STATIC_ROOT, block_dir_relative, doc_name)]
                            td_pairs.append((term_id, doc_id))
        return td_pairs

    def invert_write(self, td_pairs, index):
        """
        Melakukan inversion td_pairs (list of <termID, docID> pairs) dan
        menyimpan mereka ke index. Disini diterapkan konsep BSBI dimana 
        hanya di-mantain satu dictionary besar untuk keseluruhan block.
        Namun dalam teknik penyimpanannya digunakan srategi dari SPIMI
        yaitu penggunaan struktur data hashtable (dalam Python bisa
        berupa Dictionary)

        ASUMSI: td_pairs CUKUP di memori

        Di Tugas Pemrograman 1, kita hanya menambahkan term dan
        juga list of sorted Doc IDs. Sekarang di Tugas Pemrograman 2,
        kita juga perlu tambahkan list of TF.

        Parameters
        ----------
        td_pairs: List[Tuple[Int, Int]]
            List of termID-docID pairs
        index: InvertedIndexWriter
            Inverted index pada disk (file) yang terkait dengan suatu "block"
        """
        # TODO
        term_dict = {}
        for term_id, doc_id in td_pairs:
            if term_id not in term_dict:
                term_dict[term_id] = dict()

            if doc_id in term_dict[term_id]:
                term_dict[term_id][doc_id] += 1
            else:
                term_dict[term_id][doc_id] = 1
                
        for term_id in sorted(term_dict.keys()):
            postings_list = sorted(list(term_dict[term_id].keys()))
            tf_list = []
            for sorted_key in postings_list:
                tf_list.append(term_dict[term_id][sorted_key])
            index.append(term_id, postings_list, tf_list)

    def merge(self, indices, merged_index):
        """
        Lakukan merging ke semua intermediate inverted indices menjadi
        sebuah single index.

        Ini adalah bagian yang melakukan EXTERNAL MERGE SORT

        Gunakan fungsi orted_merge_posts_and_tfs(..) di modul util

        Parameters
        ----------
        indices: List[InvertedIndexReader]
            A list of intermediate InvertedIndexReader objects, masing-masing
            merepresentasikan sebuah intermediate inveted index yang iterable
            di sebuah block.

        merged_index: InvertedIndexWriter
            Instance InvertedIndexWriter object yang merupakan hasil merging dari
            semua intermediate InvertedIndexWriter objects.
        """
        # kode berikut mengasumsikan minimal ada 1 term
        merged_iter = heapq.merge(*indices, key = lambda x: x[0])
        curr, postings, tf_list = next(merged_iter) # first item
        for t, postings_, tf_list_ in merged_iter: # from the second item
            if t == curr:
                zip_p_tf = sorted_merge_posts_and_tfs(list(zip(postings, tf_list)), \
                                                      list(zip(postings_, tf_list_)))
                postings = [doc_id for (doc_id, _) in zip_p_tf]
                tf_list = [tf for (_, tf) in zip_p_tf]
            else:
                merged_index.append(curr, postings, tf_list)
                curr, postings, tf_list = t, postings_, tf_list_
        merged_index.append(curr, postings, tf_list)

    def retrieve_tfidf(self, query, k = 10):
        """
        Melakukan Ranked Retrieval dengan skema TaaT (Term-at-a-Time).
        Method akan mengembalikan top-K retrieval results.

        w(t, D) = (1 + log tf(t, D))       jika tf(t, D) > 0
                = 0                        jika sebaliknya

        w(t, Q) = IDF = log (N / df(t))

        Score = untuk setiap term di query, akumulasikan w(t, Q) * w(t, D).
                (tidak perlu dinormalisasi dengan panjang dokumen)

        catatan: 
            1. informasi DF(t) ada di dictionary postings_dict pada merged index
            2. informasi TF(t, D) ada di tf_li
            3. informasi N bisa didapat dari doc_length pada merged index, len(doc_length)

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
        stop_words = set(stopwords.words('english'))

        # tokenize
        tokenizer = RegexpTokenizer(r'\w+')
        rem_num = re.sub('[0-9]+', '', query)
        query_term = tokenizer.tokenize(rem_num)
        filtered = [stemmer.stem(t) for t in query_term if not t.lower() in stop_words]
        heap = []

        with InvertedIndexReader(self.index_name, directory=self.output_dir, postings_encoding=
                                 self.postings_encoding) as mapper:
            
            N = len(mapper.doc_length)
            for term in filtered:
                # handle term yg tidak ada di collection
                if self.term_id_map[term] not in mapper.postings_dict:
                    continue

                scores_per_doc = []
                df = mapper.postings_dict[self.term_id_map[term]][1]
                wtq = math.log(N / df, 10)
                postings_list, tf_list = mapper.get_postings_list(self.term_id_map[term])
                for i in range(df):
                    wtd = 0
                    if tf_list[i] > 0:
                        wtd = 1 + math.log(tf_list[i], 10)
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
        stop_words = set(stopwords.words('english'))

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

    def index(self):
        """
        Base indexing code
        BAGIAN UTAMA untuk melakukan Indexing dengan skema BSBI (blocked-sort
        based indexing)

        Method ini scan terhadap semua data di collection, memanggil parse_block
        untuk parsing dokumen dan memanggil invert_write yang melakukan inversion
        di setiap block dan menyimpannya ke index yang baru.
        """
        # loop untuk setiap sub-directory di dalam folder collection (setiap block)
        for block_dir_relative in tqdm(sorted(next(os.walk(self.data_dir))[1])):
            td_pairs = self.parse_block(block_dir_relative)
            index_id = 'intermediate_index_'+block_dir_relative
            self.intermediate_indices.append(index_id)
            with InvertedIndexWriter(index_id, self.postings_encoding, directory = self.output_dir) as index:
                self.invert_write(td_pairs, index)
                td_pairs = None
    
        self.save()

        with InvertedIndexWriter(self.index_name, self.postings_encoding, directory = self.output_dir) as merged_index:
            with contextlib.ExitStack() as stack:
                indices = [stack.enter_context(InvertedIndexReader(index_id, self.postings_encoding, directory=self.output_dir))
                               for index_id in self.intermediate_indices]
                self.merge(indices, merged_index)


if __name__ == "__main__":

    BSBI_instance = BSBIIndex(data_dir = 'collection', \
                              postings_encoding = VBEPostings, \
                              output_dir = 'index')
    BSBI_instance.index() # memulai indexing!
