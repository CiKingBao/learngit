# -*- coding: utf-8 -*-
"""
File: summary.py
Date: 2018-05-01 14:00
Author: amy

Unsupervised summary model: TextRank.
"""
import logging
import numpy as np
import networkx as nx
import multiprocessing
from collections import Counter
from gensim.models import KeyedVectors
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import pairwise_distances

from config import Config
from texts import TextPreprocessor

logger = logging.getLogger("console_file")


class Summary(object):
    summary_sentence_cnt = Config.summary_sentence_num
    lang = Config.default_lang

    @staticmethod
    def split_long_docs(src_file):
        """
        Texts to word lists.
        """
        logger.info("start preprocessing texts.")
        new_file = src_file + ".split"
        with open(src_file, "r", encoding=Config.default_encoding) as fr, \
                open(new_file, "w", encoding=Config.default_encoding) as fw:

            for raw_line in fr.readlines():
                line = raw_line.strip()
                sub_sentences = TextPreprocessor.split_sub_sentence(line)
                for s in sub_sentences:
                    fw.write(s + "\n")

    @staticmethod
    def summary_docs(src_file, output_file):
        line_num = len(open(src_file, "r", encoding=Config.default_encoding).readlines())
        if line_num <= Config.summary_sentence_num:
            logger.warning("not enough lines for summary, skip [{}]!".format(src_file))
        else:
            logger.info("start summarying doc: {} ...".format(src_file))
            txt_rnk = TextRank()
            txt_rnk.init_graph(src_file=src_file)
            txt_rnk.add_edges()
            txt_rnk.solve()
            sentences = txt_rnk.rank_vertices(Config.summary_sentence_num)
            with open(output_file, "w", encoding=Config.default_encoding) as fw:
                for line in sentences:
                    fw.write(line + "\n")
            logger.info("finish summarization for doc: {}".format(output_file))


class TextRank(object):
    """
    TextRank automatic summarization.
    1. Add vertex(text) to undirected graph.
    2. Compute edge weight(similarity).
    3. Recursivly solve till error rate below threshold.
    4. Rank vertices and output chosen sentences.
    """
    converge_threshold = 1e-6  # when error rate of each vertex <= it, the iteration will stop

    def __init__(self):
        self.G = nx.Graph()  # empty graph
        self.ranked_nodes = {}  # ranked nodes dict {node: value}
        self.node_num = 0
        self.word_lists = None
        self.similarity_type = "intersect"

    def init_graph(self, src_file):
        """
        Load sentences from src_file;
        Add them to G as vertices.
        """
        with open(src_file, "r", encoding=Config.default_encoding) as fr:
            node_id = 0
            for line in fr:
                sentence = line.strip()
                wl = self._cut_words(sentence)
                if len(wl) >= Config.min_sum_sentence_len:
                    self.G.add_node(node_id, sentence=sentence, word_list=wl)
                    node_id += 1

        self.node_num = self.G.number_of_nodes()
        self.word_lists = [v for k, v in nx.get_node_attributes(self.G, "word_list").items()]
        logger.info("node num: {:,}".format(self.node_num))
        if self.node_num <= Config.summary_sentence_num:
            logger.warning("Valid sentence num {} <= setting.ini - summay_sentence_num {}, "
                           "Don't need to do summary!".format(self.node_num, Config.summary_sentence_num))
        return

    @staticmethod
    def _cut_words(sentence):
        """
        Cut sentence into word list.
        And Remove stop words.
        """
        word_list = TextPreprocessor.text_2_word_list(sentence, stop_words=True)
        return word_list

    def add_edges(self):
        """
        Compute similarity between each sentence;
        Add and edge when similarity >= threshold.
        """
        sentence_similarity = SentenceSimilarity(self.word_lists)
        sim_type = self.similarity_type
        threshold = sentence_similarity.similarity_func_dict[sim_type]["threshold"]
        logger.info("adding edges ...")
        logger.info("similarity type: {}, threshold: {}".format(sim_type, threshold))
        pairwise_dists_matrix = sentence_similarity.compute_similarity_matrix(similarity_type=sim_type)
        i_idxes, j_idxes = np.where(pairwise_dists_matrix >= threshold)

        for node_a, node_b in zip(i_idxes, j_idxes):
            if node_a < node_b:
                self.G.add_edge(node_a, node_b, weight=pairwise_dists_matrix[node_a][node_b])
        logger.info("edge num: {:,}".format(self.G.number_of_edges()))

        return

    def solve(self):
        """
        Solve the problem recursively till it is convergent.
        Similar to PageRank's random walk model.

        Return:
            dict: {node_id: score}
        """
        logger.info("start text ranking ...")
        self.ranked_nodes = nx.pagerank(self.G, max_iter=100, tol=self.converge_threshold)
        return

    def rank_vertices(self, k):
        """
        Rank vertices.
        Args:
            k: num of output summary sentences

        Returns:
            top k representative sentences.
        """
        ranked_nodes_tmp = {node_id: score * 1e5 for node_id, score in self.ranked_nodes.items()}
        sorted_nodes = sorted(ranked_nodes_tmp, key=ranked_nodes_tmp.get, reverse=True)
        nodes = {node: ranked_nodes_tmp[node] for node in sorted_nodes}
        self.ranked_nodes = nodes
        summary = []
        i = 0
        for node_id, score in self.ranked_nodes.items():
            sentence = self.G.nodes[node_id]["sentence"]
            if sentence not in summary:
                summary.append(sentence)
                logger.debug("[id:{}]: {:.4f}: {}".format(node_id, score, sentence))
                i += 1
                if i >= k:
                    break

        return summary

    def __str__(self):
        """
        Output graph structure.
        """
        description = ""
        for edge_id, edge_attrs in self.G.edges.data():
            description += "edge [{}]: {}\n".format(edge_id, edge_attrs)
        return description


class SentenceSimilarity(object):
    """
    Compute different type of sentence similarity.
    Types including:
        Intersect, Jaccard, cosine, tfidf-emb, sif-emb.
    """
    epsilon = 1e-3
    wv = None  # word embeddings
    njobs = Config.use_cpu_num  # -1: use all cpus to parallel

    def __init__(self, word_lists):
        self.word_lists = word_lists
        self.tfidf_vectorizer = None
        self.count_vectorizer = None
        self.word2idx = None
        self.similarity_func_dict = {
            "intersect": {"func": self._intersect_similarity, "threshold": 0.4},
            "jaccard": {"func": self._jaccard_similarity, "threshold": 0.4},
            "cosine": {"func": self._cosine_similarity, "threshold": 0.6},
            "tfidf": {"func": self._cosine_similarity, "threshold": 0.6},
            "sif": {"func": self._cosine_similarity, "threshold": 0.6},
        }
        # parallel info
        usable_cpu_cnt = multiprocessing.cpu_count()
        if self.njobs < -usable_cpu_cnt:
            self.njobs = 1
        elif self.njobs > usable_cpu_cnt:
            self.njobs = -1

        if -usable_cpu_cnt <= self.njobs <= -1:
            self.njobs = -1
            self.cpu_cnt = "{}/{}".format(usable_cpu_cnt + 1 + self.njobs, usable_cpu_cnt)
        elif 1 <= self.njobs <= usable_cpu_cnt:
            self.cpu_cnt = "{}/{}".format(self.njobs, usable_cpu_cnt)
        else:
            raise ValueError(
                "cpu num can't be 0! it should be {} <= [int] <= -1 or 1 <= [int] <= {}".format(-usable_cpu_cnt,
                                                                                                usable_cpu_cnt))

    def _train_tfidf_vectorizer(self):
        """
        Train a tfidf vectorizer on current corpus.
        """
        if self.tfidf_vectorizer is None:
            def identity(wl):
                return wl

            tfidf = TfidfVectorizer(analyzer=identity,  # no preprocess because we have preprocessed it already
                                    norm="l2")

            logger.info("training a new tfidf vectorizer with current docs.")
            self.tfidf_vectorizer = tfidf.fit(self.word_lists)
            self.word2idx = self.tfidf_vectorizer.vocabulary_
        logger.info("tfidf vectorizer is ready.")

    def _train_count_vectorizer(self):
        """
        Train a count vectorizer on current corpus.
        """
        if self.count_vectorizer is None:
            def identity(wl):
                return wl

            count = CountVectorizer(analyzer=identity,  # no preprocess because we have preprocessed it already
                                    min_df=1)

            logger.info("training a new count vectorizer with current docs.")
            self.count_vectorizer = count.fit(self.word_lists)
        logger.info("count vectorizer is ready.")

    @staticmethod
    def _intersect_similarity(u, v, epsilon):
        """
        Compute the intersect similarity between sentence a and sentence b.
        Args:
            u: count vector of sentence a (np.ndarray)
            v: count vector of sentence b (np.ndarray)
            epsilon: a min float to avoid divide by zero error

        Returns:
            similarity value. (int or float)
        """
        overlap = np.sum(np.minimum(u, v))

        return (overlap * 1.0) / (np.log2(np.sum(u) + epsilon)
                                  + np.log2(np.sum(v) + epsilon))

    @staticmethod
    def _jaccard_similarity(u, v, epsilon):
        """
        Compute the Jaccard similarity between sentence a and sentence b.
        Args:
            u: count vector of sentence a (np.ndarray)
            v: count vector of sentence b (np.ndarray)
            epsilon: a min float to avoid divide by zero error

        Returns:
            similarity value. (int or float)
        """
        overlap = np.sum(np.minimum(u, v))
        union = np.sum(np.maximum(u, v))
        return (overlap * 1.0) / (union + epsilon)

    @staticmethod
    def _cosine_similarity(u, v, epsilon):
        """
        Compute the Jaccard similarity between sentence a and sentence b.
        Args:
            u: sentence a's vector (np.ndarray)
            v: sentence b's vector (np.ndarray)
            epsilon: a min float to avoid divide by zero error

        Returns:
            similarity value. (float)
        """
        return (np.dot(u, v)) / (np.linalg.norm(u) * np.linalg.norm(v) + epsilon)

    def sentence_tfidf_embedding(self, tfidf_mat):
        """
        Compute the sentence embedding based on tfidf vector and word embeddings.
        Args:
            tfidf_mat: sentence's tfidf csr matrix.

        Returns:
            sentence embedding list
        """
        logger.info("generating tfidf sentence embeddings ...")
        idx2word = {v: k for k, v in self.word2idx.items()}

        def get_csr_mat_row(row_no, mat):
            start_idx = mat.indptr[row_no]
            stop_idx = mat.indptr[row_no + 1]
            return mat.data[start_idx:stop_idx], mat.indices[start_idx:stop_idx]

        sen_embs = []
        emb_dim = self.wv.vectors.shape[-1]
        unk = np.zeros(emb_dim, dtype=float)
        for sen_idx, wl in enumerate(self.word_lists):
            tfidf, w_idxes = get_csr_mat_row(sen_idx, tfidf_mat)
            w_idxes_cnt = dict(Counter([self.word2idx[w] for w in wl]))
            embs = []
            cnts = []
            for idx in w_idxes:
                w = idx2word[idx]
                cnts.append(w_idxes_cnt[idx])
                try:
                    emb = self.wv[w]
                except KeyError:
                    emb = unk
                embs.append(emb)
            embs = np.array(embs)
            cnts = np.array(cnts)
            sen_emb = (tfidf * cnts * np.array(embs).T).sum(axis=-1) / (cnts.sum() + self.epsilon)
            sen_embs.append(sen_emb)
        sen_embs = np.array(sen_embs, dtype=np.float32)
        return sen_embs

    def sentence_sif_embedding(self, tfidf_mat):
        """
        Compute the sentence embedding based on SIF embedding.
        Ref: https://openreview.net/pdf?id=SyK00v5xx
        Args:
            tfidf_mat: sentence's tfidf csr matrix.

        Returns:
            sentence embedding list
        """
        logger.info("generating sif sentence embeddings ...")
        sen_embs = self.sentence_tfidf_embedding(tfidf_mat)

        def remove_pc(X, npc=1):
            """
            Remove the projection on the principal components
            Args:
                X: matrix.
                npc: num of principal components.

            Returns:
                XX[i, :] is the data point after removing its projection
            """

            def compute_pc(X, npc=1):
                """
                Compute the principal components without taking the data zero mean.
                Args:
                    X: matrix.
                    npc: num of principal components.

                Returns:
                    components[i,:]: the i-th vector's pc
                """
                svd = TruncatedSVD(n_components=npc, n_iter=7, random_state=0)
                svd.fit(X)
                return svd.components_

            logger.info("removing principal components ...")
            pc = compute_pc(X, npc)
            if npc == 1:
                XX = X - X.dot(pc.transpose()) * pc
            else:
                XX = X - X.dot(pc.transpose()).dot(pc)
            return XX

        sen_embs = remove_pc(sen_embs)

        return sen_embs

    @classmethod
    def load_word_embeddings(cls):
        """
        Load word embeddings of each word in the sentences.

        Returns:
            embedding list of each sentence. looks like:
            [[n-dim],
             [n-dim],
             ...
            ]  # n is the embedding dim
        """
        if cls.wv is not None:
            logger.info("using loaded word embedings ...")
            return
        logger.info("loading word embedings ...")
        wv = KeyedVectors.load_word2vec_format("resource/word2vec/news_12g_baidubaike_20g_novel_90g_embedding_64.bin",
                                               binary=True)
        cls.wv = wv
        return

    def compute_similarity_matrix(self, similarity_type):
        logger.info("start computing sentence similarities ...")
        logger.info("similarity type is: {}".format(similarity_type))
        logger.info("this may cost long time depends on your sentences num.")
        if similarity_type == "cosine":
            self._train_tfidf_vectorizer()
            _X = self.tfidf_vectorizer.transform(self.word_lists).todense()
        elif similarity_type in ["intersect", "jaccard"]:
            self._train_count_vectorizer()
            _X = self.count_vectorizer.transform(self.word_lists).todense()
        elif similarity_type == "tfidf":
            self._train_tfidf_vectorizer()
            self.load_word_embeddings()
            _X = self.tfidf_vectorizer.transform(self.word_lists)
            # compute sentence embedding based on word embeddings
            _X = self.sentence_tfidf_embedding(_X)
        elif similarity_type == "sif":
            self._train_tfidf_vectorizer()
            self.load_word_embeddings()
            _X = self.tfidf_vectorizer.transform(self.word_lists)
            # compute sentence embedding based on word embeddings
            _X = self.sentence_sif_embedding(_X)

        else:
            raise ValueError("similarity type is not defined!")

        # compute pairwise distances
        logger.info("start computing pairwise similarities ...")
        logger.info("using {} cpus ...".format(self.cpu_cnt))
        pairwise_sims_matrix = pairwise_distances(_X,
                                                  metric=self.similarity_func_dict[similarity_type]["func"],
                                                  n_jobs=self.njobs,
                                                  epsilon=self.epsilon)

        np.fill_diagonal(pairwise_sims_matrix, 0.)
        pairwise_sims_matrix[np.isnan(pairwise_sims_matrix)] = 0.
        logger.debug("sims matrix shape: {}".format(np.shape(pairwise_sims_matrix)))
        logger.debug("sims matrix: \n{}".format(pairwise_sims_matrix))
        return pairwise_sims_matrix
