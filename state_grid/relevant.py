# -*- coding: utf-8 -*-
"""
File: relevant.py
Date: 2018-05-24 10:39
Author: amy

Find relevant texts based on word dict.
"""
import os
import logging
import numpy as np
import csv
from itertools import chain
from sklearn.feature_extraction.text import TfidfVectorizer

import utils
from config import Config
from texts import TextPreprocessor

logger = logging.getLogger("console_file")


class RelevantDoc(object):
    """
    Evaluate relevance score for each doc.
    The input format can only be one of below:
        1. [[title1, content1, ...], [title2, content2, ...], ...]
        2. [content1, content2, ...]
    Input: csv file A
    Output: csv file B  line format: title_score1, title1, content_score1, content1, ...
    """
    original_docs = []
    preprocessed_texts = []
    preprocessed_word_lists = []
    doc_vectors = []
    doc_scores = []
    input_format = None
    tfidf_vectorizer = None
    word_2_idx = None
    idx_2_word = None
    target_word_set = None
    seed_word_set = None

    @classmethod
    def __init__(cls, docs):
        cls.input_format = utils.infer_format(docs)  # input file format
        cls.original_docs = cls.load_docs(docs)

    @classmethod
    def load_docs(cls, docs):
        if cls.input_format == 1:
            docs = list(chain(*docs))
        return [TextPreprocessor.clean_text(doc) for doc in docs]

    @classmethod
    def train_tfidf_vectorizer(cls):
        # train tfidf vectorizer
        if not cls.tfidf_vectorizer:
            def identity(wl):
                return wl

            tfidf = TfidfVectorizer(analyzer=identity,  # no preprocess because we have preprocessed it already
                                    min_df=2,
                                    norm="l2")

            logger.info("training a new tfidf vectorizer with current preprocessing of docs.")
            cls.tfidf_vectorizer = tfidf.fit(cls.preprocessed_word_lists)
        else:
            logger.info("using pretrained tfidf vectorizer.")
        logger.info("tfidf vectorizer is ready.")

        cls.word_2_idx = cls.tfidf_vectorizer.vocabulary_
        cls.idx_2_word = {v: k for k, v in cls.word_2_idx.items()}

    @classmethod
    def filter_word_set(cls):
        # only keep the word set contains that appeared in this corpus
        appeared_word_set = set([k for k, v in cls.word_2_idx.items()])
        target_word_set = utils.load_word_set(Config.word_dict_file_1)
        seed_word_set = utils.load_word_set(Config.word_dict_file_2)
        cls.target_word_set = set.intersection(target_word_set, appeared_word_set)
        cls.seed_word_set = set.intersection(seed_word_set, appeared_word_set)

    @classmethod
    def eval_a_doc(cls, doc_csr):
        """
        Score a doc's relevance.
        Args:
            doc_csr: doc tfidf vector scipy csr matrix

        Returns:
            doc_score: float, doc relevant score
        """

        doc_words_cnt = doc_csr.getnnz()
        top_k_word_num = int(doc_words_cnt * Config.top_k_word_percentile)
        idxes_0, idxes_1 = doc_csr.nonzero()
        idxes = [x for x in zip(list(idxes_0), list(idxes_1))]
        scores = [doc_csr[idx] for idx in idxes]
        sorted_ptr = np.argsort(scores)[::-1]
        sorted_idxes = [idxes[i][1] for i in sorted_ptr]
        sorted_scores = [scores[i] for i in sorted_ptr]
        sorted_words = [cls.idx_2_word[idx] for idx in sorted_idxes]

        # calc doc score
        doc_score = 0.0
        seed_word_times = 1. + Config.seed_word_extra_points

        for i, w in enumerate(sorted_words):
            if w in cls.seed_word_set:
                doc_score += sorted_scores[i] * seed_word_times
            if w in cls.target_word_set:
                doc_score += sorted_scores[i]
                if i < top_k_word_num:
                    doc_score += sorted_scores[i] * Config.top_k_extra_points

        return doc_score / np.log2(doc_words_cnt + 1.1)

    @classmethod
    def find_relevant_texts(cls, output_file=os.path.join(Config.relevant_dir, "relevant_docs.csv")):
        """
        Evaluate relevance of each text.
        Save "score \t text" to file.
        """
        # preprocess text
        logger.info("start preprocessing texts ...")
        cls.preprocessed_texts = [TextPreprocessor.clean_text_remove_puncs(doc) for doc in cls.original_docs]

        # text => word list
        logger.info("start text to word list ...")
        cls.preprocessed_word_lists = [TextPreprocessor.text_2_word_list(doc) for doc in cls.preprocessed_texts]

        # word list => vector
        logger.info("start word list to doc vector ...")
        cls.train_tfidf_vectorizer()
        cls.doc_vectors = cls.tfidf_vectorizer.transform(cls.preprocessed_word_lists)

        # slim word set
        logger.info("start filtering word set ...")
        cls.filter_word_set()

        # calculate doc relevant scores
        logger.info("start calculating relevant scores ...")
        with open(output_file, "w", encoding=Config.default_encoding) as fw:
            wr = csv.writer(fw)
            header = ["score", "text"]
            wr.writerow(header)
            for i, doc_v in enumerate(cls.doc_vectors):
                doc_score = cls.eval_a_doc(doc_v)
                doc_score = float("%.4f" % doc_score)
                cls.doc_scores.append(doc_score)
                wr.writerow([doc_score, cls.original_docs[i]])
        logger.info("finished.")
        logger.info("docs' relevant scores are saved in: {}".format(output_file))
