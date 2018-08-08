# -*- coding: utf-8 -*-
"""
File: cluster.py
Date: 2018-06-03 17:42
Author: amy

Clustering texts and find n nearest texts to the cluster centroids.
"""
import numpy as np
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline

import utils
from config import Config
from texts import TextPreprocessor

logger = logging.getLogger("console_file")

logger = logging.getLogger("console_file")


class Cluster(object):
    original_docs = []
    preprocessed_texts = []
    preprocessed_word_lists = []
    doc_vectors = []
    doc_scores = []
    input_format = None
    tfidf_vectorizer = None
    all_cluster_distances = None
    word_2_idx = None
    idx_2_word = None
    alg = None  # cluster algorithms: kmeans, dbscan
    cluster_labels = []

    @classmethod
    def __init__(cls, docs):
        cls.input_format = utils.infer_format(docs)  # only accept (list of strs) or (list of list of strs).
        cls.original_docs = cls.load_relevant_docs(docs)

    @classmethod
    def load_relevant_docs(cls, docs):
        texts = []
        if cls.input_format == 1:
            cols = len(docs[0])
            for k, row in enumerate(docs):
                i = 0
                while i < cols // 2:
                    if row[i] > Config.relevant_score_threshold:
                        texts.append(row[i + 1].strip())
                    i += 2
        else:
            for k, row in enumerate(docs):
                if row[0] > Config.relevant_score_threshold:
                    texts.append(row[1].strip())

        logger.info("all data size: {:,}".format(len(docs)))
        logger.info("relevant data size: {:,}".format(len(texts)))
        return list(set(texts))

    @classmethod
    def preprocess_texts(cls):
        """
        Texts to word lists.
        """
        logger.info("start preprocessing texts ...")
        cls.preprocessed_texts = [TextPreprocessor.clean_text_remove_puncs(doc) for doc in cls.original_docs]
        cls.preprocessed_word_lists = [TextPreprocessor.text_2_word_list(text, stop_words=True) for text in
                                       cls.preprocessed_texts]

    @classmethod
    def transform_doc_vector(cls):
        """
        Transform word lists to tfidf vectors.
        """
        if not cls.tfidf_vectorizer:
            # train tfidf vectorizer
            def identity(wl):
                return wl

            tfidf = TfidfVectorizer(analyzer=identity,  # no preprocess because we have preprocessed it already
                                    min_df=2,
                                    norm="l2")

            logger.info("start training a new tfidf vectorizer with preprocessed docs ...")
            cls.tfidf_vectorizer = tfidf.fit(cls.preprocessed_word_lists)
        else:
            logger.info("using pretrained tfidf vectorizer.")
        logger.info("tfidf vectorizer is ready.")

        logger.info("start transforming word list to doc vector ...")
        word_lists = cls.preprocessed_word_lists
        cls.doc_vectors = cls.tfidf_vectorizer.transform(word_lists)

    @classmethod
    def dimension_redunction(cls):
        """
        Function: Use SVD to dimension redunction to speed up k-means cluster.
        Idea: Project data to lower-dimensional space that preserves most of the variance,
              by dropping the components with lower singular values.
        Note: If the dataset is too large to be kept in the memory, IncrementalPCA can be used.
        """
        logger.debug("before svd: {}".format(cls.doc_vectors.shape[-1]))
        logger.info("start dimentsion redunction ...")
        if Config.reduct_dimension > 0:
            svd = TruncatedSVD(n_components=Config.reduct_dimension)
            normalizer = Normalizer(copy=False)
            lsa = make_pipeline(svd, normalizer)
            cls.doc_vectors = lsa.fit_transform(cls.doc_vectors)
        logger.debug("after svd: {}".format(cls.doc_vectors.shape[-1]))

    @classmethod
    def output_cluster_terms(cls):
        logger.info("Top terms per cluster:")
        order_centroids = cls.alg.cluster_centers_.argsort()[:, ::-1]
        terms = cls.tfidf_vectorizer.get_feature_names()
        for i in range(Config.cluser_num):
            term_inds = order_centroids[i, :10]
            logger.info("Cluster {}: {}".format(i, ' '.join([terms[ind] for ind in term_inds])))

    @classmethod
    def kmeans_cluster(cls):
        logger.info("start kmeans clustering ...")
        cls.dimension_redunction()
        cls.alg = KMeans(n_clusters=Config.cluser_num, init="k-means++", n_init=5, max_iter=300, verbose=True)
        cls.all_cluster_distances = cls.alg.fit_transform(cls.doc_vectors)
        cls.cluster_labels = cls.alg.labels_
        if Config.reduct_dimension <= 0:
            cls.output_cluster_terms()

    @classmethod
    def dbscan_cluster(cls):
        logger.info("start dbscan clustering ...")
        cls.alg = DBSCAN(eps=Config.eps, min_samples=5, metric="cosine")
        cls.cluster_labels = cls.alg.fit_predict(cls.doc_vectors)

    @classmethod
    def output_kmeans_cluster_docs(cls, output_dir=Config.cluster_dir):
        logger.info("start outputing cluster docs ...")
        for cluster_id in range(Config.cluser_num):
            cluster_distances = cls.all_cluster_distances[:, cluster_id]
            doc_ids = np.argsort(cluster_distances)[:Config.cluster_doc_num]
            file_name = output_dir + "cluster_{}.out".format(cluster_id)
            with open(file_name, "w", encoding="utf-8") as fw:
                for doc_id in doc_ids:
                    fw.write(cls.original_docs[doc_id] + "\n")
        logger.info("cluster outputs are saved in: {}".format(output_dir))

    @classmethod
    def output_all_cluster_docs(cls, output_dir=Config.cluster_dir):
        logger.info("start outputing cluster docs ...")
        # group by cluster id
        clusters = {}
        for doc_id, cluster_id in enumerate(cls.cluster_labels):
            if cluster_id in clusters:
                clusters[cluster_id].append(doc_id)
            else:
                clusters[cluster_id] = [doc_id]

        # output docs in each cluster
        for cluster_id in sorted(clusters):
            doc_ids = clusters[cluster_id]
            logger.debug("cluster [{}] doc id num: {}".format(cluster_id, len(doc_ids)))
            file_name = output_dir + "cluster_{}.out".format(cluster_id)
            with open(file_name, "w", encoding="utf-8") as fw:
                for doc_id in doc_ids:
                    fw.write(cls.original_docs[doc_id] + "\n")
        logger.info("cluster outputs are saved in: {}".format(output_dir))

    @classmethod
    def cluster_docs(cls, alg_name="kmeans"):
        """
        Cluster docs pipeline.
        """
        cls.preprocess_texts()
        cls.transform_doc_vector()
        if alg_name == "dbscan":
            cls.dbscan_cluster()
            cls.output_all_cluster_docs()
        elif alg_name == "kmeans":
            cls.kmeans_cluster()
            cls.output_kmeans_cluster_docs()
        else:
            raise ValueError("Invalid cluster method name: {}".format(alg_name))
