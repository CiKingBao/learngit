# -*- coding: utf-8 -*-
"""
File: config.py
Date: 2018-05-24 10:56
Author: amy

Global variables and params configuation; log configuration; enviroment preparation.
"""

import logging
import configparser
from pyltp import Segmentor
from logging.config import dictConfig

# logging configuration
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'detailed': {
            'datefmt': '%Y-%m-%d %H:%M:%S',
            'format': '{asctime} [{levelname}] {module} {lineno}: {message}',
            'style': '{',
        },
        'simple': {
            'datefmt': '%H:%M:%S',
            'format': '{asctime} [{levelname}]: {message}',
            'style': '{',
        }
    },

    'handlers': {
        'file': {
            'level': 'DEBUG',
            'class': 'logging.FileHandler',
            'filename': 'log/run.log',
            'formatter': 'detailed',
            'mode': 'w'
        },
        'console': {
            'level': 'INFO',
            'class': 'logging.StreamHandler',
            'formatter': 'simple'
        },
    },
    'loggers': {
        'console_file': {
            'handlers': ['file', 'console'],
            'level': 'DEBUG',
            'propagate': False,
        },
        'console': {
            'handlers': ['console'],
            'propagate': False,
        },
    },
}

dictConfig(LOGGING)
logger = logging.getLogger("console_file")
logger.info("logs will be output to [console] and file [log/run.log].")


class Config(object):
    cfg = configparser.ConfigParser()
    cfg.read("settings.ini")

    default_encoding = cfg["default"]["default_encoding"]
    default_lang = cfg["default"]["default_lang"]

    # file paths
    stop_words_file = cfg["file"]["stop_words_file"]

    word_dict_file_1 = cfg["file"]["word_dict_file_1"]
    word_dict_file_2 = cfg["file"]["word_dict_file_2"]

    corpus_file_csv = cfg["file"]["corpus_file_csv"]
    relevant_dir = cfg["file"]["relevant_dir"]
    cluster_dir = cfg["file"]["cluster_dir"]
    summary_dir = cfg["file"]["summary_dir"]

    # word segmentation
    tokenizer = None
    stop_words = None

    # top important words will participate the relevant score computing
    top_k_word_percentile = 0.15
    top_k_extra_points = 0.5
    seed_word_extra_points = 0.5
    relevant_score_threshold = float(cfg["relevant"]["relevant_score_threshold"])  # output all docs

    # cluster
    cluser_num = int(cfg["cluster"]["cluster_num"])
    cluster_doc_num = int(cfg["cluster"]["cluster_doc_num"])
    reduct_dimension = int(cfg["cluster"]["reduct_dimension"])
    eps = float(cfg["cluster"]["eps"])

    # summary
    summary_sentence_num = int(cfg["summary"]["summary_sentence_num"])

    max_sentence_len = int(cfg["summary"]["max_sentence_len"])
    min_sub_sentence_len = int(cfg["summary"]["min_sub_sentence_len"])
    min_sum_sentence_len = int(cfg["summary"]["min_sum_sentence_len"])
    use_cpu_num = int(cfg["summary"]["use_cpu_num"])

    @classmethod
    def _load_ltp_models(cls):
        """
        Load ltp segment and pos tagging models.
        """
        cws_model_path = "resource/ltp_data_v3.4.0/cws.model"

        # check tokenizer initializtion
        if cls.tokenizer is None:
            cls.tokenizer = Segmentor()
            cut_word_file = "data/cut_word.txt"
            with open(Config.word_dict_file_1, "r", encoding=Config.default_encoding) as fr1, \
                    open(Config.word_dict_file_2, "r", encoding=Config.default_encoding) as fr2, \
                    open(cut_word_file, "w", encoding=Config.default_encoding) as fw:
                words_1 = [w.strip() for w in fr1.readlines()]
                words_2 = [w.strip() for w in fr2.readlines()]
                words = words_1 + words_2
                fw.write("\n".join(words))

            cls.tokenizer.load_with_lexicon(cws_model_path, cut_word_file)

        logger.info("LTP models loaded.")

    @classmethod
    def _release_ltp_models(cls):
        if cls.tokenizer:
            cls.tokenizer.release()

        logger.info("LTP models released.")

    @classmethod
    def _load_seed_set(cls):
        return set([w.strip() for w in open(cls.word_dict_file_1, "r", encoding=cls.default_encoding)])

    @classmethod
    def _load_stopwords_set(cls):
        cls.stop_words = set()
        with open(Config.stop_words_file, "r", encoding=Config.default_encoding) as fr:
            for line in fr:
                cls.stop_words.add(line.strip())
        return cls.stop_words

    @classmethod
    def prepare_env(cls):
        cls._load_ltp_models()
        cls._load_stopwords_set()

    @classmethod
    def release_resources(cls):
        cls._release_ltp_models()
