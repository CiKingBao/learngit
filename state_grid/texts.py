# -*- coding: utf-8 -*-
"""
File: texts.py
Date: 2018-05-27 10:30
Author: amy

Text preprocess.
"""
import re
import sys
import logging

from config import Config

logger = logging.getLogger("console_file")


class TextPreprocessor(object):
    """
    Class for all text preprocess steps.
    """
    sentence_end_tok = "。？…！!?"

    @staticmethod
    def _remove_urls(s):
        return re.sub(r"http\S+", "", s)

    @staticmethod
    def _replace_long_white_spaces(s):
        return re.sub(r"\s+", " ", s)

    @staticmethod
    def _remove_punctuations(s):
        return re.sub(r"\W+", " ", s)

    @staticmethod
    def _remove_html_tags(s):
        return re.sub(r"<[/\w]*>", "", s)

    @staticmethod
    def _remove_useless_words(wl, stop_words=False):
        if stop_words is True:
            wl = [w for w in wl if w not in Config.stop_words]

        wl = [w for w in wl if re.match(r"[-.%+\d]+", w) is None]

        return wl

    @classmethod
    def split_sub_sentence(cls, text):
        """
        Split long sentence (len > Config.max_sentence_len) into sub sentences (len ≈ Config.min_sub_sentence_len)。
        Split by puncuations.
        Make the sub sentences proper lens.

        e.g. 不过，我对这新电表电费就不明白了: 家里因为两口子早上七点就离开家去上班、孩子去上学，中午都不回家。
        ==> ["不过, "我对这新电表电费就不明白了。", "家里因为两口子早上七点就离开家去上班。", "孩子去上学，中午都不回家。"]
        ""
        Args:
            text: str

        Returns:
            sub_sentences: list of str
        """
        # split by lens
        if len(text) <= Config.max_sentence_len:
            return [text]
        sub_sentences = [s for s in re.split(r"\W+", text) if len(s) > 0]
        merged_sub_sentences = []
        cnt = len(sub_sentences)
        i = 0
        while i < cnt:
            sen = sub_sentences[i]
            while i + 1 < cnt and len(sen) < Config.min_sub_sentence_len:
                i += 1
                sen += '，' + sub_sentences[i]
            if sen[-1] not in cls.sentence_end_tok:
                sen += '。'
            merged_sub_sentences.append(sen)
            i += 1
        return merged_sub_sentences

    @classmethod
    def clean_text(cls, text):
        """
        Clean raw text:
        1. strip
        2. remove url
        3. remove html tags: e.g. <div> <p>
        4. shorten consecutive white chars: " \t\r\n" => " "
        5. replace consecutive eos tokens => "。"
        6. make each text ends with a "。"

        Args:
            text: str

        Returns:
            preprocessed text: str
        """
        if not isinstance(text, str):
            try:
                text = str(text)
            except TypeError:
                text = ""
        text = text.strip()
        text = cls._remove_urls(text)
        text = cls._remove_html_tags(text)
        text = cls._replace_long_white_spaces(text)
        sentences = re.split(r"[{}]".format(cls.sentence_end_tok), text)
        text = "。".join([s for s in sentences if len(s) > 0])
        # add <EOS> for each doc
        if len(text) > 0:
            text += "。"
        else:
            text = ""
        return text

    @classmethod
    def clean_text_remove_puncs(cls, text):
        """
        Preprocess one text.
        1. clean text
        2. remove all punctuations
        Args:
            text: str

        Returns:
            preprocessed text: str
        """
        if isinstance(text, str):
            try:
                text = cls.clean_text(text)
                text = cls._remove_punctuations(text)

            except (TypeError, AttributeError):
                logger.warning("Error occured in preprocessing text: {}".format(text))
                sys.exit(0)

            return text.strip()

        return ""

    @classmethod
    def text_2_word_list(cls, text, stop_words=False):
        """
        Cut text to word list using config tokenizer.
        Args:
            text: str
            stop_words: remove stop_words or not

        Returns:
            word_list: list of str
        """
        # cut words and filter words
        word_list = list(Config.tokenizer.segment(text))
        word_list = cls._remove_useless_words(word_list, stop_words=stop_words)

        return word_list

    @classmethod
    def preprocess_pipline(cls, text, stop_words=False):
        """
        Preprocess pipline from raw text to word list.
        Args:
            text: str
            stop_words: remove stop_words or not

        Returns:
            word_list: list of str
        """
        text = cls.clean_text_remove_puncs(text)
        wl = cls.text_2_word_list(text, stop_words=stop_words)
        return wl


class LTPTokenizer:
    logger.info("using ltp tokenizer.")

    def tokenize(self, text):
        """
        Cut words.
        Args:
            text: str

        Returns:
            word list
        """
        return Config.tokenizer.segment(text)
