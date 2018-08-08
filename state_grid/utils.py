# -*- coding: utf-8 -*-
"""
File: utils.py
Date: 2018-05-27 09:36
Author: amy

Loading data from file.
"""
import os
import sys
import logging
import pandas as pd

from config import Config

logger = logging.getLogger("console_file")


def load_csv(file, *cols):
    """
    Load csv file by column names.
    Example Usage:
        data = load_csv("data/electric_corpus.csv", "title", "content")
    Args:
        file: csv file
        *cols: column names

    Returns:
        [[tiltle1, content1], [title2, content2], ...]
    """
    columns = list(cols)
    try:
        if len(cols) == 0:
            df = pd.read_csv(file, encoding=Config.default_encoding)
        else:
            df = pd.read_csv(file, encoding=Config.default_encoding)[columns]

    except FileNotFoundError:
        logger.error("File [{}] doesn't exist!".format(file))
        sys.exit(0)

    except KeyError:
        logger.error("Illegal [csv column name] in {}! ".format(columns))
        sys.exit(0)

    ret = [vals.tolist() for index, vals in df.iterrows()]
    return ret


def load_text(file):
    with open(file, "r", encoding="utf-8") as fr:
        return fr.read()


def load_word_set(file):
    return set([w.strip() for w in open(file, "r", encoding=Config.default_encoding)])


def infer_format(docs):
    if isinstance(docs, list) and isinstance(docs[0], list):
        # [[title1, content1], [title2, content2], ...]
        input_format = 1
    elif isinstance(docs, list) and isinstance(docs[0], str):
        # [content1, content2, ...]
        input_format = 2
    else:
        input_format = -1
        raise ValueError("Illegal input type for docs. \n "
                         "Please pass: [[title1, content1], [title2, content2], ...] or [content1, content2, ...]")
    return input_format


def clear_dir(des_dir):
    """Delete all files and subdirs under des_dir."""
    for root, dirs, files in os.walk(des_dir):
        for f in files:
            os.remove(os.path.join(root, f))
        for d in dirs:
            os.rmdir(os.path.join(root, d))
    logger.info("cleaning dir: [{}] is empty now.".format(des_dir))
    return
