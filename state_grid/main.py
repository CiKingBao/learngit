# -*- coding: utf-8 -*-
"""
File: main.py
Date: 2018-05-24 10:01
Author: amy

Entry of the project.
"""
import os
import sys
import logging

import utils
from config import Config
from relevant import RelevantDoc
from cluster import Cluster
from summary import Summary

logger = logging.getLogger("console_file")


def analysis_pipeline():
    Config.prepare_env()

    # 1. find relevant docs
    x = input("Are you sure you want to [find relevant docs]? (y/n)")
    logger.debug("Step 1. user input: {}".format(x))
    if x == "y" or x == "Y":
        logger.info(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        logger.info("Step 1. [find relevant docs]")
        data = utils.load_csv(Config.corpus_file_csv, "title", "content")[:5000]
        logger.info("csv row num: {:,}".format(len(data)))
        rel_doc = RelevantDoc(data)
        rel_doc.find_relevant_texts(output_file="output/relevant/relevant_docs.csv")

    # 2. cluster docs
    x = input("Are you sure you want to [cluster documents]? (y/n)")
    logger.debug("user input: {}".format(x))
    if x == "y" or x == "Y":
        alg = input("Which cluster method you want to use a:[kmeans], b:[dbscan] ? (a/b)")
        logger.debug("Step 2. user input: {}".format(alg))
        alg_dic = {"a": "kmeans", "b": "dbscan"}
        if alg in alg_dic:
            alg = alg_dic[alg]
        else:
            raise ValueError("Invalid cluster method choice: {}".format(alg))
        logger.info(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        logger.info("Step 2. [cluster docs]")
        data = utils.load_csv("output/relevant/relevant_docs.csv", "score", "text")
        utils.clear_dir(Config.cluster_dir)
        cluster = Cluster(data)
        cluster.cluster_docs(alg)

    # 3. text summarization
    x = input("Are you sure you want to [summarize documents]? (y/n)")
    logger.debug("Step 3. user input: {}".format(x))
    if x == "y" or x == "Y":
        logger.info(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        logger.info("Step 3. [summary docs]")
        x = input("Use a:[original documents] or b:[cutted doc pieces (shorter)]? (a/b)")
        utils.clear_dir(Config.summary_dir)
        root = Config.cluster_dir
        des_dir = Config.summary_dir
        for f in sorted(os.listdir(root)):
            if os.path.isfile(os.path.join(root, f)):
                if not f.endswith(".out"):
                    continue
                src_file = os.path.join(root, f)
                out_file = os.path.join(des_dir, f[:f.rfind(".out")] + ".sum")
                summary_model = Summary()
                if x == "b" or x == "B":
                    summary_model.split_long_docs(src_file)
                    summary_model.summary_docs(src_file=src_file + ".split", output_file=out_file)
                else:
                    summary_model.summary_docs(src_file=src_file, output_file=out_file)
        logger.info("summaries for each cluster are saved in dir: {}".format(Config.summary_dir))
        logger.info("finish summary all.")

    Config.release_resources()
    logger.info("finish pipeline.")


if __name__ == "__main__":
    analysis_pipeline()
    sys.exit(0)
