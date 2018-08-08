# State grid Project 文工具本分析使用说明

## 1. 模块介绍
整个工程分为三个主要步骤，这三个步骤依次执行，构成了一个pipeline：
1. relevant.py: 过滤出和国家电网相关的文档。
2. cluster.py: 基于过滤出的文档进行文本聚类（把相似的文档聚在一起）， 聚类算法有2种可选：  
        kmeans: 基于centroid的聚类算法，需要预先制定cluster的数量，聚类之后会筛选出每个类中最核心的n个文档。特点：聚类结果中，每个类别的文档数量相同，可用于寻找语料库中的主要话题。
        dbscan: 基于密度的聚类算法，会依据语料自身的特点进行聚类，cluster的数量无需指定，聚类之后会把每个类中所的文档输出。特点：聚类结果中，类别-1代表噪声（不属于任何类，这里面的文档可能是主题分散的）， >=0 的是聚类结果。dbscan常常能发现一些规模较小的类别，而一些负面事件或舆论常常会被分到这些小的类别当中。
3. summary.py: 基于聚类生成的每个类的文档进行自动文本摘要（无监督）。采用经典extractive摘要算法：TextRank。

其他模块：
1. utils.py: load 数据，
2. texts.py: 文本处理相关的函数。
4. config.py: 基本配置，环境初始化等函数。
5. main.py: 入口函数。

其他文件：
1. settings.ini: 参数配置文件。
2. data/high_relevant_word_dict.txt： 一级词典：与国网直接相关。相关程度高。
3. data/relevant_word_dict.txt: 二级词典：与电力领域相关。相关程度一般。  
Note: 在使用时，可以根据想过滤出的内容，按需增删这两级词典。把关注度高的词加入一级词典，其他词如领域词汇可放入二级词典。这两个词典也会自动合并成分词词典(data/cut_word.txt)。

目录：
4. output/: 目录存放的是程序执行过程中生成的文件。
5. log/: 目录存放的是日志文件：日志级别：debug。 (console输出的日志级别是：info.)
6. resource/: 目录存放的是预训练model：ltp: 分词model。word2vec: 中文词向量model。

## 2. 使用方法
### 2.1. 安装说明
```shell
$ cd state_grid/  # cd 到工程的根目录
$ pip install -r requirements  # 安装所需依赖
Note:
* 安装 pyltp 时间可能较长（10到30分钟），请耐心等待
* 本项目完全基于python3 (>=3.5)
```
### 2.2. 运行方法
```shell
在state_grid/目录下执行
$ python main.py  # 运行程序
$ tail -f log/run.log  # 实时查看日志
Note:
* 每个步骤是否执行需要输入[y/n], 有的步骤需要选择一些选项[a/b]，使用时请留心命令行提示。
* 可以直接运行整个pipeline [y,y,y]；也可以单独运行某个步骤[n,n,y]，但需要确保每个步骤有格式正确的输入文件。
```
## 3. 文件格式说明
所有数据的默认编码格式是UTF-8，非此编码请先转码成UTF-8。windows 上的编码格式默认是GBK，mac和Linux通常是UTF-8。转码方法：
```python3
with open(file, 'r', encoding='gbk') as fr, open(new_file, 'w', encoding='utf-8') as fw:
    for line in fr:
        fw.write(line)
```
共有2种文件格式：   
对于有多列输出的数据采用`csv格式`存储；对于只有文本输出的数据采用`text格式`存储，每行一个文档。  
csv format: 每行是逗号(,)分隔的item, 请确保csv文件有`header`（列名）。  
text format: 每行一个文档，可以是（句子、段落、文章）。
1. `relevant.py`
        输入格式：csv
        title,content    # ===> header
        甘肃电网研究生年龄超了一年，能通过录取吗？,郁闷啊。#  ===> row
        ...
        输出格式：csv
        score,text    # ===> header
        e.g.
        0.2526,甘肃电网研究生年龄超了一年，能通过录取吗。# ===>  row
        ...
2.  `cluster.py`
        输入格式：csv
        score,text    # ===> header
        0.2526,甘肃电网研究生年龄超了一年，能通过录取吗。# ===> row
        ....
        输出格式：k个text，k是聚类数量。
        e.g.
        电动汽车行业前景怎么样-汽车服务。
        新能源汽车专用号牌来了。你想了解的都在这。
        ....
3. `summary.py`
        输入格式：text
        e.g.
        为更好促进新能源汽车发展，更好区分辨识新能源汽车，实施差异化交通管理政策，公安部决定自今年12月1日起。
        在上海，深圳等5个城市率先试点启用新能源汽车专用号牌，上海，南京，无锡，济南，深圳成首批试点城市。
        ....
        输出格式：text
        e.g.       
        会议深入讨论了经济新常态下的能源电力转型，电力体制改革，电力系统安全，企业经营，新能源发展等重大问题。
        公司专业从事电力系统自动化产品的研发，设计，生产，销售及相关工程的实现及服务，同时提供有关电力自动化系统的整体解决方案。
        ...


 ## 4. 参数设置指南
 在使用过程中，由于使用的语料不同，为了获得更好的效果，需要对一些超参数进行调节。用户可调节的参数在配置文件`setting.ini`当中。本节介绍每个模块参数的含义以及设置方法。
 1. relevant.py：   
        # 在cluster模块中，只会处理相关性得分 > 该threshold的文档
        # relevant_score_threshold = 0.0 意味着会使用所有和国网相关的文档
        # relevant_score_threshold = 0.5 意味着会使用所有和国网相关性>0.5的文档
        relevant_score_threshold = 0.0
 2. cluster.py:
        “” kmeans  参数 “”
        # kmeans聚类需要预先指定聚类的类别数量
        cluster_num = 3
        # kmeans 聚类输出每个类的核心文档数
        cluster_doc_num = 100
        # kmeans聚类之前将高维向量降维至n维
        # reduct_dimension = 0 意味着不进行降维
        # reduct_dimension > 0 意味着降维之后的向量是reduct_dimension维
        reduct_dimension = 0
        ”“ dbscan 参数”“”
        # eps: 将两个sample视为同一个类的成员的最小距离。默认的距离类型是“cosine”。eps 增大，聚类粒度更粗，聚出的类也更少；eps 减小，聚类粒度更细，聚出的类也更多。
        eps = 0.75
 3. summary.py:
        # 生成的summary的句子数量
        summary_sentence_num = 10
        # 当一个doc的长度超过max_sentence_len时，该doc会按照标点符号切分成长度>=min_sub_sentence_len的几个句子。
        max_sentence_len = 100
        min_sub_sentence_len = 40
        # 当句子长度<min_sum_sentence_len时，我们认为它的信息量太小，不把它加入摘要句子的候选集合中。
        min_sum_sentence_len = 20
        # summary时使用的cpu 数量，由于summary模块在计算时的复杂度与输入句子的数量N的平方成正比O(N^2)，因此当N很大时，使用并行计算能加快速度。
        # -1: 使用全部cpu [默认]
        # 1: 使用一个cpu
        # -k: 留（k-1）个cpu, 其余全部使用
        # k: 使用k个cpu.
        use_cpu_num = -1
