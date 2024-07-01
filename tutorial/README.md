# SemEval2024-STR - Tutorial
Welcome to the SemEval 2024 Task 1: Semantic Textual Relatedness (STR) instructions repository. This repository is designed to guide participants through the task, providing resources, source code, and essential documentation for effective participation.

## Dependencies
+ Python >= 3.11.7
+ jupyterlab >= 4.0.9
+ numpy >= 1.26.2
+ pandas >= 2.1.4

## Directory
+ **res** - Resources including datasets, model weights, and experiment records
+ **src** - Source code including methods, models, trainers, and utility functions
+ **main.py** - The primary Python script to execute tasks and run experiments
+ **config.py** - Configuration file containing essential settings

## Setup
Please ensure required packages are already installed. A virtual environment is recommended.
```sh
$ cd SemEval2024-STR
$ pip install pip --upgrade
$ pip install jupyterlab
$ pip install -r requirements.txt
```
To open the instructional Jupyter Notebook:
```sh
$ jupyter lab
```
To execute the main script and generate results:
```sh
$ python main.py
```

## Authors
* **Ning Shi** - mrshininnnnn@gmail.com

## BibTex
```bibtex
@misc{ousidhoum2024semrel2024,
      title={SemRel2024: A Collection of Semantic Textual Relatedness Datasets for 14 Languages}, 
      author={Nedjma Ousidhoum and Shamsuddeen Hassan Muhammad and Mohamed Abdalla and Idris Abdulmumin and Ibrahim Said Ahmad and Sanchit Ahuja and Alham Fikri Aji and Vladimir Araujo and Abinew Ali Ayele and Pavan Baswani and Meriem Beloucif and Chris Biemann and Sofia Bourhim and Christine De Kock and Genet Shanko Dekebo and Oumaima Hourrane and Gopichand Kanumolu and Lokesh Madasu and Samuel Rutunda and Manish Shrivastava and Thamar Solorio and Nirmal Surange and Hailegnaw Getaneh Tilaye and Krishnapriya Vishnubhotla and Genta Winata and Seid Muhie Yimam and Saif M. Mohammad},
      year={2024},
      eprint={2402.08638},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}

@inproceedings{ousidhoum-etal-2024-semeval,
    title = "{S}em{E}val Task 1: Semantic Textual Relatedness for {A}frican and {A}sian Languages",
    author = "Ousidhoum, Nedjma  and
      Muhammad, Shamsuddeen Hassan  and
      Abdalla, Mohamed  and
      Abdulmumin, Idris  and
      Ahmad, Ibrahim Said  and
      Ahuja, Sanchit  and
      Aji, Alham Fikri  and
      Araujo, Vladimir  and
      Beloucif, Meriem  and
      De Kock, Christine",
    editor = {Ojha, Atul Kr.  and
      Do{\u{g}}ru{\"o}z, A. Seza  and
      Tayyar Madabushi, Harish  and
      Da San Martino, Giovanni  and
      Rosenthal, Sara  and
      Ros{\'a}, Aiala},
    booktitle = "Proceedings of the 18th International Workshop on Semantic Evaluation (SemEval-2024)",
    month = jun,
    year = "2024",
    address = "Mexico City, Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.semeval-1.272",
    pages = "1963--1978",
    abstract = "We present the first shared task on Semantic Textual Relatedness (STR). While earlier shared tasks primarily focused on semantic similarity, we instead investigate the broader phenomenon of semantic relatedness across 14 languages: Afrikaans, Algerian Arabic, Amharic, English, Hausa, Hindi, Indonesian, Kinyarwanda, Marathi, Moroccan Arabic, Modern Standard Arabic, Punjabi, Spanish, and Telugu. These languages originate from five distinct language families and are predominantly spoken in Africa and Asia {--} regions characterised by the relatively limited availability of NLP resources. Each instance in the datasets is a sentence pair associated with a score that represents the degree of semantic textual relatedness between the two sentences. Participating systems were asked to rank sentence pairs by their closeness in meaning (i.e., their degree of semantic relatedness) in the 14 languages in three main tracks: (a) supervised, (b) unsupervised, and (c) crosslingual. The task attracted 163 participants. We received 70 submissions in total (across all tasks) from 51 different teams, and 38 system description papers. We report on the best-performing systems as well as the most common and the most effective approaches for the three different tracks.",
}
```