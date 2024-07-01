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

@misc{ousidhoum2024semeval,
      title={SemEval Task 1: Semantic Textual Relatedness for African and Asian Languages}, 
      author={Nedjma Ousidhoum and Shamsuddeen Hassan Muhammad and Mohamed Abdalla and Idris Abdulmumin and Ibrahim Said Ahmad and Sanchit Ahuja and Alham Fikri Aji and Vladimir Araujo and Meriem Beloucif and Christine De Kock and Oumaima Hourrane and Manish Shrivastava and Thamar Solorio and Nirmal Surange and Krishnapriya Vishnubhotla and Seid Muhie Yimam and Saif M. Mohammad},
      year={2024},
      eprint={2403.18933},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```