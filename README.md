# cbert_aug
This is a implementation of paper "Conditional BERT Contextual Augmentation" https://arxiv.org/pdf/1812.06705.pdf.
Our original implementation was two-stage, for convenience, we rewrite the code. 

The *global.config* contains the global configuration for bert and classifier.
The datasets directory contains files for bert, and the aug_data directory contain augmented files for classifier.

You can run the code by: 

1.finetune bert on each dataset before run aug_dataset.py

  ```python finetune_dataset.py```
  
2.then load fine-tuned bert in aug_dataset.py

  ```python aug_dataset.py```

The hyperparameters of the models and training were selected by a grid-search using baseline models without data augmentation in each task’s validation set individually.

We upload the runing log with dropout=0.5 for all datasets, this is very close to the results in paper. You can achieve the results in paper by grid-search the hyperparameters.

|                                |        | SST5 | SST2 | Subj | MPQA | RT   | TREC |       |           |
| ------------------------------ | ------ | ---- | ---- | ---- | ---- | ---- | ---- | ----- | --------- |
| First trail                    |        |      |      |      |      |      |      | mean  | Promotion |
| CNN                            |        | 41.2 | 79.4 | 91.1 | 85.1 | 75.4 | 88.4 | 76.77 |           |
|                                | +cbert | 42.5 | 80.5 | 92.5 | 87.1 | 78.2 | 91.0 | 78.63 | +1.86     |
| RNN                            |        | 39.2 | 79.7 | 93.0 | 86.0 | 76.7 | 89.8 | 77.40 |           |
|                                | +cbert | 42.6 | 82.2 | 94.2 | 87.7 | 79.0 | 91.0 | 79.45 | +2.05     |
|                                |        |      |      |      |      |      |      |       |           |
| Add   dev-set when fine-tuning |        |      |      |      |      |      |      | mean  | Promotion |
| CNN                            |        | 40.0 | 79.6 | 91.0 | 85.4 | 75.7 | 88.2 | 76.65 |           |
|                                | +cbert | 42.7 | 80.3 | 92.4 | 87.1 | 78.1 | 90.6 | 78.53 | +1.88     |
| RNN                            |        | 39.2 | 79.7 | 93.0 | 86.0 | 76.7 | 89.8 | 77.4  |           |
|                                | +cbert | 43.1 | 82.5 | 94.1 | 88.0 | 78.8 | 91.4 | 79.65 | +2.25     |


If you have any question, please open an issue.

Please cite this paper if you use this method or codes:
```sh
@inproceedings{wu2019conditional,
  title={Conditional BERT Contextual Augmentation},
  author={Wu, Xing and Lv, Shangwen and Zang, Liangjun and Han, Jizhong and Hu, Songlin},
  booktitle={International Conference on Computational Science},
  pages={84--95},
  year={2019},
  organization={Springer}
}
```




The classifier code is from <https://github.com/pfnet-research/contextual_augmentation>, thanks to the author.
