# cbert_aug

Thanks @liuyaxin 's effort to rewrite the code with huggingface's latest transformer library.
If you want to reproduce the results in paper, you can switch to the develop branch.


We arrange the original code of cbert from https://github.com/1024er/cbert_aug.git. 
Our original implementation was two-stage, for convenience, we rewrite the code. 

The *global.config* contains the global configuration for bert and classifier.
The datasets directory contains files for bert, and the aug_data directory contain augmented files for classifier.

You can run the code by: 

1.finetune bert on each dataset before run cbert_augdata.py

  ```python cbert_finetune.py```
  
  you can use *python cbert_finetune.py --task_name='TREC'* to change the task you want to perform, you can also set your own parameters in the same way to acquire different results.
  
2.then load fine-tuned bert in cbert_augdata.py

  ```python cbert_augdata.py```
  
  notice that if you want to change the default dataset used in original code, you have to alter the parameter "dataset" in *global.config* firstly.

The hyperparameters of the models and training were selected by a grid-search using baseline models without data augmentation in each task’s validation set individually.

We upload the runing log with dropout=0.5 for all datasets, this is very close to the results in paper. You can achieve the results in paper by grid-search the hyperparameters.

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
