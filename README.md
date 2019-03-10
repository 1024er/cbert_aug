# cbert_aug
This is a implementation of paper "Conditional BERT Contextual Augmentation" https://arxiv.org/pdf/1812.06705.pdf.
Our original implementation was two-stage, for convenience, we rewrite the code. 

The *global.config* contains the global configuration for bert and classifier.
The datasets directory contains files for bert, and the aug_data directory contain augmented files for classifier.

You can run the code in two ways. Our default way is the second.
  - You can load the original bert and run aug_dataset_wo_ft.py directly
    - python aug_dataset_wo_ft.py.py
  - Or you can finetune bert on each dataset before run aug_dataset.py, and then load fine-tuned bert in aug_dataset.py
    1. python finetune_dataset.py
    2. python aug_dataset.py

The hyperparameters of the models and training were selected by a grid-search using baseline models without data augmentation in each task’s validation set individually.

We show the process of augmenting stsa.binary and training rnn-classifier on it, in aug_data/stsa.binary directory.

The classifier code is from <https://github.com/pfnet-research/contextual_augmentation>, thanks to the author.
