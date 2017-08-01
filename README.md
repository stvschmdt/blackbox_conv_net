# blackbox_convolutional_net: researching adversarial adjustments to classifiers

[usage] running the program from cli example
python nicefolk.py --augments 10 --iters 30 --split 400

[help] -- help optional arguments:
  -h, --help            show this help message and exit
  --data_dir DATA_DIR   directory for storing input data
  --batch_size BATCH_SIZE
                        batch size for stochastic gradient descent
  --optimize OPTIMIZE   threshold for adam optimizer
  --iters ITERS         cnn training epochs
  --augments AUGMENTS   number of attack augmentations to sample for epsilon
                        on inputs
  --split SPLIT         train test set percent split
  --fsplit FSPLIT       train test set percent split
  --nograph             turn graphics off

[dependencies] python packages
numpy
scikit learn
tensorflow (cuda gpu not necessary but extremely helpful)

[about] convolutional neural net modeling black box ML algorithms with adversarial transferability
 - implementation of collection of papers/talks by Ian Goodfellow et al
 - black box NN and SVM to use as orcale
 - CNN models against oracle
 - CNN imitates oracle to find breaking point in decision boundary
 - CNN attacks oracle for transferability
