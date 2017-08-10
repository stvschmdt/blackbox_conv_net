# blackbox_convolutional_net
## researching adversarial attacks to classifiers
[about] 

many decisions around us in the world today deploy a classifier(s) for decision making - self driving cars, IRS tax fraud teams, MS Windows Defender, credit fraud detection or credit applications...basically anything which sees too many inputs to manually handle depends on a feature set of attributes for a machine to learn parameter on which to make decisions like true/false, good/bad, hot dog/not hog dog or multi class learning as in airplane/giraffe/truck or text digits 5/6/3/2/etc and the source code for this is mostly hidden and locked away for IP sake. This collection of parameters, hyperparameters comes with much time of tuning, data collection and creation of various ensemble learning models to make one program or decision. The goal with black box attacks is to model the model - a meta-model if you will, saying let's use this original model (software/hardware) as an oracle and see what you classify a set of inputs as (mind you these could be right or wrong resulting classifications), to learn your decision boundaries. Once these decision boundaries are learned, one can mathematically modify the input slightly so as to cause both the meta-model and thusly (transferability) the original decision making model to misclassify this controlled modification. i.e. changing several pixels in a 5 to get the original model to say it is a 6 or an 8. here is an exploration of several known and new methods of creating the attack as the 'most important' element is that this perturbation is indistinguishable to a human observer. meaning a human who double checks the work of a machine would say post hoc 'eh, i can't really tell if that's a sloppy 6 or a sloppy 5, so I will trust the machine decision' or a priori 'this stop sign doesn't look tampered with, so i will allow it to be passed into the machine for decision on if its a stop sign'. this is key as secondary systems or preprocessing systems (thinking of you, humans), may say, 'wow that image of a stop sign was repainted yellow, i bet that will cause a problem for the machine', whereas an image of a stop sign with a couple small smudges on it, would be allowed through to the decision module. the goal is high transferability - we can't keep sending forged inputs to an oracle without raising some flags, and ease of crafting these attack inputs to use minimal computing power which hinges on mathematically solving for minimal perturbations which mimic the human philosophical question of plausibility of the attack input.  

[usage] running the program from command line example

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
  --schmidt             use schmidt modification

[dependencies] python packages

numpy
scikit learn
tensorflow (cuda gpu not necessary but extremely helpful)

[less about] convolutional neural net modeling black box ML algorithms with adversarial transferability

 - implementation of collection of papers/talks by Ian Goodfellow et al
 - black box NN and SVM to use as orcale
 - CNN models against oracle
 - CNN imitates oracle to find breaking point in decision boundary
 - CNN attacks oracle for transferability
