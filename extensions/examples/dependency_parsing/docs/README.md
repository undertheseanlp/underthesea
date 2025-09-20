# [WIP] A simple experiment with Vietnamese Dependence Parsing

In this experiment, we try to build our very first Vietnamese dependency parser for underthesea. We exams both traditional methods (such as MaltParser) and deep learning methods. We also wonder how Vietnamese word embeddings (e.g. PhoBert) works on dependency parsing task [TBD]. We train and test our models in VLSP 2020 Dependency Parsing Dataset and report results in LAS, UAS score. We release our model as a function in underthesea v1.3.0 and our training code at https://github.com/undertheseanlp/playground.

### Introduction

Dependency parsing is the task of extracting a dependency parse of a sentence that represents its grammatical structure and defines the relationships between "head" words and words, which modify those heads. 

There are some deep learning methods for dependence parser tasks, begin with Manning work (2016), we focus on *biaffine attention* [[1]](#references) - a proven method which give a best result in VLSP 2019.

### Background and Related Work

These are several studies about Vietnamese Dependency Parsing

* In 2008, N.L. Minh et al: MST parser on a corpus consisting of 450 sentences.
* In 2013, N.T. Luong et al: MatlParser on a Vietnamese dependency treebank
* In 2014, N. Q. Dat et al: a new conversion method to automatically transform a constiuent-based VietTreebank in to dependency trees
* In 2017, N. K. Hieu: build BKTreebank, a dependency treebank for Vietnamese
* In 2017, a Vietnamese dependency treebank of 3,000 sentences is included for the CoNLL shared-task: "Multilingual Parsing from Raw Text to Universal Dependencies": 48 dependency labels for Vietnamese based on Stanford dependency labels set.
* In 2019, Vietnamese dependency parsing shared task in VLSP2019
* In 2020, VLSP organized the second shared task about dependency parsing

### Experiments Description

In these experiments, we use MaltParser as a baseline method. We further do some experiments with deep learning methods, especially *biaffine attention* [[1]](#references) method.

**MaltParser**

MaltParser is developed by Johan Hall, Jens Nilsson and Joankim Nivre. It is a data-driven parser generator for dependency parsing. Giving a treebank in dependency format, MaltParser can be used to induce a parser for the language of the treebank. MaltParser supports several parsing algorithms and learning algorithms, and allows user-defined models, consisting of arbitrary combinations of lexical features, part-of-speech features and dependency features. 

We run a simple MaltParser experiment with default config (see table 1)

*Table 1: Default configs for MaltParser*

```
2planar
  reduceonswitch (-2pr)                 false
config
  logfile (-lfi)                        stdout
  workingdir (  -w)                     user.dir
  name (  -c)                           dp-model-2
  logging ( -cl)                        info
  flowchart (  -m)                      learn
  type (  -t)                           singlemalt
  url (  -u)                            
covington
  allow_shift ( -cs)                    false
  allow_root ( -cr)                     true
graph
  max_sentence_length (-gsl)            256
  root_label (-grl)                     ROOT
  head_rules (-ghr)                     
guide
  features (  -F)                       
  data_split_threshold (  -T)           50
  kbest_type ( -kt)                     rank
  data_split_structure (  -s)           
  data_split_column (  -d)              
  learner (  -l)                        liblinear
  decision_settings (-gds)              T.TRANS+A.DEPREL
  classitem_separator (-gcs)            ~
  kbest (  -k)                          -1
input
  charset ( -ic)                        UTF-8
  reader ( -ir)                         tab
  reader_options (-iro)                 
  format ( -if)                         /appdata/dataformat/conllx.xml
  infile (  -i)                         /home/anhv/.underthesea/datasets/VLSP2020-DP-R1/train.txt
  iterations ( -it)                     1
lib
  external ( -lx)                       
  save_instance_files ( -li)            false
  options ( -lo)                        
  verbosity ( -lv)                      silent
multiplanar
  planar_root_handling (-prh)           normal
nivre
  enforce_tree ( -nt)                   true
  allow_reduce ( -ne)                   false
  allow_root ( -nr)                     true
output
  charset ( -oc)                        UTF-8
  outfile (  -o)                        
  format ( -of)                         
  writer_options (-owo)                 
  writer ( -ow)                         tab
planar
  no_covered_roots (-pcov)               false
  acyclicity (-pacy)                     true
  connectedness (-pcon)                  none
pproj
  marking_strategy ( -pp)               none
  lifting_order (-plo)                  shortest
  covered_root (-pcr)                   none
singlemalt
  mode ( -sm)                           parse
  diagnostics ( -di)                    false
  use_partial_tree ( -up)               false
  propagation ( -fp)                    
  parsing_algorithm (  -a)              nivreeager
  guide_model ( -gm)                    single
  null_value ( -nv)                     one
  diafile (-dif)                        stdout 
```

**Biaffine Attenion for Neural Dependency Parsing**

Our attempt is running an experiment using deep biaffine attention for neural dependency parsing method [[1]](#references), which yield a promise result in VLSP2019-DP dataset [[2]](#references).

The input to the model is a sequence of tokens and their part of speech tags, which is then put through a multilayer bidirectional LSTM network. The output state of the final LSTM layer is then fed through four separate ReLU layers, producing four specialized vector representations: one of the word as a dependent seeking its head; of of the word as a head seeking all its dependents; another for the word as a dependent deciding on its label; and a fourth of the word as head deciding on the labels of its dependents. These vectors are then used in two biaffine classifiers: the first computes a score for each pair of tokens, with the highest score for a given token indicating that token's most probable head; the second computes a score for each label for a given token/head pair, with the highest score representing the most probable label for the arc from the head to the dependent. This is show graphically in Figure 1.

*Figure 1: Biaffine Attention for Neural Dependency Parsing*

![](img/biaffine_attention_dependency_parsing.png)

Model Parameters

```
BiaffineDependencyModel(
  (word_embed): Embedding(5407, 100)
  (feat_embed): CharLSTM(169, 50, n_out=100, pad_index=0)
  (embed_dropout): IndependentDropout(p=0.33)
  (lstm): BiLSTM(200, 400, num_layers=3, dropout=0.33)
  (lstm_dropout): SharedDropout(p=0.33, batch_first=True)
  (mlp_arc_d): MLP(n_in=800, n_out=500, dropout=0.33)
  (mlp_arc_h): MLP(n_in=800, n_out=500, dropout=0.33)
  (mlp_rel_d): MLP(n_in=800, n_out=100, dropout=0.33)
  (mlp_rel_h): MLP(n_in=800, n_out=100, dropout=0.33)
  (arc_attn): Biaffine(n_in=500, n_out=1, bias_x=True)
  (rel_attn): Biaffine(n_in=100, n_out=85, bias_x=True, bias_y=True)
  (criterion): CrossEntropyLoss()
)
```

 
We use codebase from [supar code](https://github.com/yzhangcs/parser) work on Vietnamese Dependency Parsing task.

### Dataset

**VLSP2020 Dependency Parsing Dataset**

We show test results on the [VLSP 2020 Dependency Parsing dataset](https://vlsp.org.vn/vlsp2020/eval/udp), training data 
consists 10,000 dependency-annotated sentences. We concat two file `DP-Package2.18.11.2020.txt` and `VTB_2996.txt` as 
train data, and get `VTB_400.txt` file as test data. 

*Figure 2: Example of an annotated sentence in VLSP 2020 Dependency Parsing dataset*

![](img/dp_example_1.png)

### (current) Results

Detail score after using MaltParser, we consider this result as baseline of our experiments  

*Table 2: detail score using MaltParser*
 
```
Metric     | Precision |    Recall |  F1 Score | AligndAcc
-----------+-----------+-----------+-----------+-----------
Tokens     |    100.00 |    100.00 |    100.00 |
Sentences  |    100.00 |    100.00 |    100.00 |
Words      |    100.00 |    100.00 |    100.00 |
UPOS       |    100.00 |    100.00 |    100.00 |    100.00
XPOS       |    100.00 |    100.00 |    100.00 |    100.00
UFeats     |    100.00 |    100.00 |    100.00 |    100.00
AllTags    |    100.00 |    100.00 |    100.00 |    100.00
Lemmas     |    100.00 |    100.00 |    100.00 |    100.00
UAS        |     75.41 |     75.41 |     75.41 |     75.41
LAS        |     66.11 |     66.11 |     66.11 |     66.11
CLAS       |     62.70 |     62.17 |     62.43 |     62.17
MLAS       |     60.74 |     60.23 |     60.48 |     60.23
BLEX       |     62.70 |     62.17 |     62.43 |     62.17 
```

*To reproduce this result, you can run* 

```
export MALT_PARSER=/home/anhv/Downloads/maltparser-1.9.2
python malt_train.py 
```

UAS and LAS after training 240 epochs

```
2020-11-29 23:05:58,924 Epoch 240 saved
2020-11-29 23:05:58,924 dev:   - UCM: 30.67% LCM:  6.98% UAS: 87.28% LAS: 72.63%
2020-11-29 23:05:58,924 test:  - UCM: 30.67% LCM:  6.98% UAS: 87.28% LAS: 72.63%
2020-11-29 23:05:58,924 0:33:46.407770s elapsed, 0:00:05.960023s/epoch
```

*To reproduce this result, you can run* 

```
python nn_train.py 
```

### References

[1] Dozat, T., & Manning, C. D. (2017). Deep biaffine attention for neural dependency parsing. ArXiv:1611.01734 [Cs]. http://arxiv.org/abs/1611.01734

[2] Nguyen et al. (2019). NLP@UIT at VLSP 2019: A Simple Ensemble Model for Vietnamese Dependency Parsing. https://vlsp.org.vn/sites/default/files/2019-10/VLSP2019-DP-NguyenDucVu.pdf