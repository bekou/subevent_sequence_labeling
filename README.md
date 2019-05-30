# Sub-event detection from twitter streams as a sequence labeling problem


Implementation of our paper
[Sub-event detection from twitter streams as a sequence labeling problem](https://arxiv.org/pdf/1903.05396.pdf).

# Requirements
* Ubuntu 16.04
* Anaconda 5.2.0
* Numpy 1.16.1
* Gensim 3.6.0
* Pytorch 0.4.1
* PrettyTable 0.7.2

## Task
The goal is, given a main event (i.e., soccer match), to identify its core sub-events (e.g., goals, kick-off, yellow cards) from Twitter streams.

## Configuration
The model has several parameters that could be specified in the configuration files (see [config](https://github.com/bekou/subevent_sequence_labeling/tree/master/tagger_bin/configs)).

## Run the model
There are two directories (one for the word-level model and one the tweet-level model)
```
./run_<name_of_the_model>.sh
```

## Notes

Please cite our work when using this software.

Giannis Bekoulis, Johannes Deleu, Thomas Demeester, Chris Develder. Sub-event detection from twitter streams as a sequence labeling problem, In the Proceedings of the 2019 Annual Conference of the North American Chapter of the Association for Computational Linguistics, 2019


