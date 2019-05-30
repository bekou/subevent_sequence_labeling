# -*- coding: utf-8 -*-
import random
import gensim
import numpy as np
import ast
import re
import torch
from prettytable import PrettyTable

"""Generic set of classes and methods"""

def thresholdedEventPredictions(threshold,tag_scores):
    predicted_scores, predicted_labels = torch.max(tag_scores, dim=1)
    mlabels=[]
    for labelIdx in range(len(predicted_labels)):
            label=predicted_labels[labelIdx].item()
            score=predicted_scores[labelIdx].item()
            #print (label)
            #print (score)

            if label==1 and score>threshold:
                 mlabels.append(1)
            else:
                 mlabels.append(0)
    mlabels=torch.tensor(mlabels, dtype=torch.long)
    
    return mlabels
    
    
def appendToFile(filename,line):
    with open(filename, 'a') as the_file:
        the_file.write(line+'\n')
        the_file.close()

def listOfIdsToTags(lst_ids,tags):
    lstTags= []
    for nerId in lst_ids:
        lstTags.append(tags[nerId])
    return lstTags
def transformToECtags(tags):
    ECtags=[]
    for tag in tags:
        if tag.startswith("B-") or tag.startswith("I-"):
                ECtags.append(tag[2:])
        else:
                ECtags.append(tag)
    return ECtags

def getDictionaryKeyByIdx(mydict,idx):
    for key, value in mydict.items():  # for name, age in dictionary.iteritems():  (for Python 2.x)
        if value == idx:
            return key
    #return mydict.keys()[mydict.values().index(idx)]
def getSegmentationDict(lst):
    return {k: v for v, k in enumerate(lst)}

def getSortedTagsFromBIO(tag_to_ix):
    BIOtags = []
    ECtags = []
    for tag,value in tag_to_ix.items():
            BIOtags.append(tag)
            if tag.startswith("B-") or tag.startswith("I-"):
                ECtags.append(tag[2:])
            else:
                ECtags.append(tag)

    BIOtags = list(set(BIOtags))
    BIOtags.sort()
    ECtags = list(set(ECtags))
    ECtags.sort()
    return BIOtags, ECtags


def prepare_sequence(seq, to_ix):
    #idxs = [to_ix[w] for w in seq]
    idxs=[]
    #for w in seq:
    #    try:
    #        idxs.append(to_ix[w])
    #    except:
    #        idxs.append(to_ix['UNK'])
    #print (to_ix.keys())
    #embeddingsList=[]
    #for key, value in to_ix.items():
    #    #temp = [key, value]
    #    embeddingsList.append(value)

    for w in seq:
        idxs.append(getEmbeddingId(w, to_ix))

    return torch.tensor(idxs, dtype=torch.long)


def getEmbeddingId(word, embeddingsList):
    # modified method from http://cistern.cis.lmu.de/globalNormalization/globalNormalization_all.zip
    if word != "<empty>":
        if not word in embeddingsList:
            if re.search(r'^\d+$', word):
                word = "0"
            if word.islower():
                word = word.title()
            else:
                word = word.lower()
        if not word in embeddingsList:
            word = "<unk>"
        curIndex = embeddingsList[word]
        return curIndex

def strToLst(string):
    return ast.literal_eval(string)


def lstToString(lst):
    return ' '.join(lst)

###run one time to obtain the characters
def getCharsFromDocuments(documents):
    chars = []
    for doc in documents:
        for tokens in doc.tokens:
            for char in tokens:
                # print (token)
                chars.append(char)
    chars = list(set(chars))
    chars.sort()
    return chars


###run one time to obtain the ner labels
def getEntities(documents):
    BIOtags = []
    ECtags = []
    for doc in documents:
        for tag in doc.BIOs:
            BIOtags.append(tag)
            if tag.startswith("B-") or tag.startswith("I-"):
                ECtags.append(tag[2:])
            else:
                ECtags.append(tag)

    BIOtags = list(set(BIOtags))
    BIOtags.sort()
    ECtags = list(set(ECtags))
    ECtags.sort()
    return BIOtags, ECtags


def getECfromBIO(BIO_tag):
    if BIO_tag.startswith("B-") or BIO_tag.startswith("I-"):
        return (BIO_tag[2:])
    else:
        return (BIO_tag)




def tokenToCharIds(token, characters):
    charIds = []
    for char in token:
        charIds.append(characters.index(char))
    return charIds


def labelsListToIds(listofLabels, setofLabels):
    labelIds = []
    for label in listofLabels:
        labelIds.append(setofLabels.index(label))

    return labelIds





def getLabelId(label, setofLabels):
    return setofLabels.index(label)

def strToBool(str):
    if str.lower() in ['true', '1']:
        return True
    return False



def getEmbeddingId(word, embeddingsList):
    # modified method from http://cistern.cis.lmu.de/globalNormalization/globalNormalization_all.zip
    if word != "<empty>":
        if not word in embeddingsList:
            if re.search(r'^\d+$', word):
                word = "0"
            if word.islower():
                word = word.title()
            else:
                word = word.lower()
        if not word in embeddingsList:
            word = "<unk>"
        curIndex = embeddingsList[word]
        return curIndex


def readWordvectorsNumpy(wordvectorfile, isBinary=False):
    wordvectors = []
    words = []
    model = gensim.models.KeyedVectors.load_word2vec_format(wordvectorfile, binary=isBinary)

    vectorsize = model.vector_size

    for key in list(model.vocab.keys()):
        wordvectors.append(model.wv[key])
        words.append(key)

    zeroVec = [0 for i in range(vectorsize)]
    random.seed(123456)
    randomVec = [random.uniform(-np.sqrt(1. / len(wordvectors)), np.sqrt(1. / len(wordvectors))) for i in
                 range(vectorsize)]
    wordvectors.insert(0, randomVec)
    words.insert(0, "<unk>")
    #wordvectors.insert(0, zeroVec)
    #words.insert(0, "<empty>")

    wordvectorsNumpy = np.array(wordvectors)
    vocab_to_idx = {k: v for v, k in enumerate(words)}

    return wordvectorsNumpy, vectorsize, vocab_to_idx


def readIndices(wordvectorfile, isBinary=False):
    # modified method from http://cistern.cis.lmu.de/globalNormalization/globalNormalization_all.zip
    indices = {}
    curIndex = 0
    indices["<empty>"] = curIndex
    curIndex += 1
    indices["<unk>"] = curIndex
    curIndex += 1

    model = gensim.models.KeyedVectors.load_word2vec_format(wordvectorfile, binary=isBinary,unicode_errors='ignore')

    count = 0
    # c=0
    for key in list(model.vocab.keys()):
        indices[key] = curIndex
        curIndex += 1

    return indices



def printParameters(config):

    t = PrettyTable(['Params', 'Value'])

    #dataset
    t.add_row(['Config', config.config_fname])
    t.add_row(['Pretrained Embeddings', config.pretrained_embeddings])
    t.add_row(['Embeddings', config.filename_embeddings])
    t.add_row(['Embeddings size ', config.embeddings_size])
    t.add_row(['Train', config.filename_train])
    t.add_row(['Dev', config.filename_dev])
    t.add_row(['Test', config.filename_test])

 #training
    t.add_row(['Epochs ', config.nepochs])
    t.add_row(['Optimizer ', config.optimizer])
    #t.add_row(['Activation ', config.activation])
    t.add_row(['Learning rate ', config.learning_rate])

    #t.add_row(['Patience ', config.nepoch_no_imprv])
    t.add_row(['Use dropout', config.use_dropout])
    t.add_row(['Ner loss ', config.ner_loss])
    t.add_row(['Ner classes ', config.ner_classes])
    t.add_row(['BIO LSTM ', config.use_BIO_LSTM])
    t.add_row(['Bin features ', config.bin_features])
    t.add_row(['Independent event threshold ', config.threshold])
                                                          
                                                          


    # hyperparameters

    t.add_row(['Bidirectional BIO LSTM ', config.bidirectionalBIO_LSTM])
    t.add_row(['Batch Norm ', config.batch_norm])
    t.add_row(['N filters ', config.n_filters])
    t.add_row(['Filter sizes ', config.filter_sizes])
    t.add_row(['Dropout cnn ', config.dropout_cnn])
    t.add_row(['Cnn pool ', config.cnn_pool])
    t.add_row(['Bin representation ', config.bin_representation])
    t.add_row(['Dropout lstm1 output ', config.dropout_lstm1_output])


    t.add_row(['Dropout embedding ', config.dropout_embedding])
    #t.add_row(['Dropout lstm ', config.dropout_lstm])
    t.add_row(['Dropout lstm2 output ', config.dropout_lstm2_output])
    t.add_row(['Dropout fcl ner ', config.dropout_fcl_ner])
    t.add_row(['Dropout fcl rel ', config.dropout_fcl_rel])
    #t.add_row(['Hidden lstm size ', config.hidden_size_lstm])
    t.add_row(['LSTM layers ', config.num_lstm_layers])
    t.add_row(['Hidden nn size ', config.hidden_dim])


    #evaluation
    #t.add_row(['Evaluation method ', config.evaluation_method])


    print(t)
