from torch.utils.data import Dataset, DataLoader
import pandas as pd
import utils
import numpy as np
import torch


class BinDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, tsv_file,isTrain,pretrained_embeddings=False, word_to_ix={}, tag_to_ix={},event_to_ix={},pad_length=0):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        # self.landmarks_frame = pd.read_csv(csv_file)
        col_vector = ['time', 'tweet']
        self.bin_df = pd.read_csv(tsv_file, names=col_vector, encoding="utf-8",
                                  engine='python', sep="\t")
        self.matches = []
        self.word_to_ix = word_to_ix
        self.tag_to_ix = tag_to_ix
        self.event_to_ix = event_to_ix

        self.pad_length=pad_length

        self.BIOset, self.ECset = utils.getSortedTagsFromBIO(self.tag_to_ix)
        self.tag_to_ix = utils.getSegmentationDict(self.BIOset)
        self.ec_to_ix = utils.getSegmentationDict(self.ECset)



        if isTrain == True:
            self.createDics(self.bin_df,pretrained_embeddings)
        #print(word_to_ix)
        #print(self.tag_to_ix)
        #print(self.ec_to_ix)
        #print (self.event_to_ix)

        self.preprocess(self.bin_df)

    def getDictionaries(self):
        return self.word_to_ix, self.tag_to_ix,self.event_to_ix,self.ec_to_ix

    def createDics(self, bin_dataframe,pretrained_embeddings):

        bin_np = bin_dataframe.as_matrix()

        if pretrained_embeddings==False: # maybe not need this!

            self.word_to_ix["<unk>"] = len(self.word_to_ix)

        #initialize the event dictionary
        self.event_to_ix["non-event"]= len(self.event_to_ix)
        self.event_to_ix["event"]= len(self.event_to_ix)

        # initialize the tags dictionary
        self.tag_to_ix["B-Other"] = len(self.tag_to_ix)
        self.tag_to_ix["I-Other"] = len(self.tag_to_ix)

        for line in bin_np:
            if line[1] != None:
                if pretrained_embeddings==True:
                    continue
                else:

                    for word in utils.strToLst(line[1]):
                        if word not in self.word_to_ix:
                            self.word_to_ix[word] = len(self.word_to_ix)

            else:
                tag = utils.strToLst(line[0])['corrected_tags']
                if tag not in self.tag_to_ix:
                    self.tag_to_ix[tag] = len(self.tag_to_ix)

        self.BIOset, self.ECset = utils.getSortedTagsFromBIO(self.tag_to_ix)
        self.tag_to_ix = utils.getSegmentationDict(self.BIOset)
        self.ec_to_ix = utils.getSegmentationDict(self.ECset)



    def preprocess(self, bin_dataframe):
        bin_np = bin_dataframe.as_matrix()
        docNr = -1

        bin_tweets = []

        bin_tweet_lengths=[]
        bin_tweets_text=[]

        previous_match = ""

        match = []
        for i in range(bin_np.shape[0]):

            if bin_np[i][1] == None or i == bin_np.shape[0] - 1:  # append all docs including the last one
                if (i == bin_np.shape[0] - 1):  # append last line
                    tweet_text = utils.lstToString(utils.strToLst(bin_np[i][1])).split()
                    tweet, tweet_length = utils.prepare_sequence(
                        tweet_text, self.word_to_ix,
                        pad_length=self.pad_length)
                    bin_tweets.append(tweet)
                    bin_tweet_lengths.append(tweet_length)
                    bin_tweets_text.append(tweet_text)


                if (docNr != -1):
                    #bin_tweets = np.asarray(bin_tweets)


                    try:
                        tag_id = self.tag_to_ix[target]

                        if target.startswith("B-") or target.startswith("I-"):
                            ec_id=self.ec_to_ix[target[2:]]
                        else:
                            ec_id=self.ec_to_ix[target]
                    except:
                        print(target)
                        if target.startswith("B-"):
                            tag_id = self.tag_to_ix["B-Other"]

                        elif target.startswith("I-"):
                            tag_id = self.tag_to_ix["I-Other"]

                        ec_id = self.ec_to_ix["Other"]

                    if target=="O":
                        event_duration_idx = self.event_to_ix["non-event"]
                    else:
                        event_duration_idx = self.event_to_ix["event"]                                  
                    if event_id==-1:
                        independent_event_idx = self.event_to_ix["non-event"]
                    else:
                        independent_event_idx = self.event_to_ix["event"]

                    #print (len(bin_tweets))
                    #print (torch.stack(bin_tweets))
                    match.append([torch.stack(bin_tweets), tag_id,ec_id,event_duration_idx,independent_event_idx,event_type,event_id,bin_tweet_lengths])

                    #print (utils.getDictionaryKeyByIdx(self.tag_to_ix,tag_id),utils.getDictionaryKeyByIdx(self.ec_to_ix,ec_id),utils.getDictionaryKeyByIdx(self.event_to_ix,event_id))

                    # match=np.append(match,bin_tokens)
                    # match['match_bins'].append(bin)

                docNr += 1
                if i != bin_np.shape[0] - 1:
                    infoDict = utils.strToLst(bin_np[i][0])

                    if previous_match != infoDict['doc']:
                        # print (infoDict['doc'])

                        # match = {'match_bins': np.empty((0)),"match_name": infoDict['doc']}
                        previous_match = infoDict['doc']
                        match = []

                        self.matches.append(match)

                    bin_tweets = []
                    bin_tweet_lengths=[]
                    bin_tweets_text=[]
                    target = infoDict['corrected_tags']
                    event_type = infoDict['event_type']
                    event_id = infoDict['event_id']
                    match_name= infoDict['doc']                           


                    # {'bin': infoDict['bin'],'targets': infoDict['corrected_tags'],'tweets':[],'timestamps':[],'tokens':""}
            else:

                # bin['tweets'].append(strToLst(bin_np[i][1]))
                # bin_tokens+=" "+lstToString(strToLst(bin_np[i][1]))
                # bin['timestamps'].append(int(bin_np[i][0]))
                # print ((lstToString(strToLst(bin_np[i][1])).split()))
                #print (bin_tokens)
                tweet_text=utils.lstToString(utils.strToLst(bin_np[i][1])).split()
                tweet,tweet_length=utils.prepare_sequence(tweet_text, self.word_to_ix,
                                       pad_length=self.pad_length)
                bin_tweets.append(tweet)
                bin_tweet_lengths.append(tweet_length)
                bin_tweets_text.append(tweet_text)

    def __len__(self):
        return len(self.matches)

    def __getitem__(self, idx):
        # img_name = os.path.join(self.root_dir,
        #                       self.landmarks_frame.iloc[idx, 0])
        # bin_id = self.bin_tokens[idx]
        match = self.matches[idx]
        sample = {'match': match}

        return sample