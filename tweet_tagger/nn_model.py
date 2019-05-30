import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
import sys
import gc
import traceback
import logging
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable



class LSTMTagger(nn.Module):

    def __init__(self,config,device):

        super(LSTMTagger, self).__init__()

        self.device=device

        self.config=config

        if self.config.ner_classes == "EVENT_independent":
           target_size = len(self.config.event_to_ix)
        elif self.config.ner_classes=="BIO":
            target_size= len(self.config.tag_to_ix)
        elif self.config.ner_classes == "EC_duration":
            target_size = len(self.config.ec_to_ix)

        self.word_embeddings = nn.Embedding(len(self.config.word_to_ix), self.config.embeddings_size)#.to('cuda')

        if self.config.pretrained_embeddings==True:

            self.word_embeddings.weight= nn.Parameter(torch.FloatTensor(self.config.wordvectors)) 
        self.convs = nn.ModuleList(
            [nn.Conv2d(in_channels=1, out_channels=self.config.n_filters, kernel_size=(fs, self.config.embeddings_size)) for fs in self.config.filter_sizes])

        self.cnn_dropout = nn.Dropout(self.config.dropout_cnn)

        self.lstm = nn.LSTM(self.config.embeddings_size, self.config.embeddings_size)#.to('cuda')

        if self.config.bin_features != "cnn":
            self.hidden_size = self.config.embeddings_size
        else:
            self.hidden_size = len(self.config.filter_sizes) * self.config.n_filters

        self.dense1_bn = nn.BatchNorm1d(self.hidden_size)
        self.lstm2 = nn.LSTM(self.hidden_size * 1, self.hidden_size)  # .to('cuda')
        self.hidden2tag = nn.Linear(self.hidden_size*1, target_size)#.to('cuda')
        self.hidden2score = nn.Linear(self.hidden_size, 1)  # .to('cuda')

        self.relu=nn.ReLU()
        self.tanh=nn.Tanh()

        self.twolayerhidden2tag = torch.nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, target_size),
        )

        self.dropoutEmbeddings = nn.Dropout(self.config.dropout_embedding)
        self.dropoutLSTM2out = nn.Dropout(self.config.dropout_lstm2_output)

        self.softmax=nn.Softmax(dim=0)
        self.attention_weights2 = nn.Parameter(torch.FloatTensor(self.config.embeddings_size))
        self.softmax3 = nn.Softmax(dim=-1)

        nn.init.uniform(self.attention_weights2.data, -0.005, 0.005)                                                            
    def sort_batch(self,X,  lengths):
        lengths, indx = lengths.sort(dim=0, descending=True)

       
        X = X[indx]
       
        return X,  lengths,#X.transpose(0, 1), y, lengths  # transpose (batch x seq) to (seq x batch)


    def forward(self, bins):
        lst_encodings = []
        bio_targets = []

        ec_targets = []
        event_duration_targets = []
        independent_events_targets = []
        ec_independent_targets = []
        ec_independent_event_ids = []
        
        for bin in bins:
            
            bin_tweets = bin[0]
            
            bin_tweets=bin_tweets.to('cuda')
            
            bin_tweets=bin_tweets.squeeze(0)
           
            bio_targets.extend(bin[1].data.tolist())
            ec_targets.extend(bin[2])
            event_duration_targets.extend(bin[3]) #(0,1) with duration  0000111111111111110000
            independent_events_targets.extend(bin[4]) #(0,1) without duration  0000111110000111110000
               
            ec_independent_targets.extend(bin[5]) #(0,1,2,3,4) without duration  0000111110000222220000
            ec_independent_event_ids.extend(bin[6]) #(-1,0,1,2,3,4) without duration  -1-1-1-100000-1-1-1-111111-1-1-1-1

            tweets_length= torch.stack(bin[7]).to('cuda')

            tweets_length=tweets_length.squeeze(1)
           

            bin_tweets, tweets_length = self.sort_batch(bin_tweets, tweets_length)

            embeds = self.word_embeddings(bin_tweets).permute(1,0,2)
           

            if self.config.use_dropout == True:
                embeds = self.dropoutEmbeddings(embeds)
            
            embeds = pack_padded_sequence(embeds, tweets_length)  # unpad

            if self.config.tweet_representation == "embeddings":
                tweet_representation = embeds
                tweet_representation, _ = pad_packed_sequence(tweet_representation)
                
            elif self.config.tweet_representation == "lstm":
                lstm_output, _ = self.lstm(embeds)
                tweet_representation, _ = pad_packed_sequence(lstm_output)

            if self.config.bin_representation == "avg":
                
                tl_expanded=(tweets_length).unsqueeze(1)

                avg_pool_byhand = torch.sum(tweet_representation, dim=0) / tl_expanded.to(
                    dtype=torch.float)

                bin_representation = avg_pool_byhand

            elif self.config.bin_representation == "max":

                max_pool_byhand = torch.cat(
                    [torch.max(i[:l], dim=0)[0].view(1, -1) for i, l in zip(tweet_representation.permute(1, 0, 2), tweets_length)], dim=0)


                bin_representation = max_pool_byhand

            bin_representation=bin_representation.view(bin_representation.size(0),1, bin_representation.size(1))

            if self.config.bin_features == "avg":
                avg_pool = F.adaptive_avg_pool1d(bin_representation.permute(1, 2, 0), 1)#.view(1,1, -1)

                avg_pool.squeeze_(2).unsqueeze_(0) # 1,1,32

                bin_features = avg_pool



            elif self.config.bin_features == "max":

                max_pool = F.adaptive_max_pool1d(bin_representation.permute(1, 2, 0), 1)#.view(1,1, -1)
                max_pool.squeeze_(2).unsqueeze_(0) # 1,1,32

                bin_features = max_pool

            elif self.config.bin_features == "tweet_attention":


                bin_representation.squeeze_(1)


                scores = self.hidden2score(bin_representation)

                normalized_scores = self.softmax(scores)

                a_e = normalized_scores * bin_representation
                attented_representation = torch.sum(a_e, dim=0)


                bin_features = attented_representation#.view(1,1, -1)
                bin_features.unsqueeze_(0).unsqueeze_(0)
                

            elif self.config.bin_features == "tweet_attention2":

                
                bin_representation.squeeze_(1)
                inputs=bin_representation
                
                scores = self.relu(inputs.matmul(self.attention_weights2))
                
                scores = self.softmax3(scores)
                
                weighted = torch.mul(inputs, scores.unsqueeze(-1).expand_as(inputs))

                representations = weighted.sum(0).squeeze()

                bin_features = representations#.view(1, 1, -1)

                bin_features.unsqueeze_(0).unsqueeze_(0)

            elif self.config.bin_features == "cnn":

               
                embedded = bin_representation.squeeze(1).unsqueeze_(0).unsqueeze_(0)
               
                conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]

                if self.config.cnn_pool == "max":

                    pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]

                elif self.config.cnn_pool == "avg":
                    pooled = [F.avg_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]

               
                bin_features = torch.cat(pooled, dim=1)
                if self.config.use_dropout == True:
                    bin_features = self.cnn_dropout(torch.cat(pooled, dim=1))

                bin_features.unsqueeze_(0)
                

            if self.config.non_linearity_bin_features=="relu":

                bin_features=self.relu(bin_features)
            
            elif self.config.non_linearity_bin_features=="tanh":

                bin_features=self.tanh(bin_features)
           
            lst_encodings.append(bin_features)

        lst_encodings = torch.cat(lst_encodings)  # .to(self.device)

        if self.config.batch_norm==True:


            lst_encodings = self.dense1_bn(lst_encodings.view(1,lst_encodings.size(2),lst_encodings.size(0)))

            lst_encodings=lst_encodings.view(lst_encodings.size(2),1,lst_encodings.size(1))


        if self.config.use_BIO_LSTM==True:

            lstm_out2,_ = self.lstm2(
                lst_encodings)


            if self.config.use_dropout == True:

                lstm_out2 = self.dropoutLSTM2out(lstm_out2)

            match_representation=lstm_out2
        else:
            match_representation=lst_encodings
        match_representation = match_representation.squeeze(1)

        tag_space = self.hidden2tag(match_representation)


        return tag_space, bio_targets,ec_targets,event_duration_targets,independent_events_targets,ec_independent_targets,ec_independent_event_ids

