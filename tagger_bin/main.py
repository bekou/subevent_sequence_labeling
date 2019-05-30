import torch

from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn

import time
import reader
import nn_model
import evaluators
from training import trainingUtils,computeMaxScores
import utils
import gc
import os
from sklearn.externals import joblib
import sys
from build_data import build_data


torch.manual_seed(1)
torch.set_default_tensor_type('torch.cuda.FloatTensor')


config=build_data(sys.argv[1])
utils.printParameters(config)

device='cuda:0'
model = nn_model.LSTMTagger(config,device).to(device)
loss_function = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
training_util=trainingUtils(config,device,"Train")
dev_util=trainingUtils(config,device,"Dev")
test_util=trainingUtils(config,device,"Test")
maxScores=computeMaxScores(config,training_util,dev_util,test_util)
for epoch in range(config.nepochs):  
        epoch_loss=0
        training_util.initializeEvaluator(epoch)
        model.train()
        for i_batch, sample_batched in enumerate(config.train_loader):
            

            model.zero_grad()
            tag_scores,bio_targets,ec_targets,event_duration_targets,independent_event_targets,ec_independent_targets,ec_independent_event_ids = model(sample_batched['match'])
          
            training_util.addTargets(tag_scores,bio_targets,ec_targets,independent_event_targets)        
            
            loss = loss_function(tag_scores,training_util.getTargets())

            training_util.predict(ec_independent_targets,ec_independent_event_ids,bio_targets)

            epoch_loss += loss

            loss.backward()
            optimizer.step()
        
        print("Loss {}".format(epoch_loss))
        training_util.printInfo()
        
        # See what the scores are after training
        with torch.no_grad():
        
            dev_util.initializeEvaluator(epoch)
            model.eval()
            

            for i_batch, sample_batched in enumerate(config.dev_loader):
               
                tag_scores,bio_targets,ec_targets,event_duration_targets,independent_event_targets,ec_independent_targets,ec_independent_event_ids = model(sample_batched['match'])

                dev_util.addTargets(tag_scores,bio_targets,ec_targets,independent_event_targets)   

                dev_util.predict(ec_independent_targets,ec_independent_event_ids,bio_targets)

        

        dev_util.printInfo()


        # See what the scores are after training
        with torch.no_grad():
            test_util.initializeEvaluator(epoch)
            model.eval()
            
                
            for i_batch, sample_batched in enumerate(config.test_loader):
               

                tag_scores,bio_targets,ec_targets,event_duration_targets,independent_event_targets,ec_independent_targets,ec_independent_event_ids = model(sample_batched['match'])
                test_util.addTargets(tag_scores,bio_targets,ec_targets,independent_event_targets)   

                test_util.predict(ec_independent_targets,ec_independent_event_ids,bio_targets)

        test_util.printInfo(dev_util.relaxedEventEvaluator)
 
        maxScores.compute(epoch)
        
if config.ner_classes != "EVENT_independent":       
    line=str([['Best test ', maxScores.best_score_test],['Test relaxed best dev ', maxScores.test_relaxed_best_dev],['Test token best dev ', maxScores.test_token_best_dev],['Config', config.config_fname]])                

else:       
    line=str([['Best test micro', maxScores.best_micro_test],['Best test macro', maxScores.best_macro_test],['Best overall micro dev', maxScores.overall_micro_dev],['Best overall macro dev', maxScores.overall_macro_dev],['Config', config.config_fname]])                

utils.appendToFile("logs/results_"+sys.argv[2],''.join(line))    
   