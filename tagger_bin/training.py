import evaluators
import time
import utils
import torch

class trainingUtils:
    """Set of classes and methods for training the model and computing the ner and head selection loss"""


    def __init__(self,config,device,set):
        """"Initialize data"""
        self.config=config
        self.targets=[]
        self.tag_scores=[]
        self.device=device
        self.tokenEvaluator=""
        self.relaxedEvaluator=""
        self.relaxedEventEvaluator=""
        self.start_time=0
        self.set=set
        self.overallMacro=-1
        self.overallMicro=-1
   
    def initializeEvaluator(self,epoch):
    
        epoch_loss = 0

        if self.config.ner_classes=="BIO":

            self.tokenEvaluator = evaluators.nerTokenEvaluator(list(self.config.tag_to_ix.keys()))

            self.relaxedEvaluator = evaluators.relaxedChunkEvaluator(self.config.tag_to_ix)

        elif self.config.ner_classes=="EC_duration":

            self.tokenEvaluator = evaluators.nerTokenEvaluator(list(self.config.ec_to_ix.keys()))

            self.relaxedEvaluator = evaluators.relaxedChunkEvaluator(self.config.tag_to_ix)

        elif self.config.ner_classes== "EVENT_independent":

            self.relaxedEventEvaluator = evaluators.relaxedEventEvaluator(list(self.config.event_to_ix.keys()))
        if self.set=="Train":     
            print("Epoch {}".format(epoch))
        self.start_time = time.time()
        
    def addTargets(self,tag_scores, bio_targets,ec_targets,independent_event_targets):
            self.tag_scores=tag_scores
            
            if self.config.ner_classes == "BIO":

                self.targets = bio_targets

            elif self.config.ner_classes == "EC_duration":

                self.targets = ec_targets

            elif self.config.ner_classes == "EVENT_independent":

                self.targets=independent_event_targets


            self.targets=torch.tensor(self.targets, dtype=torch.long).to(self.device)
            
    def getTargets(self):
        return self.targets
        
    def predict(self,ec_independent_targets,ec_independent_event_ids,bio_targets):
    
        if self.config.ner_classes == "EVENT_independent":
                if self.config.threshold != 0:
                    predicted_labels = utils.thresholdedEventPredictions(self.config.threshold, self.tag_scores)
                else:
                    predicted_scores, predicted_labels = torch.max(self.tag_scores, dim=1)
                self.relaxedEventEvaluator.add(predicted_labels, self.targets, ec_independent_targets,
                                               ec_independent_event_ids)
        elif self.config.ner_classes == "BIO" or self.config.ner_classes=="EC_duration":
                predicted_scores, predicted_labels = torch.max(self.tag_scores, dim=1)
                self.tokenEvaluator.add(predicted_labels, self.targets)

                self.relaxedEvaluator.add(predicted_labels, self.targets,bio_targets)
                
          
    def printInfo(self,relaxedEventEvaluatorDev=""):
        print('-------'+self.set+'-------')
        if self.config.ner_classes == "BIO" or self.config.ner_classes == "EC_duration":
        
            tokenF1 = self.tokenEvaluator.getOverallF1()
            print("F1 score {}".format(tokenF1))
            self.tokenEvaluator.printInfo()

            self.relaxedEvaluator.printInfoMicro()
            
            relaxedF1, relaxedF1_nother=self.relaxedEvaluator.getOverallF1()   
            
        elif self.config.ner_classes == "EVENT_independent":
            self.relaxedEventEvaluator.printInfo()
            
            if self.set=="Test":
            
                test_tps,test_fns,test_fps=self.relaxedEventEvaluator.getContincencyTable()
                dev_tps,dev_fns,dev_fps=relaxedEventEvaluatorDev.getContincencyTable()
                
                self.overallMacro=(self.relaxedEventEvaluator.getMacroF1()+relaxedEventEvaluatorDev.getMacroF1())/2
                self.overallMicro=evaluators.getF1(dev_tps+test_tps,dev_fps+test_fps,dev_fns+test_fns)
        
        elapsed_time = time.time() - self.start_time
        print("Elapsed {} time in {} seconds and {} minutes".format(self.set,elapsed_time,elapsed_time/60))
        print()
        

class computeMaxScores:

        def __init__(self,config,train_util,dev_util,test_util):
            """"Initialize data"""
            self.config=config
            self.best_score_dev=0
            self.best_score_test=0
            self.best_score_dev_token=0
            self.epoch_dev_max=0
            self.epoch_test_max=0
            self.epoch_loss=0
            self.best_test_token=0
            self.best_dev_token=0
            self.test_relaxed_best_dev=0
            self.test_token_best_dev=0
            self.train_score_token=0
            self.train_score_relaxed=0
            #independent best
            self.best_micro_dev=0
            self.best_macro_dev=0
            self.best_micro_test=0
            self.best_macro_test=0
            self.overall_micro_dev=0
            self.overall_macro_dev=0
            self.train_util=train_util
            self.dev_util=dev_util
            self.test_util=test_util

        def compute(self,epoch):
            if self.config.ner_classes != "EVENT_independent":
                if self.dev_util.tokenEvaluator.getOverallF1() > self.best_score_dev_token:
                    self.test_token_best_dev = self.test_util.tokenEvaluator.getOverallF1()
                    self.best_score_dev_token = self.dev_util.tokenEvaluator.getOverallF1()
                    print("- Best dev score token {} so far in {} epoch '\n'".format(self.dev_util.tokenEvaluator.getOverallF1(), epoch))

                if self.test_util.tokenEvaluator.getOverallF1() > self.best_test_token:
                    self.best_test_token= self.test_util.tokenEvaluator.getOverallF1()
                    print("- Best test score token {} so far in {} epoch '\n'".format(self.test_util.tokenEvaluator.getOverallF1(), epoch))
                if self.dev_util.relaxedEvaluator.getOverallF1()[0]>self.best_score_dev:
                    self.best_score_dev = self.dev_util.relaxedEvaluator.getOverallF1()[0]
                    self.epoch_dev_max=epoch
                    self.test_relaxed_best_dev=self.test_util.relaxedEvaluator.getOverallF1()[0]
                    self.best_dev_token=self.dev_util.tokenEvaluator.getOverallF1()
                    print ("- Best dev score relaxed {} so far in {} epoch '\n'".format(self.dev_util.relaxedEvaluator.getOverallF1()[0],epoch))

                if self.test_util.relaxedEvaluator.getOverallF1()[0]>self.best_score_test:
                    self.best_score_test = self.test_util.relaxedEvaluator.getOverallF1()[0]
                    self.epoch_test_max=epoch
                    print ("- Best test score relaxed {} so far in {} epoch '\n'".format(self.test_util.relaxedEvaluator.getOverallF1()[0],epoch))
            else:
                if self.dev_util.relaxedEventEvaluator.getMicroF1()> self.best_micro_dev:                        
                    self.best_micro_dev=self.dev_util.relaxedEventEvaluator.getMicroF1()                            
                    self.overall_micro_dev=self.test_util.overallMicro   
                    self.epoch_dev_max=epoch                            
                if self.dev_util.relaxedEventEvaluator.getMacroF1()> self.best_macro_dev:                        
                    self.best_macro_dev=self.dev_util.relaxedEventEvaluator.getMacroF1()                            
                    self.overall_macro_dev=self.test_util.overallMacro                            
                if self.test_util.relaxedEventEvaluator.getMicroF1()> self.best_micro_test:                        
                    self.best_micro_test=self.test_util.overallMicro
                    self.epoch_test_max=epoch                            
                if self.test_util.relaxedEventEvaluator.getMicroF1()> self.best_macro_test:                        
                    self.best_macro_test=self.test_util.overallMacro           