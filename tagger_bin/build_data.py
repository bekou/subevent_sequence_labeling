import os
import utils
import parsers
from sklearn.externals import joblib
import os.path
import reader
from torch.utils.data import Dataset, DataLoader
import os


""""Read the configuration file and set the parameters of the model"""

class build_data():
    def __init__(self,fname):


        config_file=parsers.read_properties(fname)
        #print("\nConfiguration file {} loaded \n".format(fname))
        self.config_fname=fname

        # load data
        self.pretrained_embeddings=utils.strToBool(config_file.getProperty("pretrained_embeddings"))


        self.filename_embeddings = config_file.getProperty("filename_embeddings")

        #print(os.path.basename(self.filename_embeddings))

        name_of_embeddings = ""


        self.embeddings_size=int(config_file.getProperty("embeddings_size"))
        self.word_to_ix={}

        if self.pretrained_embeddings==True:

            name_of_embeddings = "_"+os.path.basename(self.filename_embeddings)

            if os.path.isfile(self.filename_embeddings+".pkl")==False:
                        self.wordvectors,  self.embeddings_size, self.word_to_ix = utils.readWordvectorsNumpy(self.filename_embeddings, isBinary=True if self.filename_embeddings.endswith(".bin") else False)


                        joblib.dump(( self.wordvectors,  self.embeddings_size, self.word_to_ix), self.filename_embeddings+".pkl")

            else:
                    self.wordvectors, self.embeddings_size, self.word_to_ix = joblib.load(self.filename_embeddings + ".pkl")  # loading is faster



        self.filename_train=config_file.getProperty("filename_train")
        self.filename_dev = config_file.getProperty("filename_dev")
        self.filename_test=config_file.getProperty("filename_test")


        if os.path.isfile(self.filename_train +name_of_embeddings+ ".pkl") == False:

            train = reader.BinDataset(self.filename_train,isTrain=True,pretrained_embeddings=self.pretrained_embeddings,word_to_ix=self.word_to_ix)

            joblib.dump(train, self.filename_train +name_of_embeddings+ ".pkl")

        else:
            train = joblib.load(self.filename_train+name_of_embeddings + ".pkl")  # loading is faster

        self.word_to_ix, self.tag_to_ix, self.event_to_ix, self.ec_to_ix = train.getDictionaries()

        
        if os.path.isfile(self.filename_dev +name_of_embeddings+  ".pkl") == False:

            dev = reader.BinDataset(self.filename_dev,isTrain=False, word_to_ix=self.word_to_ix, tag_to_ix=self.tag_to_ix, event_to_ix=self.event_to_ix)

            joblib.dump(dev, self.filename_dev +name_of_embeddings+  ".pkl")

        else:
            dev = joblib.load(self.filename_dev +name_of_embeddings+  ".pkl")  # loading is faster

        

        if os.path.isfile(self.filename_test +name_of_embeddings+  ".pkl") == False:

            test = reader.BinDataset(self.filename_test,isTrain=False, word_to_ix=self.word_to_ix, tag_to_ix=self.tag_to_ix, event_to_ix=self.event_to_ix)

            joblib.dump(test, self.filename_test +name_of_embeddings+  ".pkl")

        else:
            test = joblib.load(self.filename_test +name_of_embeddings+  ".pkl")  # loading is faster
        
        print (train)

        self.train_loader = DataLoader(train, batch_size=1, shuffle=False)
        self.dev_loader = DataLoader(dev, batch_size=1, shuffle=False)
        self.test_loader = DataLoader(test, batch_size=1, shuffle=False)

        print ()
        # training
        self.nepochs = int(config_file.getProperty("nepochs"))
        self.optimizer = config_file.getProperty("optimizer")
        #self.activation =config_file.getProperty("activation")
        self.learning_rate =float(config_file.getProperty("learning_rate"))
                                                                                             
        #self.nepoch_no_imprv = int(config_file.getProperty("nepoch_no_imprv"))
        self.use_dropout = utils.strToBool(config_file.getProperty("use_dropout"))
        self.use_BIO_LSTM = utils.strToBool(config_file.getProperty("use_BIO_LSTM"))
        self.ner_loss = config_file.getProperty("ner_loss")
        self.ner_classes = config_file.getProperty("ner_classes")
        self.bin_features = config_file.getProperty("bin_features").lower()
        try:
            self.threshold = float(config_file.getProperty("threshold"))
        except:
            self.threshold=0

        # hyperparameters
        self.n_filters        = int(config_file.getProperty("n_filters"))
        self.filter_sizes     = utils.strToLst(config_file.getProperty("filter_sizes"))
        self.batch_norm       = utils.strToBool(config_file.getProperty("batch_norm"))
        self.cnn_pool         = config_file.getProperty("cnn_pool").lower()
        self.dropout_cnn      = float(config_file.getProperty("dropout_cnn"))
        self.bin_representation = config_file.getProperty("bin_representation").lower()
        self.dropout_lstm1_output = float(config_file.getProperty("dropout_lstm1_output"))

        try:
            self.bidirectionalBIO_LSTM =  utils.strToBool(config_file.getProperty("bidirectionalBIO_LSTM"))
        except:
            self.bidirectionalBIO_LSTM=False

        self.dropout_embedding = float(config_file.getProperty("dropout_embedding"))
        #self.dropout_lstm = float(config_file.getProperty("dropout_lstm"))
        self.dropout_lstm2_output = float(config_file.getProperty("dropout_lstm2_output"))
        self.dropout_fcl_ner = float(config_file.getProperty("dropout_fcl_ner"))
        self.dropout_fcl_rel = float(config_file.getProperty("dropout_fcl_rel"))
        #self.hidden_size_lstm =int(config_file.getProperty("hidden_size_lstm"))
        self.hidden_dim = int(config_file.getProperty("hidden_dim"))
        #self.hidden_size_n2 = config_file.getProperty("hidden_size_n2")
        self.num_lstm_layers = int(config_file.getProperty("num_lstm_layers"))

        # evaluation
        #self.evaluation_method =config_file.getProperty("evaluation_method")
        #self.root_node=bool(config_file.getProperty("root_node"))

        self.shuffle=False
        #self.batchsize=1





