import tensorflow
import numpy
import os
import time
import datetime
import FileFunction
import TextConvNet
from tensorflow.contrib import learn
import Run_TextCNN
import Run_TextRNN
import Run_TextLSTM

#%%Parameters%%#

# Data loading params
#Percent of data being split for validation set
dev_percent = 0.1
#Data file for positive examples
pos_dat_file = './data/rt-polaritydata/rt-polarity.pos'
#Data file for negative examples
neg_dat_file = './data/rt-polaritydata/rt-polarity.neg'

#-->Hyperparameters
#Embedding size
embed_size = 128
#String for filter size of 3, 4, and 5
temp = "3,4,5"
#Filter size
fil_sz = list(map(int, temp.split(",")))
#Number of filters
fil_num = 128
#Original dropout keep probability
drop_keep = 0.5
#Initialize l2_lambda_reg as 0.0
l2 = 0.0
#Learning Rate
learning_rate = 1e-5

#-->Training parameters
#Size of each batch
batch_sz = 64
#Total number of epochs
epoch_num = 200
#Validation evaluations
eval_on_dev = 100
#Checkpoint evaluations
ckpt = 100
#Total number of checkpoints
ckpt_num = 5

#-->Misc Parameters
soft_pl = True
log_pl = False

#%%Prep Data for Text Classification%%#

#-->Load data
print("\nLoad data from input files...")
x_text, y = FileFunction.import_data(pos_dat_file, neg_dat_file)

#-->Build vocabulary
print("\nBuilding Vocabulary...")
#Determine the maximum length of the document from the combined data of positive and negative samples
max_document_length = max([len(x.split(" ")) for x in x_text])
#Process the vocabulary using VocabularyProcessory from tensorflow.contrib learn
process_voc = learn.preprocessing.VocabularyProcessor(max_document_length)
#Convert data into numpy array and use fit_transform from the vocabulary above
x = numpy.array(list(process_voc.fit_transform(x_text)))

#-->Randomly shuffle data
#Random number generator through numpy
numpy.random.seed(10)
#Create index for start and end for randomly shuffled data
shuffle_indices = numpy.random.permutation(numpy.arange(len(y)))
#Shuffled value of x
x_shuffled = x[shuffle_indices]
#Shuffled value of y
y_shuffled = y[shuffle_indices]

#-->Split data into training and testing data.
#Testing/Validation data is dev_percent of data. In this case it is 10% of the data
index_dev = -1 * int(dev_percent * float(len(y)))
x_train, x_dev = x_shuffled[:index_dev], x_shuffled[index_dev:]
y_train, y_dev = y_shuffled[:index_dev], y_shuffled[index_dev:]

Run_TextCNN.Run_TextCNN(x_train,y_train,process_voc,embed_size,fil_sz,fil_num,learning_rate,soft_pl,log_pl,l2,
                        ckpt,ckpt_num,epoch_num,batch_sz,eval_on_dev,drop_keep)



dictionary,reverse_dictionary = Run_TextRNN.build_dataset(x_text)

Run_TextRNN.run_RNN(len(dictionary),learning_rate,dictionary,reverse_dictionary,x_text)

Run_TextLSTM.run_LSTM(len(dictionary),learning_rate,dictionary,reverse_dictionary,x_text)