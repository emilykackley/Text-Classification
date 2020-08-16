# Text-Classification
- 2018

## Convolutional Neural Network (CNN) for the purpose of Text Classification
  ### Approach
  - Load all parameters for training and testing
  - Prepare the data for text classification
  - Build the vocabulary
    * Determine the maximum lenght of the text document by splitting it into words from the combined data of positive and negative samples. 
    * Process the vocabulary by using the learn library from tensorflow.contrib
    * Use VocabularyProcessor using the maximum length of the text document
  - Randomly shuffle the data to creat a model with better results
  - Split the data into training and validation data (10% validation, 90% training)
  - Begin training the model with a tensorflow graph and session.
  - Create a CNN model
  - The training procedure is defined by creating the global step, optimizer, gradients, and the training optimizer
  - Test the model by using the validation set by repeating above for validation set
## Recurrent Neural Network (RNN) for the purpose of Text Classification
  ### Approach
  - Load all parameters for training and testing
  - Prepare the data for text classification
  - Create the dictioinary and reverse dictionary from x_text
  - Begin training the model with a tensorflow graph and session. 
  - Create a RNN model
  - The training procedure was defined by creating the global step, optimizer, gradients, and the training optimizer
  - Test the model by using the validation set by repeating above for the validation set
## LSTM for the purpoase of Text Classification
  ### Approach
  - Load all parameters for training and testing
  - Create the dictionary and reverse dictionary from x_text
  - Begin training the model with a tensorflow graph and session
  - Create a LSTM model
  - The training procedure was then defined by creating the global step, optimizer, gradients, and the training optimizer
  - Test the model by using the validation set by repeating above for validation
## Limitations 
- This implementations is limited to a fixed learning rate. Ideally, the learning rate should be adapting in each iteration
- CNN's and RNN's have a hight computational cost and are slow to train without a good GPU
