# Sentiment Classification

Sentiment classification using a simple LSTM model.


#### [Report](https://drive.google.com/file/d/1TsYpc00hfyAFGBvnTVzrSag8ijz80Tfd/view?usp=share_link)


**PROBLEM**

The task is to create a predictive model for classifying text as positive, neutral, or negative (1, 0 or -1), using **only** the given dataset and without using any pretrained models.

**APPROACH**


**ARCHITECTURE**

After experimenting with various architectures, I have settled on the following:
1. (Optional) Static (non-trainable) GloVe embedding function that gives a stacked tensor of 100 dimensional word embeddings of the input sentence (padded or truncated to 64 words).
2. Stacked LSTM with 8 LSTM layers 
3. Linear layer with softmax that outputs probabilities for the 3 classes
4. (Optional) Max layer that outputs the corresponding predicted goldlabel

This configuration was trained for ~50 epochs on CPU alone (each epoch took around 2 minutes only). It is considerably faster and less resource intensive than  a transformer implementation and gives comparable results.



**TRAINING**

Since converting sentences to embedding representations for each sentence in the train set in every epoch would have been time-consuming, I solved this issue by preprocessing training sentences into embedding tensors using the aforementioned function beforehand and saving it in a DataLoader. I also converted the training set goldlabels into one hot vectors that could directly be passed into the Cross Entropy loss function. This led to a very efficient training process that was carried out on CPU in less than two hours.

The Adam optimiser, was initially used with default settings. In later iterations, it had stopped showing much improvement and decreasing the learning rate showed better results. 
The initial loss was approximately 1.1 and after training it stood at 0.744.


**PERFORMANCE**

It achieved an F1 score of 0.656 on the given validation dataset, compared to the highest achieved score of 0.677.
