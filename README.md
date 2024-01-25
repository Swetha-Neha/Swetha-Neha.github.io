# Harnessing Machine Learning to Decode User Reviews in the Mobile App Landscape Using Extra-Long Neural Network (XLNet) #

Mobile applications have become an attractive arena for almost every commercial use case. The reviewer’s viewpoint of an application is proportional to their level of satisfaction with it. Consequently, this helps other users obtain insights before downloading or purchasing the apps. Ratings and reviews in the app marketplaces present a unique opportunity to collect candid, unfiltered feedback about the products in real time. Natural Language Processing has evolved so significantly that it can now identify the emotion from text data. In this research, XLNet, a pre-trained permutation language model is implemented to analyze the sentiment. The user review data from the Google Play Store were extracted using Python web scraping followed by Exploratory Data Analysis, a rigorous data preprocessing step to achieve clean and consistent data. The preprocessed data was then used in the model-building phase starting with word embedding to convert the discrete tokenized words to continuous vector representations. An attention mask was implemented to focus on the sequence of the input phrase when making predictions. The final step the model-building phase with pre-training lets the model capture the contextual relationship within the input data. To get the optimal performance of the model, hyperparameter tuning was performed.


## Model Architecture ##
The data flow architecture of the XLNet starts with the input layer. The preprocessed data after the transformation phase is split into 80% training data, 10% validation data, and the remaining 10 % for testing purposes. The pre-trained model includes 12 hidden layers of encoder-decoder transformer blocks with 768 hidden layers in each, a feed-forward layer size of 3072, and 12 self-attention heads. The input data is tokenized with positional encoding before training the model. Figure 4 represents the XLNet architecture and the complete data flow.


### Input Layer ###
The preprocessed raw data after data transformation is passed to the input layer.

#### Architecture and Data Flow of XLnet ####

<div align=”center”>
    <img src= https://github.com/Swetha-Neha/Swetha-Neha.github.io/assets/124639055/259e8ff2-b313-4ca4-b28b-d103a98cd123>
</div>


### Embedding Layer ###
The objective of the embedding layer is to encode each token’s inherent meaning and relationship with other words. In this study, the embedding layer is set to a size of 128 containing the semantic and contextual information of the input sentence. The model utilizes pre-trained word embeddings to map the tokens to high-dimensional vectors. In this phase, the position of each word is calculated based on the equation (1) and (2). The embeddings are vectors representing the position of the tokens in the sequence. This helps the model understand the long-range dependencies and context. The token and positional embeddings are concatenated to create a single vector representation for each word.  The quality of embedding significantly impacts the model's performance and pre-trained on large corpus data captures rich semantic information. These embeddings lay the groundwork and transferred to the two-stream self-attention layer.

### Two-Stream Self-Attention and Feed-Forward Layer ###
This is a pivotal layer where the XLnet model excels in understanding the bidirectional context. The content layers of the encoder are where the two-stream attention uses standard self-attention mechanisms to understand the meaning of the words. The query layer is where the positions and context of the tokens are captured by computing possible permutations of the input sequence. At this stage, XLNet learns how the change in the order of the token changes the meaning of the sentence. . After integration, the query stream and content are sent to the feed-forward network for further enhancement. The input of the feed-forward network is the output of the two-stream self-attention mechanism in the previous layer which is transformed into multiple keys (K), value(V), and query(Q) vectors. These transformations involve weights for each token and relative positioning introduces the non-linear positional attention of the tokens. The attention scores are calculated by taking the dot product of modifying the query and key vectors. These scores represent the weight of the complete vector and are added as the output of the attention layers. The old memory stores the attention score of the previous attention layer and the new memory stores the new attention information relevant to the current attention score computation. The add and norm block of the Transformer-XL model adds the output of the previous attention layers to the input of the next attention layer. This feed-forward network of encoder works the same as a standard encoder except that the decoder includes the cross-attention mechanism for focusing on the relevant tokens for the downstream tasks and Masked self-attention where the model tries to predict the future tokens by the masking method. Transformer-XL model does not explicitly distinguish the encoder and decoder instead has a recurrence feed-forward network. The output of each attention layer is then concatenated to be passed as input to the Softmax layer.


#### Transformer-XL Model ####


![image](https://github.com/Swetha-Neha/Swetha-Neha.github.io/assets/124639055/4a745ffe-a9ad-46ce-a56d-1da9c17186d5)


                     
Note. The above figure depicts the transformer-XL architecture starting from the input embedding, feed-forward network, and the final Softmax output layer. From “ Language Modeling for Source Code with Transformer-XL“ by  Dowdell, T., & Zhang, H. (2020).arXiv preprint arXiv:2007.15813.
 
### Softmax Layer ###
 The final layer of the XLNet model is the softmax layer which provides the probability distribution of the multiclass sentiment classification task. In this study, the input information is classified into positive, negative, and neutral target classes. The softmax layer produces a vector of probabilities for each class and the class with the highest probability is predicted as the output class.
