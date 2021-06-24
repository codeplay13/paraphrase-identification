# paraphrase-identification

### Note:
* As of now, there are two techniques implemented. 
1. Bag of Words to featurize and cosine similarity to calculate similarity. Although it gives a good result on test set with **0.8 F1-score**, it might not work well on simple sentences (for eg. I am good vs. I am not good) as Bag of Words doesn't take the order of words into consideration. This can be solved to a certain point by using *ngram_range* parameter in *CountVectorizer* (for eg. ngram_range = (1, 3)).
2. Sentence embedding using MiniLM model with cosine similarity. MiniLM has the best tradeoff between speed and performance.
* Another good approach would be to train a classifier model on the dataset after extracting features from the texts. This might not work well as the size of the dataset is not very large (~3k) and the number of features can be very high.
* New datapoints can be generated for not-matching texts to augment the dataset.

### There are two main contents:
1. paraphrase_identification.py - Builds the model on the dataset and calculates F1-Score on test set.
2. app.py - A Flask web application to provide a simple interface for paraphrase identification.


## Steps to run:

First, clone the repository or download as zip. Open the terminal/cmd and get into ```paraphrase_identification``` as working directory.

* To calculate the F1-Score on test dataset, run ```python paraphrase_identification.py```
* To run the Dockerfile, follow the steps:
1. Build image from Dockerfile: ```docker build -t webapp .```
2. Once it is built successfully, run the image: ```docker run -p 5000:5000 webapp```
3. Open the browser and paste the link ```http://localhost:5000```
