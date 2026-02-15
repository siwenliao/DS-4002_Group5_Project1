# Topic Modeling Analysis of McDonald’s Restaurant Reviews
- DS 4002 Group 5, Project 1
- Group Leader: Siwen Liao
- Group Members: Siwen Liao, JJ Sutkus, Jonah Lee
- Feb 14, 2026

## Repository Contents
The goal of this project is to use Latent Dirichlet Allocation (LDA) to understand the most common themes in positive and negative McDonald's restaurant reviews to pinpoint the area most in need of improvement. The DS-4002_Group5_Project1 repository contains the DATA folder (contains three versions of our dataset of interest with more information located in the data appendix), the SCRIPTS folder (contains our code for preprocessing/cleaning the dataset, fitting the LDA model and assigning topics and sentiments to reviews, and performing a hypothesis test), the OUTPUTS folder (contains screenshots or PDFs of our results from LDA and the hypothesis test), and the LICENSE and READ.md folders.

## Section 1: Software and Platform

### Software/Platform
This project was developed and run using: 
- Google Colab/Jupyter Notebook on a Mac

### Packages
The following Python packages are required:
- spacy.cli
- pandas
- string
- nltk
- en_core_web_md
- re
- gensim
- corpora (from gensim)
- pyLDAvis
- pyLDAvis.gensim_models
- CoherenceModel (from gensim.models)
- random
- numpy
- scipy
- pathlib

## Section 2: Documentation Map
```text
DS-4002_Group5_Project1/
│
├── DATA/
│ ├── Data Appendix.pdf
│ ├── McDonalds_Reviews.csv
│ ├── McDonalds_Reviews_Cleaned.csv
│ ├── McDonalds_Reviews_With_Topics.csv
│
├── OUTPUT/
│ ├── Hypothesis_Testing.pdf
│ ├── LDA Topic-Word Probability Distribution .pdf
│ ├── Most_Common_Topics_By_Sentiment.pdf
│ ├── Topic_Distribution_For_PositiveNegative_Reviews.pdf
│ ├── coherence_different_k_values.png
│ ├── evaluation_metrics.png
│
├── SCRIPTS/
│ ├── 01_preprocess.py
│ ├── 02_doc_term_matrix_and_LDA.py
│ ├── 03_hypothesis_test.py
│
├── LICENSE
└── README.md
```

### Folder Descriptions
- **DATA**: Contains original dataset of 33,396 McDonald's Reviews downloaded from Kaggle (scraped from Google), the cleaned dataset after preprocessing, and the cleaned dataset along with topics, topic probabilities, and sentiments assigned to each review. Also contains data appendix that includes more information about each dataset.
- **SCRIPTS**: Contains python scripts for preprocessing, LDA modeling, and hypothesis testing.
- **OUTPUT**: Contains screenshots of the sample coherence scores when finding the optimal k value to fit the LDA model and the evaluation metrics for the final model. Contains PDFs of hypothesis testing results, the topic-word probability distribution for the LDA model, the most common topics by sentiment (positive, negative, and neutral), and the topic distribution for positive and negative reviews.
- **LICENSE**: MIT license was selected based on recommendation from the DS4002 Ml3 Rubric.
- **README.md**: Instructions, documentation, and respository overview. 

## Section 3: Instructions for Reproduction

- **Step 1**: Clone the repository. Cloning creates a complete local copy of the repository, including all files and branches. Make sure that you can see the DATA, OUTPUT, and SCRIPTS folders.
- **Step 2**: In Google Colab, or the desired platform of choice, run the 01_preprocess.py script. Make sure DATA/McDonalds_Reviews.csv exists. Note that the code to install the necessary packages and make the necessary downloads are included. The 01_preprocess.py script will read the McDonalds_Reviews.csv dataset, lowercase all text, remove punctuation and digits, and discard short and non-informative words (stopwords) to isolate only the words (generally nouns and adjectives) that contain the most information about the topic of the review. It will also reduce words to their base form (fries to fry, parking to park, etc.) and produce the cleaned dataset: McDonalds_Reviews_Cleaned.csv. 
- **Step 3**: After preprocessing the data, run the 02_doc_term_matrix_and_LDA.py script. This script will read the McDonalds_Reviews_Cleaned.csv dataset, tokenize reviews, build the dictionary and bag-of-words document-term matrix to prepare for LDA, evaluate mutiple k values based on coherence scores to find the most optimal k value for the LDA model, fits the LDA model with the chosen k value on the entire dataset, and output the topic-word distribution (topics and words that indicate each topic). Then, it will also assign each review to the most-likely topic and produce the dataset with topic assignments. The output should include the coherence scores for the multiple k values, the topic-word distribution output, the perplexity and coherence scores for the fitted LDA model, the most common topics based on sentiment of reviews, and the topic distribution for positive and negative reviews.
  - Some notes for Step 3:
    - You will need to uncomment the !pip install --upgrade gensim pyLDAvis spacy pandas scikit-learn if you do not have the necessary packages already installed.
    - Since LDA is a probabilistic model and the optimal k value is found by computing coherence scores of different k values using a random sample of the full dataset to reduce runtime, we fix a random seed to improve consistency in results and help with reproducibility. 
    - While the coherence score for k=3 was the highest, k=6 was chosen because it was the highest k value before a noticeable dip in coherence for k >= 7. A higher k value implies more topics existing throughout the data, which provide more interesting results. In summary, k=6 was a balance between coherence scores and producing interesting/interpretable results.
    - The topic labels for the 6 different topics are:
      - 0: 'Bad service, wrong orders, rude staff'
      - 1: 'Fast and quick service, clean place, friendly staff'
      - 2: 'Overall good price, quality, service'
      - 3: 'Food quality issues (cold/old fries, burgers, nuggets)'
      - 4: 'Long wait times, slow drive-thru'
      - 5: 'Miscellanous, general complains about environment'
    - 4 or 5-star reviews are considered positive, 1 or 2-star reviews are considered negative, and 3-star reviews are considered neutral. 
    - This script may run for 5-10 minutes.
- **Step 4**: The 03_hypothesis_test.py script performs two one-proportion z-tests to test whether at least 30% of positive and negative reviews are assigned to food quality and bad service, respectively. It also reports the distribution of the 6 topic labels by sentiment. The z-test conclusions indicate whether food quality and customer service at at least as prevalent as the 30% benchmark.
- **Step 5**: Verify your outputs match those in the OUTPUT folder. 


