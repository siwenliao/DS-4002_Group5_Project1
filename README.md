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
- Google Colab/Jupyter Notebook

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

### Folder Descriptions
- **DATA/**: Contains original dataset of 33,396 McDonald's Reviews downloaded from Kaggle (scraped from Google), the cleaned dataset after preprocessing, and the cleaned dataset along with topics, topic probabilities, and sentiments assigned to each review. Also contains data appendix that includes more information about each dataset.
- **SCRIPTS/**: Contains python scripts for preprocessing, LDA modeling, and hypothesis testing.
- **OUTPUT/**: Contains screenshots of the sample coherence scores when finding the optimal k value to fit the LDA model and the evaluation metrics for the final model. Contains PDFs of hypothesis testing results, the topic-word probability distribution for the LDA model, the most common topics by sentiment (positive, negative, and neutral), and the topic distribution for positive and negative reviews.
- **LICENSE/**: MIT license was selected based on recommendation from the DS4002 Ml3 Rubric.
- **README.md/**: Instructions, documentation, and respository overview. 

## Section 3: Instructions for Reproduction

- Step 1: 

