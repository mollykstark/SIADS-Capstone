# SIADS Capstone: University of Michigan School of Information Applied Data Science
*Team 11 (The Aces)*: Book Recommender from Goodreads Data

## Table of Contents:
```
SIADS-MILESTONE-1
└───src
    │   app.py
    │   
    ├───data
    │   │   10000_reviews.csv
    │   │   authors.csv
    │   │   book_titles.csv
    │   │   books_meta.csv
    │   │   desc_emb.npy
    │   │   merged_df.csv
    │   │   rev_emb.npy
    │   
    └───notebooks
        │   book_recommender_system.ipynb
        │   creating_csvs.ipynb 
        │   data_preprocessing.ipynb
```

## How to run our code:

* Due to GitHub size limits, we are unable to upload the original JSON files that we sourced our data from, and instead uploaded the csv files we created from that data. The `creating_csvs.ipynb` notebook takes in those JSON files and creates the csvs, so there is no need to run it to use our recommender system. We have included it in our repository to preserve data provenance. The JSON files required to run this notebook are `goodreads_book_authors.json`, `goodreads_books.json`, and `goodreads_reviews_spoiler_raw.json`.
* We chose to separate our data preprocessing into a separate notebook (`data_preprocessing.ipynb`) from our recommender system notebook to have a clearer delineation between project steps. At the end of this notebook, all relevant data is saved in a single csv file that is included in this repository. Due to this, it is also unnecessary to run this notebook to use our recommender system.

__There are 2 methods a user can use to access our recommender system:__
1. To access the dashboard locally, you can run the `book_recommender_system.ipynb` notebook to create an instance of the dashboard within your IDE, as well as a locally hosted instance that can be accessed using your browser.
2. The other way a user can access our recommender system is by simply navigating to the [instance of our dashboard that is hosted on Hugging Face](https://huggingface.co/spaces/umichmads/capstone2025). There is no need to run any code with this method.


## Data Access Statement:
All the data we used over the course of this project was accessed from [this website](https://cseweb.ucsd.edu/~jmcauley/datasets/goodreads.html). On this website they state that they collected this data for academic use only, and request that anyone who uses the datasets cite the following sources:
* Mengting Wan, Julian McAuley, "Item Recommendation on Monotonic Behavior Chains", in RecSys'18. [[bibtex]](https://dblp.uni-trier.de/rec/bibtex/conf/recsys/WanM18)
* Mengting Wan, Rishabh Misra, Ndapa Nakashole, Julian McAuley, "Fine-Grained Spoiler Detection from Large-Scale Review Corpora", in ACL'19. [[bibtex]](https://dblp.uni-trier.de/rec/bibtex/conf/acl/WanMNM19)