# Text Forensics: Who Sent the Postcard?  
> A tiny authorship-attribution demo that uses classic NLP to unmask the writer of a 1908 Arctic postcard.

## What it does  
Three friends—Emma Goldman, Matthew Henson and Wu Ting-Fang—have left us 461 personal letters.  
We treat each letter as a bag-of-words, train a Naïve Bayes classifier on the vocabulary habits of each friend, and then ask the model to name the most likely author of a mysterious postcard found in the archives.

## Dataset  
| Author        | Letters | Source notebook               |
|---------------|---------|-------------------------------|
| Emma Goldman  | 154     | `goldman_emma_raw.ipynb`      |
| Matthew Henson| 141     | `henson_matthew_raw.ipynb`    |
| Wu Ting-Fang  | 166     | `wu_tingfang_raw.ipynb`       |

The raw text cells are imported at runtime with `import_ipynb`.

## How it works  
1. Concatenate all letters into one list (`friends_docs`).  
2. Build a joint vocabulary with `sklearn.feature_extraction.text.CountVectorizer`.  
3. Vectorise every letter → sparse term-frequency matrix (`friends_vectors`).  
4. Attach labels: 154 × “Emma”, 141 × “Matthew”, 166 × “Tingfang”.  
5. Train `MultinomialNB` on the labelled vectors.  
6. Vectorise the mystery postcard with the **same** vocabulary.  
7. Predict the author → prints the most probable friend.

## Requirements
Python ≥ 3.8
scikit-learn
import-ipynb
