Absolutely! Here's the **rephrased competition overview** for *Spooky Author Identification*, following your preferred structure and tone:

---

## Goal  
Your task is to predict the author of short text excerpts. Each sentence in the dataset is written by either Edgar Allan Poe, HP Lovecraft, or Mary Shelley. For each `id` in the test set, you must predict the probability that the sentence was written by each of the three authors.

## Background  
This challenge invites you to dive into eerie excerpts from classic horror literature. By analyzing writing patterns and linguistic features, you'll train a classifier that can distinguish between the styles of Poe, Lovecraft, and Shelley. The dataset has been created by segmenting longer texts into individual sentences using CoreNLP, making it a fun and spooky introduction to natural language processing.

## Evaluation  
Submissions are evaluated using **multi-class logarithmic loss**. For each sentence in the test set, you must submit the predicted probabilities of it being written by each author.

Your submission file should include a header and follow this format:

```
id,EAP,HPL,MWS
id07943,0.33,0.33,0.33
id12345,0.90,0.05,0.05
...
```

## Data description
- **train.csv** – the training set with `id`, `text`, and `author`  
- **test.csv** – the test set without `author`  
- **sample_submission.csv** – a sample output file in the correct format


