# '''https://www.ted.com/talks/andrew_zimmerman_jones_does_time_exist/transcript'''
import re
import string
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim
import numpy as np
import nltk.data
import ssl
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from transformers import pipeline
import string
import os
import nltk
from nltk.tokenize.treebank import TreebankWordDetokenizer
import spacy
from summa import keywords
from gensim import corpora, models
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from rouge_score import rouge_scorer
from sklearn.metrics import precision_score, recall_score, f1_score

# To disable SSL certificate verification in Python
ssl._create_default_https_context = ssl._create_unverified_context

nltk.data.path.append("/Users/user/nltk_data")
nltk.download('punkt')
nltk.download('stopwords')
nlp = spacy.load("en_core_web_sm")


# Specify the path to the .txt file
filepath = "/Users/user/Desktop/Fall Semester/CS688 A2 Web Mining and Graph Analytics - Zlatko Vasilkoski/Project/Submission/Transcripts_Does_Time_Exist.txt"
output_filepath = "/Users/user/Desktop/Fall Semester/CS688 A2 Web Mining and Graph Analytics - Zlatko Vasilkoski/Project/"

# read and extract from the .txt file
def extract_file(filepath):
    try:
        text = ""
        with open(filepath, "r") as file:
            lines = file.readlines()
            for line in lines:
                # print(line.strip()) # this is printing the transcript
                text += str(line) # Convert line to string and concatenate the lines into the 'text' variable

        return text
    except FileNotFoundError:
        print(f"The file {filepath} was not found.")
    except Exception as e:
        print("Error reading .txt:", e)
        return ""

# Preprocess the text
def preprocess_text(text):
    if text is not None and not isinstance(text, str):
        text = str(text)
    if text is not None:
        # Tokenize the text into sentences
        sentences = sent_tokenize(text)
        words = [word_tokenize(sentence) for sentence in sentences]
        # Include punctuations and convert to lowercase for each word
        cleaned_words = [word.lower() for sublist in words for word in sublist if
                         (word.isalpha() or word in string.punctuation) and word.lower() not in stopwords.words('english')]
        # Join the words back into a cleaned text
        cleaned_text = ' '.join(cleaned_words)
        return cleaned_text
    else:
        return ""

# Read and preprocess the text
text = extract_file(filepath)
# print(text)
preprocessed = preprocess_text(text)

# Use Word CLoud to check the frequency of words (the size of the text indicates its frequency)
# generate word cloud
# generate the wordcloud object, set the height and width, set the random_state parameter to ensure
# reproducibility of results and set the stopwords parameter so that the irrelevant words such as pronouns are discarded.
# text is the input to the generate() method
wordcloud = WordCloud(width = 300, height = 200, random_state=1, background_color='blue', collocations=False, stopwords = STOPWORDS).generate(text)
plt.figure(figsize=(12, 8))
# Display image
plt.imshow(wordcloud)
# No axis
plt.axis("off")
plt.show()

#  KEYWORD EXTRACTIONS
# Function to extract keywords using TF_IDF
def extract_keywords_tfidf(preprocessed):
    stop_words = 'english'
    vectorizer = TfidfVectorizer(stop_words=stop_words)
    tfidf_matrix = vectorizer.fit_transform([preprocessed])
    feature_names = vectorizer.get_feature_names_out()
    sorted_indices = np.argsort(tfidf_matrix.sum(axis=0))[0, ::-1]
    keywords_tfidf = [feature_names[idx] for idx in sorted_indices]
    return keywords_tfidf

# Extract Keywords using TF_IDF
keywords_tfidf = extract_keywords_tfidf(preprocessed)
print("Keywords Extracted Using TF-IDF:", "\n",  keywords_tfidf)

# Extracted keywords using LDA
# Tokenize the preprocessed text
tokenized_text = preprocessed.split()
# Create a dictionary and a corpus
dictionary = corpora.Dictionary([tokenized_text])
corpus = [dictionary.doc2bow(tokenized_text)]
# Train the LDA model
lda_model = models.LdaModel(corpus, num_topics=1, id2word=dictionary)
# Get the top keywords from the LDA model
lda_keywords = [dictionary[word_id] for word_id, _ in lda_model.get_topic_terms(0, topn=20)]
# Print the extracted keywords
print("Keywords Extracted using LDA:", "\n", lda_keywords)
print()

# Keywords extraction using TextRank
text_rank_keywords = keywords.keywords(preprocessed)
# Print the extracted keywords
print("Keywords Extracted using TextRank:", "\n", text_rank_keywords)
print()


# SUMMARIZATIONS
# BERT-based abstractive summarization
# The specific BERT model used in this case is the default one, which is "facebook/bart-large-cnn" for abstractive summarization.
def bert_summarization(text):
    summarizer = pipeline("summarization")
    summary = summarizer(text, max_length=150,min_length=50, do_sample=False)
    return summary[0]['summary_text']

# BERT-based abstractive summarization using the original text
bert_summary = bert_summarization(text)
# save the BERT_based abstractive summaries in separate files or data structures
bert_summary_filepath = os.path.join(output_filepath, 'bert_summary.txt')
with open(bert_summary_filepath, "w") as file:
    file.write(bert_summary)

# BERT-based abstractive summarization using the preprocessed text
bert_summary_p = bert_summarization(preprocessed)
# save the BERT_based abstractive summaries in separate files or data structures
bert_summary_filepath = os.path.join(output_filepath, 'bert_summary_preprocessed.txt')
with open(bert_summary_filepath, "w") as file:
    file.write(bert_summary_p)


# Summarization using NLTK
def summarize_with_nltk(text, num_sentences=5):
    # Tokenize the text into sentences
    sentences = sent_tokenize(text)
    # Combine preprocessed sentences into a single string
    text_combined = ' '.join(sentences)
    words = [word.lower() for word in nltk.word_tokenize(text_combined) if
             word.isalpha() and word.lower() not in stopwords.words('english')]
    frequency = nltk.probability.FreqDist(words)
    important_sentences = [
        (sentence, sum(frequency[word] for word in nltk.word_tokenize(sentence.lower()) if word in frequency)) for
        sentence in sentences]
    # Sort sentences by importance and select top sentences
    selected_sentences = sorted(important_sentences, key=lambda x: x[1], reverse=True)[:num_sentences]
    # Detokenize selected sentences to form the summary
    summary = ' '.join(sentence for sentence, score in selected_sentences)

    return summary

summary_result = summarize_with_nltk(preprocessed)

NLTK_summary_filepath = os.path.join(output_filepath, "NLTK_summary.txt")
with open(NLTK_summary_filepath, "w") as file:
    file.write(summary_result)


# Summarization using spaCy
def summarize_with_spacy(text, num_sentences=10):
    # Process the input text using SpaCy
    doc = nlp(text)
    # Calculate the importance score for each sentence based on the sum of token ranks
    sentence_scores = [sum(token.rank for token in sent) for sent in doc.sents]
    # Get the indices of the top-scoring sentences
    top_sentence_indices = sorted(range(len(sentence_scores)), key=lambda i: sentence_scores[i], reverse=True)[:num_sentences]
    # Get the top sentences from the original text
    selected_sentences = [list(doc.sents)[i].text for i in top_sentence_indices]
    # Combine the selected sentences to form the summary
    summary = ' '.join(selected_sentences)

    return summary

summary_result_spacy = summarize_with_spacy(preprocessed)

spacy_summary_filepath = os.path.join(output_filepath, "Spacy_Summary.txt")
with open(spacy_summary_filepath, "w") as file:
    file.write(summary_result_spacy)


# QUALITY OF SUMMARIZED TEXT
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])

def print_rouge_scores(label, rouge_scores):
    print(f"ROUGE Scores for Summarization ({label}):")
    print(f"ROUGE-1: {rouge_scores['rouge1'].fmeasure}")
    print(f"ROUGE-2: {rouge_scores['rouge2'].fmeasure}")
    print(f"ROUGE-L: {rouge_scores['rougeL'].fmeasure}")
    print()

# Calculate and print ROUGE scores for BERT
rouge_scores_b = scorer.score(preprocessed, bert_summary_p)
print_rouge_scores("BERT", rouge_scores_b)

# Calculate and print ROUGE scores for NLTK
rouge_scores_NLTK = scorer.score(preprocessed, summary_result)
print_rouge_scores("NLTK", rouge_scores_NLTK)

# Calculate and print ROUGE scores for spaCy
rouge_scores_s = scorer.score(preprocessed, summary_result_spacy)
print_rouge_scores("spaCy", rouge_scores_s)


# KEYWORDS EXTRACTION AFTER SUMMARIZATION
# Extracted Keywords using TF-IDF
# Apply preprocessing steps to the summarized text
preprocessed_summarized_text = preprocess_text(bert_summary_p)
# Perform the second keyword extraction on the preprocessed summarized text
second_extraction_TFIDF = extract_keywords_tfidf(preprocessed_summarized_text)
# Print the second set of extracted keywords
print("Keywords from Second Extraction after BERT-based Summarization:")
print(second_extraction_TFIDF)
print()

# Extracted keywords using LDA
# Tokenize the preprocessed text
tokenized_text = bert_summary_p.split()
# Create a dictionary and a corpus
dictionary = corpora.Dictionary([tokenized_text])
corpus = [dictionary.doc2bow(tokenized_text)]
# Train the LDA model
lda_model = models.LdaModel(corpus, num_topics=1, id2word=dictionary)
# Get the top keywords from the LDA model
lda_keywords_new = [dictionary[word_id] for word_id, _ in lda_model.get_topic_terms(0, topn=10)]
# Print the extracted keywords
print("Keywords using LDA  after BERT-based Summarization:")
print(lda_keywords_new)
print()

# Keywords extraction using TextRank
text_rank_keywords_new = keywords.keywords(bert_summary_p)
# Print the extracted keywords
print("Keywords using TextRank after BERT-based Summarization:")
print(text_rank_keywords_new)
print()


# PERFORMANCE METRICS FOR KEYWORD EXTRACTION
def performance_metrics_keyword_extract(label, keywords, second_extraction):
    precision = precision_score(keywords, second_extraction, average='micro')
    recall = recall_score(keywords, second_extraction, average='micro')
    f1 = f1_score(keywords, second_extraction, average='micro')

    print(f"Metrics for Extracted Keywords using {label}:")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-Score: {f1}")
    print()

# Using the first 10 words (TF_IDF)
keywords_tfidf_summ_p = ['time', 'universe', 'fundamental', 'physics', 'property', 'exist', 'individual', 'equation', 'coloring', 'einstein']
second_extraction_TFIDF_summ_p = ['time', 'bend', 'einstein', 'fundamental', 'ways', 'combining', 'complicated', 'consistent', 'define', 'describing']
performance_metrics_keyword_extract("TF-IDF", keywords_tfidf_summ_p, second_extraction_TFIDF_summ_p)

# Using the first 10 words (LDA)
keywords_LDA_summ_p = [',', '.', 'time', 'universe', '?', 'one', 'fundamental', 'physics', 'property', 'exist']
second_extraction_LDA_summ_p = [',', 'time', '.', 'bend', 'einstein', 'fundamental', 'always', 'behaves', 'predictable', 'pass']
performance_metrics_keyword_extract("LDA", keywords_LDA_summ_p, second_extraction_LDA_summ_p)

# Using the first 10 words (TextRank)
keywords_TR_summ_p = ['time measurements', 'equation', 'ways', 'change', 'equations describing']
second_extraction_TR_summ_p = ['time', 'relativity', 'fundamental', 'complicated', 'equations describing']
performance_metrics_keyword_extract("Textrank", keywords_TR_summ_p, second_extraction_TR_summ_p)







