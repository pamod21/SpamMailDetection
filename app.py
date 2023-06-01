import gradio as gr
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.corpus import stopwords
import pandas as pd
import string

nltk.download('stopwords')
nltk.download('punkt')

# Load the dataset
df = pd.read_csv("mail_DataSet.csv", encoding='latin1')

# Perform preprocessing and data transformations
df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
df.rename(columns={'v1': 'target', 'v2': 'text'}, inplace=True)
df.loc[df['target'] == 'spam', 'target'] = 0
df.loc[df['target'] == 'ham', 'target'] = 1
df.drop_duplicates(keep='first', inplace=True)
df.reset_index(drop=True, inplace=True)
df['char_count'] = df['text'].apply(len)
df['word_count'] = df['text'].apply(lambda x: len(nltk.word_tokenize(x)))
df['sent_count'] = df['text'].apply(lambda x: len(nltk.sent_tokenize(x)))

# Convert target column to integer type
df['target'] = df['target'].astype(int)

# Define and initialize the TF-IDF vectorizer
tfidf = TfidfVectorizer(max_features=3000)

# Fit the TF-IDF vectorizer on your transformed text data
X = tfidf.fit_transform(df['text']).toarray()

# Initialize the Multinomial Naive Bayes classifier
mnb = MultinomialNB()

# Fit the classifier on your training data
mnb.fit(X, df['target'])


def transform_text(text):
    # Convert text to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    
    # Tokenize the text
    tokens = nltk.word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    
    # Join the tokens back into a single string
    transformed_text = " ".join(filtered_tokens)
    
    return transformed_text

def hehe(text):
    x = transform_text(text)
    vect = tfidf.transform([x])
    result = mnb.predict(vect)
    if result == 0:
        div = "spam"
    else:
        div = "ham"
    return div

demo = gr.Interface(
    fn=hehe, enable_timeout=False,
    inputs=gr.Textbox(placeholder="Enter an email..."), 
    outputs="label", 
    examples=[["enter an email"]]
)

demo.launch()
