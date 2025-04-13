import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

df = pd.read_csv("training.1600000.processed.noemoticon.csv", encoding="ISO-8859-1", header=None)
df.columns = ["target", "id", "date", "flag", "user", "text"]

df = df[["target", "text"]]

def clean_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    return text

df["text"] = df["text"].apply(clean_text)

df["tokens"] = df["text"].apply(word_tokenize)

df["clean_text"] = df["tokens"].apply(lambda tokens: " ".join(tokens))

# print(df[df["target"] == 0]["tokens"].sample(1).values[0])

vectorizer = TfidfVectorizer(max_features=5000)

X = vectorizer.fit_transform(df["clean_text"])
y = df["target"].apply(lambda x: 1 if x == 4 else 0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

my_tweet = "I think you are a good doctor"
my_tweet_cleaned = clean_text(my_tweet)
my_tweet_tokens = word_tokenize(my_tweet_cleaned)
my_tweet_final = " ".join(my_tweet_tokens)
my_tweet_vector = vectorizer.transform([my_tweet_final])

print(my_tweet_vector)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)



y_pred_single = model.predict(my_tweet_vector)
y_pred = model.predict(X_test)
print(y_pred_single)
print(classification_report(y_test, y_pred))