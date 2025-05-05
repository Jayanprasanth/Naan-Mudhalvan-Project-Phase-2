import import pandas as pd
import nltk
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

nltk.download('stopwords')
from nltk.corpus import stopwords

# 1. Load the dataset (you can replace this with a CSV path)
def load_data():
    data = {
        'text': [
            'The economy is growing at a record pace!',
            'Scientists discover new element on Mars.',
            'Aliens landed in my backyard yesterday.',
            'Vaccines cause more harm than good.',
            'Government confirms UFO sightings are real.',
            'Earth is flat and NASA is lying to us.',
            'New study shows coffee boosts lifespan.',
            'Fake news spreads faster than real news.',
            'The moon landing was staged.',
            'Stock market hits all-time high.'
        ],
        'label': [0, 0, 1, 1, 1, 1, 0, 0, 1, 0]  # 0: Real, 1: Fake
    }
    return pd.DataFrame(data)

def clean_text(text):
    tokens = text.lower().translate(str.maketrans('', '', string.punctuation)).split()
    return ' '.join([word for word in tokens if word not in stopwords.words('english')])

def preprocess(df):
    df['cleaned'] = df['text'].apply(clean_text)
    return df

def vectorize_text(text_data):
    vectorizer = TfidfVectorizer()
    features = vectorizer.fit_transform(text_data)
    return features, vectorizer

def train_model(X, y):
    model = LogisticRegression()
    model.fit(X, y)
    return model


def evaluate(model, X_test, y_test):
    predictions = model.predict(X_test)
    print("\nAccuracy:", accuracy_score(y_test, predictions))
    print("Report:\n", classification_report(y_test, predictions))

def main():
    df = load_data()
    df = preprocess(df)

    X, vectorizer = vectorize_text(df['cleaned'])
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = train_model(X_train, y_train)
    evaluate(model, X_test, y_test)

    user_input = input("\nEnter a news headline to check if it's fake: ")
    cleaned_input = clean_text(user_input)
    vec_input = vectorizer.transform([cleaned_input])
    result = model.predict(vec_input)[0]
    print("Prediction:", "Fake News" if result == 1 else "Real News")

if __name__ == "__main__":
    main()
