import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import nltk
from nltk.corpus import stopwords
import re
import warnings

warnings.filterwarnings("ignore")

# Download stopwords if not already present
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load CSV file
df = pd.read_csv("C:/Users/user/Desktop/ALL INTERNSHIPS (IMP)/8) INTERGRAVITY (1april-1june)/ANSWERS/movie.csv")

# Drop rows with missing values in review or sentiment
df = df.dropna(subset=['review', 'sentiment'])

# Clean review text
def clean_text(text):
    text = re.sub(r"[^a-zA-Z]", " ", str(text))  # Convert to string just in case
    text = text.lower().split()
    text = [word for word in text if word not in stop_words]
    return " ".join(text)

df['cleaned_review'] = df['review'].apply(clean_text)

# Encode sentiment labels (make sure they're only positive or negative)
df = df[df['sentiment'].isin(['positive', 'negative'])]
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

# Ensure no NaN in sentiment
df = df.dropna(subset=['sentiment'])

# Features and labels
X = df['cleaned_review']
y = df['sentiment']

# Vectorize
vectorizer = TfidfVectorizer()
X_vect = vectorizer.fit_transform(X)

# Ensure enough samples before stratifying
if y.nunique() == 2 and len(y) >= 10:
    X_train, X_test, y_train, y_test = train_test_split(
        X_vect, y, test_size=0.2, stratify=y, random_state=42
    )
else:
    print("⚠️ Not enough samples to stratify. Using random split without stratify.")
    X_train, X_test, y_train, y_test = train_test_split(
        X_vect, y, test_size=0.2, random_state=42
    )

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Report
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, zero_division=0))