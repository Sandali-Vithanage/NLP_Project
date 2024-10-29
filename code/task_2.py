import pandas as pd
import os
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
from nltk.corpus import stopwords
import string

nltk.download('vader_lexicon')
nltk.download('stopwords')

file_path = '../dataset/processed/multimodal_dataset.csv'
df = pd.read_csv(file_path, encoding='utf-8')

# Select only the text column
text_data = df['text'].dropna()  # Ensure no NaN values
print(text_data.head())

# Word Frequency Analysis with Stop Word, Punctuation, and Space Removal
all_text = ' '.join(text_data)

# Create a translation table to remove punctuation
translator = str.maketrans('', '', string.punctuation)

# Tokenize, remove punctuation, convert to lower case, and strip whitespace
words = [
    word.translate(translator).lower().strip()  # Remove punctuation, convert to lower case, and strip whitespace
    for word in all_text.split()
]

# Filter out stop words and empty strings
stop_words = set(stopwords.words('english'))
words = [word for word in words if word and word not in stop_words]  # Only include non-empty words

# Count word frequency
word_freq = Counter(words)

# Get the most common words
most_common_words = word_freq.most_common(20)

plt.figure(figsize=(10, 6))
plt.bar(*zip(*most_common_words))
plt.title('Top 20 Most Common Words (Excluding Stop Words, Punctuation, and Spaces)')
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.show()

# Create a word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(words))

plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of All Text (Excluding Stop Words, Punctuation, and Spaces)')
plt.show()

# Sentiment Analysis with VADER
sia = SentimentIntensityAnalyzer()

# Calculate sentiment scores
df['sentiment'] = text_data.apply(lambda x: sia.polarity_scores(x)['compound'])

plt.figure(figsize=(10, 6))
plt.hist(df['sentiment'], bins=30, edgecolor='black')
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment Score')
plt.ylabel('Frequency')
plt.show()

print("EDA Complete. Summary of Insights:")
print(f"Total text entries analyzed: {len(text_data)}")
print(f"Average sentiment score: {df['sentiment'].mean()}")
