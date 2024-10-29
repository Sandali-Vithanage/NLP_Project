import pandas as pd
import nltk
import string
from nltk.corpus import stopwords
from gensim import corpora
from gensim.models import LdaModel
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis
import webbrowser

nltk.download('stopwords')

file_path = '../dataset/processed/multimodal_dataset.csv'
df = pd.read_csv(file_path, encoding='utf-8')

text_data = df['text'].dropna()

# Preprocess the text data
def preprocess_text(text):
    translator = str.maketrans('', '', string.punctuation)
    stop_words = set(stopwords.words('english'))
    tokens = text.translate(translator).lower().split()
    tokens = [word for word in tokens if word not in stop_words]
    return tokens

processed_data = text_data.apply(preprocess_text)

# Create a dictionary and a document-term matrix
dictionary = corpora.Dictionary(processed_data)
corpus = [dictionary.doc2bow(text) for text in processed_data]

# Perform LDA
num_topics = 10  # You can optimize this later
lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10)

# Prepare and save the visualization
vis = gensimvis.prepare(lda_model, corpus, dictionary)
pyLDAvis.save_html(vis, 'lda_visualization.html')

# open the visualization in a web browser
webbrowser.open('lda_visualization.html')

# Print the topics
for idx, topic in lda_model.print_topics(-1):
    print(f"Topic {idx}: {topic}")
