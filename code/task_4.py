import pandas as pd
import nltk
import os
import string
from nltk.corpus import stopwords
from gensim import corpora
from gensim.models import LdaModel
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis
import webbrowser
import gensim
from gensim.models import LdaModel
from gensim.corpora import Dictionary
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('stopwords')

# Load dataset
file_path = '../dataset/processed/multimodal_dataset.csv'
df = pd.read_csv(file_path, encoding='utf-8')

# Extract text data
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
num_topics = 5
lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10)

# Prepare and save the LDA visualization
output_dir = os.path.abspath('../output')
output_file = 'Task_4_lda_visualization.html'
vis = gensimvis.prepare(lda_model, corpus, dictionary)
output_path = os.path.join(output_dir, output_file)
pyLDAvis.save_html(vis, output_path)

# Open the visualization in a web browser
webbrowser.open(f'file://{output_path}')

# Print the topics
for idx, topic in lda_model.print_topics(-1):
    print(f"Topic {idx}: {topic}")

# Get the top 10 words for each topic
topn = 10  # Limit to top N words per topic
topic_word_matrix = [lda_model.get_topic_terms(topicid=i, topn=topn) for i in range(num_topics)]

# Create a sparse matrix for cosine similarity calculation (using only the word probabilities)
topic_word_matrix_sparse = [[word[1] for word in topic_word_matrix[i]] for i in range(num_topics)]

# Calculate cosine similarities between topics
topic_similarities = cosine_similarity(topic_word_matrix_sparse)

# Print the cosine similarity matrix
print("Cosine Similarities:")
print(topic_similarities)

# Create a DataFrame for the cosine similarity matrix for better visualization
cosine_df = pd.DataFrame(topic_similarities,
                         index=[f"Topic {i+1}" for i in range(num_topics)],
                         columns=[f"Topic {i+1}" for i in range(num_topics)])

# Set up the figure for the heatmap
plt.figure(figsize=(8, 6))

# Create the heatmap using seaborn
sns.heatmap(cosine_df, annot=True, cmap='coolwarm', fmt='.2f', cbar=True,
            linewidths=0.5, square=True, annot_kws={"size": 12},
            cbar_kws={'label': 'Cosine Similarity'})

# Add title to the heatmap
plt.title("Cosine Similarity Heatmap between Topics", fontsize=15)

# Show the plot
plt.tight_layout()
plt.show()
