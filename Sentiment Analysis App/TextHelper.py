import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Download required resources if not already available
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

def prepreoccessing(text):
    text = text.lower()
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = word_tokenize(text)
    stop_word = set(stopwords.words('english'))
    filtered = [i for i in text if i not in stop_word]
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(i) for i in filtered]
    return ' '.join(stemmed_tokens)
