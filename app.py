from flask import Flask, request, jsonify
import zipfile
import joblib
import re
import string
#from nltk.corpus import stopwords
#from nltk.tokenize import word_tokenize
#from nltk.stem import WordNetLemmatizer
import nltk

# NLTK veri yolunu ayarla
nltk_data_path = os.path.join(os.path.dirname(__file__), 'nltk_data')
if not os.path.exists(nltk_data_path):
    with zipfile.ZipFile('nltk_data.zip', 'r') as zip_ref:
        zip_ref.extractall(nltk_data_path)
nltk.data.path.append(nltk_data_path)

app = Flask(__name__)

# Türkçe durak kelimeler listesi
turkish_stop_words = [
    'acaba', 'ama', 'ancak', 'arada', 'aslında', 'az', 'bana', 'bazen', 'bazı', 'belki', 'ben', 'biri', 'birkaç',
    'birçok', 'bu', 'buna', 'bunda', 'bunu', 'bunun', 'da', 'daha', 'de', 'defa', 'diye', 'dolayı', 'fakat',
    'gibi', 'hem', 'hep', 'hepsi', 'her', 'hiç', 'için', 'ise', 'kez', 'ki', 'kim', 'mı', 'mu', 'mü', 'nasıl',
    'ne', 'neden', 'nerde', 'nerede', 'nereye', 'niye', 'o', 'sadece', 'sanki', 'şayet', 'şey', 'siz', 'şu',
    'tüm', 've', 'veya', 'ya', 'yani'
]
# İngilizce durak kelimeler listesi
english_stop_words = set(stopwords.words('english'))

# Tüm durak kelimeler
stop_words = english_stop_words.union(turkish_stop_words)

# Lemmatizer
lemmatizer = WordNetLemmatizer()

# Veri temizleme fonksiyonu
def clean_text(Description, stop_words, lemmatizer):
    Description = Description.lower()
    Description = Description.translate(str.maketrans('', '', string.punctuation))
    Description = re.sub(r'\d+', '', Description)
    word_tokens = word_tokenize(Description)
    filtered_words = [word for word in word_tokens if word not in stop_words]
    lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]
    cleaned_text = ' '.join(lemmatized_words)
    return cleaned_text

# Modelleri ve vektörleştiriciyi yükleme
from google.colab import drive

# Loading the dataset
drive.mount('/content/drive')

priority_model = joblib.load('/content/drive/MyDrive/NLP/priority_model.pkl')
category_model = joblib.load('/content/drive/MyDrive/NLP/category_model.pkl')
vectorizer = joblib.load('/content/drive/MyDrive/NLP/tfidf_vectorizer.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    description = data['description']
    description_tfidf = vectorizer.transform([description])

    priority_prediction = priority_model.predict(description_tfidf)[0]
    category_prediction = category_model.predict(description_tfidf)[0]

    response = {
        'priority': priority_prediction,
        'category': category_prediction
    }

    return jsonify(response)

if __name__ == "__main__":
    app.run(port=5000, debug=True)