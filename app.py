from flask import Flask, request, jsonify
from simple_salesforce import Salesforce
import joblib
import requests

app = Flask(__name__)

# Salesforce bağlantı bilgileri
consumer_key = '3MVG9buXpECUESHh30pZX3pDyGhthnzMbc.X1wGQgEQjPFcWyRq_76ZAg7DTOBChAOxcXyWuNmLVDTf6TW6BS'
consumer_secret = '4E96C47C752E2B253DFAE9FE9BADD9F72E9D2DBB885CB5518FCB3D802F0C5953'
username = 'integrationapi@flypgs.com.partial'
password = 'Pegasus2024*'
security_token = 'Zu8570EJKA7aUJHZXti5qQMex'

# Salesforce OAuth URL
auth_url = 'https://test.salesforce.com/services/oauth2/token'

# OAuth Token Request
data = {
    'grant_type': 'password',
    'client_id': consumer_key,
    'client_secret': consumer_secret,
    'username': username,
    'password': password + security_token
}

response = requests.post(auth_url, data=data)
response_data = response.json()

# Erişim belirteci (access token) ve instance URL'si
access_token = response_data['access_token']
instance_url = response_data['instance_url']

# Salesforce ile bağlantı kurma
sf = Salesforce(instance_url=instance_url, session_id=access_token)

# Modelleri yükleyin
import joblib

# Modelinizi yükleyin (örnek bir model yolu)
from google.colab import drive

drive.mount('/content/drive')

priority_model = joblib.load('/content/drive/MyDrive/NLP/priority_model.pkl')
category_model = joblib.load('/content/drive/MyDrive/NLP/category_model.pkl')
vectorizer = joblib.load('/content/drive/MyDrive/NLP/tfidf_vectorizer.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    # JSON formatında gelen Case ID'yi alın
    # data = request.json
    # case_id = data['case_id']

    # Belirli bir Case kaydını Id değerine göre sorgulama
    query = "SELECT Id, Description FROM Case WHERE Id = '5003L000000hqTtQAI'"
    cases = sf.query(query)

    # İlk kaydı alalım (zaten sadece bir kayıt dönecek)
    case = cases['records'][0] if cases['records'] else None

    if case:
        description = case['Description']

        # Description alanı üzerinden tahmin yapmak
        descriptions = [case['Description'] for case in cases['records']]
        description_tfidf = vectorizer.transform(descriptions)
        priority_prediction = priority_model.predict(description_tfidf)
        category_prediction = category_model.predict(description_tfidf)

        # Tahminleri Salesforce'a geri yazmak
           update_data = {
            'Prediction_Priority__c': priority_prediction[0],  # Örneğin, tahminlerin yazılacağı özel bir alan
            'Priority_Scenario__c': category_prediction[0]
        }
        sf.Case.update(case['Id'], update_data)

        return jsonify({
            'case_id': case['Id'],
            'priority_prediction': priority_prediction[0],
            'category_prediction': category_prediction[0]
        })
    else:
        return jsonify({'error': 'No case found'}), 404

if __name__ == '__main__':
    app.run(debug=True)
