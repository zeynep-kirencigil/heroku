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

priority_model = joblib.load('priority_model.pkl')
category_model = joblib.load('category_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    case_id = data['case_id']

    # Belirli bir Case kaydını Id değerine göre sorgulama
    query = f"SELECT Id, Description FROM Case WHERE Id = '{case_id}'"
    cases = sf.query(query)

    # İlk kaydı alalım (zaten sadece bir kayıt dönecek)
    case = cases['records'][0] if cases['records'] else None

    if case:
        description = case['Description']
        descriptions = [description]
        description_tfidf = vectorizer.transform(descriptions)
        
        # Tahminler
        priority_prediction = priority_model.predict(description_tfidf)
        category_prediction = category_model.predict(description_tfidf)

        # Tahminleri Salesforce'a geri yazmak
        update_data = {
            'Prediction_Priority__c': priority_prediction[0],  # Örneğin, tahminlerin yazılacağı özel bir alan
            'Priority_Scenario__c': category_prediction[0]
        }
        sf.Case.update(case['Id'], update_data)

        return jsonify({'message': 'Case updated successfully.'})
    else:
        return jsonify({'message': 'No case found.'})

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
