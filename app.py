from flask import Flask, request, jsonify
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler


app = Flask(__name__)

data = pd.read_excel(f'./dataFinal.xlsx')
X = data.drop("Status Beasiswa BI", axis=1)
y = data["Status Beasiswa BI"]
scaler = MinMaxScaler()
X_Scaled = scaler.fit_transform(X)
model = KNeighborsClassifier(n_neighbors=9, metric='euclidean')
model.fit(X_Scaled, y)


@app.route('/')
def home():
    return "Hello, Flask!"


@app.route('/prediksi', methods=['POST'])
def prediksi():
    input_user = request.json
    df_input_user = pd.DataFrame([input_user])
    X_Scaled_input = scaler.fit_transform(df_input_user)
    y_pred = model.predict(X_Scaled_input)
    if (y_pred == 0):
        hasil = "Tidak menerima beasiswa"
    elif (y_pred == 1):
        hasil = "Menerima Beasiswa"
    return jsonify({"message": hasil})


if __name__ == '__main__':
    app.run(debug=True)
