from flask import Flask
import warnings
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from flask import render_template, jsonify, request

warnings.filterwarnings("ignore")

app = Flask(__name__)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)

@app.route("/", methods=["GET", "POST"])
def home():
    return render_template("tes.php", SM=SM)


SM = ["SMA", "MA", "SMK"]
jurusan = {
    "SMA": ["IPA", "IPS"],
    "MA": ["IPA", "IPS"],
    "SMK": [
        "Akuntansi",
        "Multimedia",
        "Rekayasa Perangkat Lunak",
        "Teknik Komputer Jaringan",
        "Teknik dan Bisnis Sepeda Motor",
        "Teknik Instalasi Tenaga Listrik",
        "Teknik Produksi Migas",
        "Teknik Kendaraan Ringan Otomotif",
        "Teknik Desain dan Informasi Bangunan",
        "Teknik Mekatronika",
    ],
}

# Pemanggilan data

data = pd.read_csv("static/data_final_final.csv", delimiter=",")

# Membagi data menjadi X dan Y

X = data.drop("jurusan", axis=1)
y = data[["jurusan"]]

# Membagi data menjadi train dan test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

# Modelling

knn = KNeighborsClassifier(3)
knn.fit(X_train, y_train)


@app.route("/get_dynamic_options", methods=["POST"])
def get_dynamic_options():
    selected_option = request.form["selectedOption"]
    options = jurusan.get(selected_option, [])
    return jsonify(options)


@app.route("/hasil", methods=["GET", "POST"])
def hasil():
    cekhasil = request.form.get("check")
    if cekhasil == "betul":
        jurusanlama = request.form.get("s2")
        nilai1 = request.form.get("nilai1")
        nilai2 = request.form.get("nilai2")
        prediksi = predict_model(jurusanlama, nilai1, nilai2)
        jurusanbaru = ""
        if prediksi == "1":
            jurusanbaru = "Jurusan Teknologi Industri"
        elif prediksi == "2":
            jurusanbaru = "Jurusan Teknologi Informasi"
        elif prediksi == "3":
            jurusanbaru = "Jurusan Administrasi Bisnis"
        return render_template("hasil_tes.php", prediction=jurusanbaru, label=prediksi)
    else:
        return render_template("tes.php", SM=SM)


def predict_model(a, b, c):
    jurusanlama = a
    jurusanlama2 = 0
    nilai1 = int(b) / 100
    nilai2 = int(c) / 100
    nilai3 = 0
    nilai4 = 0
    nilai5 = 0
    nilai6 = 0
    nilai7 = 0
    nilai8 = 0
    nilai9 = 0
    nilai10 = 0
    nilai11 = 0
    nilai12 = 0
    nilai13 = 0
    nilai14 = 0
    if jurusanlama == "Teknik dan Bisnis Sepeda Motor":
        nilai3 = 1
    elif jurusanlama == "Teknik Komputer Jaringan":
        nilai4 = 1
    elif jurusanlama == "IPA":
        nilai5 = 1
    elif jurusanlama == "IPS":
        nilai6 = 1
    elif jurusanlama == "Teknik Instalasi Tenaga Listrik":
        nilai7 = 1
    elif jurusanlama == "Akuntansi":
        nilai8 = 1
    elif jurusanlama == "Rekayasa Perangkat Lunak":
        nilai9 = 1
    elif jurusanlama == "Teknik Kendaraan Ringan Otomotif":
        nilai10 = 1
    elif jurusanlama == "Multimedia":
        nilai11 = 1
    elif jurusanlama == "Teknik Produksi Migas":
        nilai12 = 1
    elif jurusanlama == "Teknik Desain dan Informasi Bangunan":
        nilai13 = 1
    elif jurusanlama == "Teknik Mekatronika":
        nilai14 = 1
    if jurusanlama == "Teknik dan Bisnis Sepeda Motor":
        jurusanlama2 = 1
    elif jurusanlama == "Teknik Komputer Jaringan":
        jurusanlama2 = 2
    elif jurusanlama == "IPA":
        jurusanlama2 = 3
    elif jurusanlama == "IPS":
        jurusanlama2 = 4
    elif jurusanlama == "Teknik Instalasi Tenaga Listrik":
        jurusanlama2 = 5
    elif jurusanlama == "Akuntansi":
        jurusanlama2 = 6
    elif jurusanlama == "Rekayasa Perangkat Lunak":
        jurusanlama2 = 7
    elif jurusanlama == "Teknik Kendaraan Ringan Otomotif":
        jurusanlama2 = 8
    elif jurusanlama == "Multimedia":
        jurusanlama2 = 9
    elif jurusanlama == "Teknik Produksi Migas":
        jurusanlama2 = 10
    elif jurusanlama == "Teknik Desain dan Informasi Bangunan":
        jurusanlama2 = 11
    elif jurusanlama == "Teknik Mekatronika":
        jurusanlama2 = 12
    result = [[
        jurusanlama2,nilai1,nilai2,nilai3,nilai4,nilai5,nilai6,nilai7,nilai8,nilai9,nilai10,nilai11,nilai12,nilai13, nilai14,
        ]]
    hasil = knn.predict(result)
    return str(hasil[0])
