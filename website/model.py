from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import warnings
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

warnings.filterwarnings("ignore")

"""### Pemanggilan data"""

data = pd.read_csv("static\data_final.csv", delimiter=",")

"""## Transformasi Data

### Menghapus data dengan jurusan dan nilai yang tidak sesuai"""

data = data[data["Jurusan yang diambil saat SMA / SMK / MA"] != "SMK"]
data = data[data["Jurusan yang diambil saat SMA / SMK / MA"] != "Religi/Agama"]
data = data[data["Nilai Matematika"] >= 50]
data = data[data["Nilai Bahasa Inggris"] >= 50]
data = data[data["Nilai Kejurusan"] >= 50]
data = data.reset_index(drop=True)

"""### Menghapus kolom yang tidak digunakan"""

data = data.drop(labels=["Timestamp", "Username", "Nama Lengkap", "IPK"], axis=1)

"""### Merubah nama dan posisi dari data"""

data = data.rename(
    columns={
        "Program Studi": "jurusan",
        "Jurusan yang diambil saat SMA / SMK / MA": "jurusanlama",
        "Nilai Matematika": "nilai1",
        "Nilai Bahasa Inggris": "nilai2",
        "Nilai Kejurusan": "nilai3",
    }
)
data = data[["jurusanlama", "nilai1", "nilai2", "nilai3", "jurusan"]]

"""### Menggabungkan jurusan sama yang penulisannya berbeda"""

# Mengubah tulisan pada jurusan lama menjadi huruf kecil
data["jurusanlama"] = [x.lower() for x in data["jurusanlama"]]

# Mengubah penulisan yang salah & menggabungkan jurusan sama yang berbeda penulisan
data["jurusanlama"] = [x.replace("tbsm ", "tbsm") for x in data["jurusanlama"]]
data["jurusanlama"] = [x.replace("akuntansi keuangan lembaga ", "akt") for x in data["jurusanlama"]]
data["jurusanlama"] = [x.replace("akuntansi keuangan ", "akt") for x in data["jurusanlama"]]
data["jurusanlama"] = [x.replace("akuntansi ", "akt") for x in data["jurusanlama"]]
data["jurusanlama"] = [x.replace("akuntansi", "akt") for x in data["jurusanlama"]]
data["jurusanlama"] = [x.replace("titl untuk jurusan teknik instalasi tenaga listrik", "titl")for x in data["jurusanlama"]]
data["jurusanlama"] = [x.replace("teknik kendaraan ringan", "tkro") for x in data["jurusanlama"]]
data["jurusanlama"] = [x.replace("teknik produksi migas ", "pem") for x in data["jurusanlama"]]
data["jurusanlama"] = [x.replace("teknik desain dan informasi bangunan ", "dpib")for x in data["jurusanlama"]]
data["jurusanlama"] = [x.replace("teknik mekatronika ", "meka") for x in data["jurusanlama"]]
data["jurusanlama"] = [x.replace("jurusan multimedia", "mm") for x in data["jurusanlama"]]
data["jurusanlama"] = [x.replace("multimedia", "mm") for x in data["jurusanlama"]]

"""### Mengubah prodi menjadi jurusan"""

data["jurusan"] = [x.replace("Teknik Listrik", "jtin") for x in data["jurusan"]]
data["jurusan"] = [x.replace("Teknik Elektronika Telekomunikasi", "jtin") for x in data["jurusan"]]
data["jurusan"] = [x.replace("Teknologi Rekayasa Mekatronika", "jtin") for x in data["jurusan"]]
data["jurusan"] = [x.replace("Teknologi Rekayasa Sistem Elektronik", "jtin")for x in data["jurusan"]]
data["jurusan"] = [x.replace("Teknik Mesin", "jtin") for x in data["jurusan"]]
data["jurusan"] = [x.replace("Teknologi Rekayasa Jaringan Telekomunikasi", "jtin")for x in data["jurusan"]]
data["jurusan"] = [x.replace("Akuntansi Perpajakan", "akt") for x in data["jurusan"]]
data["jurusan"] = [x.replace("Sistem Informasi", "jti") for x in data["jurusan"]]
data["jurusan"] = [x.replace("Teknik Informatika", "jti") for x in data["jurusan"]]
data["jurusan"] = [x.replace("Teknologi Rekayasa Komputer", "jti") for x in data["jurusan"]]

"""### Menambahkan kolom baru berdasarkan jurusan lama"""

for i in data["jurusanlama"].unique():
  data[i] = 0

# Mengubah posisi kolom jurusan
temp_cols = data.columns.tolist()
index = data.columns.get_loc("jurusan")
new_cols = (
  temp_cols[0:index] + temp_cols[index + 1 :] + temp_cols[index : index + 1]
)
data = data[new_cols]

"""### mengubah value kategorikal menjadi angka"""

# Mengubah data jurusan lama
data["jurusanlama"] = [x.replace("tbsm", "1") for x in data["jurusanlama"]]
data["jurusanlama"] = [x.replace("tkj", "2") for x in data["jurusanlama"]]
data["jurusanlama"] = [x.replace("ipa", "3") for x in data["jurusanlama"]]
data["jurusanlama"] = [x.replace("ips", "4") for x in data["jurusanlama"]]
data["jurusanlama"] = [x.replace("titl", "5") for x in data["jurusanlama"]]
data["jurusanlama"] = [x.replace("akt", "6") for x in data["jurusanlama"]]
data["jurusanlama"] = [x.replace("rpl", "7") for x in data["jurusanlama"]]
data["jurusanlama"] = [x.replace("tkro", "8") for x in data["jurusanlama"]]
data["jurusanlama"] = [x.replace("mm", "9") for x in data["jurusanlama"]]
data["jurusanlama"] = [x.replace("pem", "10") for x in data["jurusanlama"]]
data["jurusanlama"] = [x.replace("dpib", "11") for x in data["jurusanlama"]]
data["jurusanlama"] = [x.replace("meka", "12") for x in data["jurusanlama"]]

# Mengubah data jurusan
data["jurusan"] = [x.replace("jtin", "1") for x in data["jurusan"]]
data["jurusan"] = [x.replace("jti", "2") for x in data["jurusan"]]
data["jurusan"] = [x.replace("akt", "3") for x in data["jurusan"]]

"""### Melakukan resampling untuk data yang tidak imbang"""

class_count_1, class_count_2, class_count_3 = data["jurusan"].value_counts()

class_1 = data[data["jurusan"] == "1"]
class_2 = data[data["jurusan"] == "2"]
class_3 = data[data["jurusan"] == "3"]

class_1_over = class_1.sample(class_count_1, replace=True)
class_3_over = class_3.sample(class_count_1, replace=True)

data = data[data["jurusan"] == "2"]
data = data.append(class_1_over, ignore_index=True)
data = data.append(class_3_over, ignore_index=True)

# Mengacak data menggunakan function sample()
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

"""### Melakukan scaling menggunakan minMaxScaler"""

scaler = MinMaxScaler()
data[["nilai1", "nilai2", "nilai3"]] = scaler.fit_transform(data[["nilai1", "nilai2", "nilai3"]])

"""### Memasukkan data menggunakan one-hot encoding"""

for i in data.index:
  if data.iloc[i]["jurusanlama"] == "1":
      data.loc[i, "tbsm"] = 1
  elif data.iloc[i]["jurusanlama"] == "2":
      data.loc[i, "tkj"] = 1
  elif data.iloc[i]["jurusanlama"] == "3":
      data.loc[i, "ipa"] = 1
  elif data.iloc[i]["jurusanlama"] == "4":
      data.loc[i, "ips"] = 1
  elif data.iloc[i]["jurusanlama"] == "5":
      data.loc[i, "titl"] = 1
  elif data.iloc[i]["jurusanlama"] == "6":
      data.loc[i, "akt"] = 1
  elif data.iloc[i]["jurusanlama"] == "7":
      data.loc[i, "rpl"] = 1
  elif data.iloc[i]["jurusanlama"] == "8":
      data.loc[i, "tkro"] = 1
  elif data.iloc[i]["jurusanlama"] == "9":
      data.loc[i, "mm"] = 1
  elif data.iloc[i]["jurusanlama"] == "10":
      data.loc[i, "pem"] = 1
  elif data.iloc[i]["jurusanlama"] == "11":
      data.loc[i, "dpib"] = 1
  elif data.iloc[i]["jurusanlama"] == "12":
      data.loc[i, "meka"] = 1

data = data.drop(labels=["nilai3"], axis=1)

"""### Membagi data menjadi X dan Y"""

X = data.drop("jurusan", axis=1)
y = data[["jurusan"]]

"""### Membagi data menjadi train dan test"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

"""### Menentukan nilai K yang paling bagus"""

test_scores = []
train_scores = []

for i in range(1, 10):
    knn_test = KNeighborsClassifier(i)
    knn_test.fit(X_train, y_train)
    train_scores.append(knn_test.score(X_train, y_train))
    test_scores.append(knn_test.score(X_test, y_test))

"""### Menampilkan skor tertinggi untuk train dan test"""

max_train_score = max(train_scores)
train_scores_ind = [i for i, v in enumerate(train_scores) if v == max_train_score]

max_test_score = max(test_scores)
test_scores_ind = [i for i, v in enumerate(test_scores) if v == max_test_score]
k = list(map(lambda x: x + 1, test_scores_ind))[0]

"""### Modelling"""

knn = KNeighborsClassifier(k)
knn.fit(X_train, y_train)
print(knn.score(X_train, y_train))
print(knn.score(X_test, y_test))

"""### Confusion Matrix"""

y_pred = knn.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix: \n", cm)
print("Classification report: \n", classification_report(y_test, y_pred))
