import cv2
import numpy as np #listeleme ve hesaplama
import pandas as pd #veri okuma ve veri işleme
import pickle # yaptığımız modeleri kaydedebileceğimiz ve daha sonrada kullanabileceğimiz 
import tensorflow as tf  #yapay zeka kütüphanesi sadece verileri okumak için kullanıcaz

from sklearn.ensemble import RandomForestClassifier #makine öğr. modellemeyi içnde bulunduran kütüphane, ensemle :birleşik manasına gelir
from sklearn.metrics import accuracy_score  #temel bir yapı yaptığımız için sadece bu başarı skoru kullanılır
import os

path = "karakterseti/"

siniflar = os.listdir(path)
tek_batch = 0 # tek veri kümesi eğiteceğimiz için
#kaç veri gireceğimizi bilmiyoruz

urls = [] #karakterseti/1/1.jpg
sinifs = []

for sinif in siniflar:
    resimler = os.listdir(path+sinif)
    for resim in resimler:
        urls.append(path+sinif+"/"+resim)
        sinifs.append(sinif)
        tek_batch+=1
df = pd.DataFrame({"adres":urls,"sinif":sinifs}) #serilerin bir arada bulunmasına dataframe denir

#tek hat halinde almak
#öznitelik çıkartma algoritması
def islem(img):
    
    yeni_boy = img.reshape((1600,5,5))
    orts = []
    for parca in yeni_boy:
        ort = np.mean(parca)
        orts.append(ort)
    orts = np.array(orts)
    orts = orts.reshape(1600,)
    return orts

def on_isle(img):
    return img/255

target_size=(200,200) #resim boyutları
batch_size=tek_batch   #her batch boyutunu tek batch yapmak içn

#datagenerator,veri çoğaltma,
train_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=on_isle)
#eğitim seti,sıra ile getirme,dataframeden işle,bağımlı değişken,bağımsız değişken
train_set = train_gen.flow_from_dataframe(df,x_col = "adres",y_col="sinif",
                                              target_size=target_size,
                                              color_mode = "grayscale",
                                              shuffle=True,#veri setini karıştırır
                                              class_mode='sparse', #sparse: int tip döndürür ,
                                              batch_size=batch_size)

#resimleri alma,train_y sonucunu alma,islem fonksiyonuna tabii tutma
images,train_y = next(train_set)
train_x = np.array(list(map(islem,images))).astype("float32")#elde ettiğimiz sonucu arraya dönüştürme
train_y= train_y.astype(int)#int64 depolar

print("Eğitiliyor")
rfc = RandomForestClassifier(n_estimators=10,criterion="entropy")#10 ağaçtan,entropy modu ile çalış uzaktan okuma modu
rfc.fit(train_x,train_y)#fit : eğit
pred = rfc.predict(train_x) #tahmin
acc = accuracy_score(pred,train_y)#ne kadar başarılı?

print("başarı:",acc)

#dosya adresi
dosya = "rfc_model.rfc"
#dosya uzantısı aç ve içine yazma işlemini yap, WriteByte şeklinde
pickle.dump(rfc,open(dosya,"wb"))







 




        




















