<!-- Proje-Resmi -->
<!-- ne kadar fotograf olmali, dosyalar nerede olmali zip icerisinde , zip i atma, ipynb deki bagimliliklari kendinden cek  vs  -->

## 👀 cv-model-training-pt Overview  1/3  
<h1 align="center">Computer Vision AI model training for rasberry pi 5</h1>  


En basta ana bir main klasorun ve icinde bir data.yaml dosyan olsun.Icinde kullanacagin tum classlarin idleri dursun.Cunku classlari aktarirken ana data.yaml icinde yazan class id'sine gore yeni eklenen datalarin labellarinin ilk karakteri degistirilerek eklenir.  

Dikkat edecegin sey ana data.yaml dosyandaki class adin ile ekleyecegin class datasinin data.yaml'daki adi ayni olmali  

WINDOWS
py -m pip install questionary PyYAML
py merge_yolo_datasets.py

<details>
<summary>Linux Packages</summary>
<details>
  <summary>Arch Packages</summary>
   sudo pacman -S python xdg-utils
   </details>

<details>
  <summary>Debian/Ubuntu Packages</summary>
   sudo apt update
   sudo apt install python3 python3-venv python3-pip xdg-utils
   </details>

<details>
  <summary>Fedora Packages</summary>
   sudo dnf install python3 python3-pip xdg-utils
   </details>
</details>
   


python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install questionary PyYAML
python3 merge_yolo_datasets.py

AFTER FIRST RUN
source .venv/bin/activate
python3 merge_yolo_datasets.py

Kod, çalıştırıldığı dizindeki dataset klasörlerini otomatik bulur. Beklenen yapı:

merge_yolo_datasets.py


bear/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
└── data.yaml

<details>
<summary>Veri yayma yolu:</summary>

Şunların çoğu train içinde olmalı:

Farklı ayı türleri ve görünümleri
Gece, gündüz ve IR görüntüleri
Yakın, uzak, küçük ve büyük ayılar
Kısmen ağaç arkasında kalan hayvanlar
Farklı mevsimler ve hava koşulları
Farklı kamera açıları
Boş orman gibi negatif görüntüler

Örneğin 3.000 ayı görüntüsünün yaklaşık 2.400’ü train içine konur.

Model val görüntülerinden öğrenmez. Her eğitim turundan sonra bunlarda ölçüm yapılır.

val, şu kararları etkiler:

En iyi checkpoint’in seçilmesi
Early stopping
Precision, recall ve mAP ölçümleri
Modelin ezberlemeye başlayıp başlamadığının görülmesi
Confidence threshold gibi ayarların belirlenmesi

3.000 görüntüde yaklaşık 300 görüntü val içine konur.

val içinde de mutlaka ayı bulunmalıdır. Ancak bunlar train içindeki ayı görüntülerinin kopyaları veya komşu video kareleri olmamalıdır.

Test

test, model tamamen bittikten sonra yalnızca son değerlendirme için kullanılır.

Test sonucuna bakarak sürekli eğitim ayarı değiştirirsen test artık tarafsız olmaz ve pratikte ikinci bir val setine dönüşür.

3.000 görüntüde yaklaşık 300 görüntü test içine konur.
</details>



Eksik train, valid, test, images ve labels klasörlerini gerektiğinde oluşturur.

Ana menü
1 → Gerekli class’ları filtrele
2 → Datasetleri ana dataset içinde birleştir
3 → Ana dataseti 80/10/10 yeniden bölüştür(coklu class i sececeksin)
5 → Ana dataseti kontrol et
4 → ZIP’e uygun klasör düzenine çevir
6 → ZIP oluştur

1. Class filtreleme

Bir veya birden fazla class seçebilirsin.

İki çalışma biçimi bulunuyor:

Seçilen class kutuları korunur.
İstenmeyen kutular label’dan kaldırılır.
Hedef kutusu kalmayan resim negatif görüntü olarak tutulur ve label’ı boş bırakılır.

Veya:

Yalnızca seçilen class bulunan resimler tutulur.
Aynı görüntüde seçilmeyen başka bir class varsa görüntü kullanılmaz.
Kullanılmayan resim ve label çiftleri:
_filtered_out/

klasörüne taşınabilir veya açık onay verdikten sonra silinebilir.

Kalan sınıfların ID’leri 0’dan başlayarak yeniden sıralanır ve data.yaml güncellenir.

2. Dataset birleştirme

Önerilen seçenek sınıfları data.yaml içindeki isimlere göre eşleştirir.

Örneğin kaynak dataset:

names:
  0: polar_bear
  1: black_bear
  2: brown_bear

Ana dataset:

names:
  0: bear
  1: boar
  2: deer
  3: wolf
  4: cow
  5: person
  6: dog

Birleştirme sonucu:

names:
  0: bear
  1: boar
  2: deer
  3: wolf
  4: cow
  5: person
  6: dog
  7: polar_bear
  8: black_bear
  9: brown_bear

Kaynak label’lardaki:

0 → 7
1 → 8
2 → 9

dönüşümü kopyalama sırasında otomatik yapılır. Kaynak dosyalar değiştirilmez.

İki dataset içinde aynı isimli class varsa yeni class oluşturulmaz. Örneğin kaynakta ve hedefte brown_bear varsa hedefteki mevcut ID kullanılır.

Dosya adı çakışırsa hem resim hem label birlikte yeniden adlandırılır:

bear_dataset__image001.jpg
bear_dataset__image001.txt
NEW_CLASS_ID boş bırakma konusu

Çok sınıflı dataseti değişiklik yapmadan kopyalamak ancak iki datasetin ID anlamları tamamen aynıysa güvenlidir:

Kaynak 0 = bear
Hedef  0 = bear

Kaynakta 0=polar_bear, hedefte 0=bear ise doğrudan kopyalamak yanlıştır.

Yeni kodda bunun yerine data.yaml adlarına göre eşleştir seçeneğini kullanmalısın. Kod hedef ID’leri otomatik belirler.

3. Oranlı yeniden bölüştürme

Kod önce şunları sorar:

Kaç class/dataset klasörünüz var?
Tek class içeren klasörleri seçin
Çok class içeren klasörleri seçin
Train yüzdesi
Valid yüzdesi
Test yüzdesi

Klasör seçiminde:

Yukarı/aşağı: hareket
Space: seçme/kaldırma
Enter: onaylama

Örneğin:

train: %70
valid: %20
test:  %10

valid veya test klasörü bulunmasa bile oluşturulur. %0 verdiğin split de boş olarak oluşturulur.

Tek sınıflı datasetlerde resim sayısı temel alınır.

Çok sınıflı datasetlerde label içerikleri okunur ve sınıf dağılımı yaklaşık korunur. Örneğin mevcut oran:

polar_bear : black_bear : brown_bear
3          : 5          : 8

ise her sınıftan zorla eşit sayıda görüntü seçilmez. Yaklaşık 3:5:8 dağılımı korunur. Nadir sınıf içeren görüntüler öncelikli yerleştirilir.

Aynı görüntüde üç class varsa bu görüntü üç class’ın hedefine de katkı sağlar. Bu nedenle çok etiketli datasetlerde her sınıf hedefini matematiksel olarak tam tutturmak her zaman mümkün değildir; toplam train/valid/test resim sayısı ise belirlenen orana tam olarak dağıtılır.

Eksik label denetimi

Bir resmin aynı isimli label’ı bulunamazsa:

image001.jpg isimli dosyaya ait label bulunamadı


(1) Resmi aç
(2) Devam et / şimdilik atla
(3) Programı sonlandır

Resmi açtıktan sonra:

(1) Resmi sil
(2) Boş label oluştur
(3) Hiçbir şey yapma ve programı bitir

Boş label oluşturma seçeneği negatif görüntüler içindir.

Resmi bulunmayan bağımsız .txt label dosyaları otomatik değiştirilmez; program hata vererek konumlarını gösterir.

Son doğrulama

Her işlemden sonra kod şunları kontrol eder:

Her resmin aynı isimli .txt dosyası var mı?
Her label’ın karşılık gelen resmi var mı?
Aynı kök ada sahip birden fazla resim var mı?
Class ID negatif mi?
Class ID data.yaml içinde tanımlı mı?
Koordinatlar 0–1 aralığında mı?
Detection veya polygon satırı geçerli mi?
Boş negatif label’lar korunmuş mu?
Split başına resim ve label sayıları eşit mi?
Her sınıfta kaç kutu var?

Örnek çıktı:

train: resim=7000, label=7000, kutu=8450, negatif=900 [OK]
valid: resim=2000, label=2000, kutu=2410, negatif=250 [OK]
test : resim=1000, label=1000, kutu=1195, negatif=125 [OK]

Resim sayısıyla kutu sayısının eşit olması gerekmez. Bir resimde sıfır, bir veya birden fazla kutu bulunabilir. Eşit olması gereken, resim sayısı ile label dosyası sayısıdır.

Yedekleme

Filtreleme, birleştirme ve bölüştürme öncesinde otomatik olarak label ve data.yaml yedeği oluşturulur:

bear_filter_backup_....zip
bear_merge_backup_....zip
bear_split_backup_....zip

Bölüştürme yedeği görüntülerin tamamını içermez; büyük datasetlerde yüzlerce GB’lık gereksiz yedek oluşmaması için yalnızca label ve YAML tutulur.

Önemli sınırlama: Kod sınıf oranlarını korur fakat aynı videodan veya aynı fotokapan olayından gelen benzer kareleri otomatik olarak tanıyamaz. Olay sızıntısını engellemek için bu karelerin önceden aynı olay grubu altında tutulması gerekir.
