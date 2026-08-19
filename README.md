<!-- Proje-Resmi -->
<!-- ne kadar fotograf olmali, dosyalar nerede olmali zip icerisinde , zip i atma, ipynb deki bagimliliklari kendinden cek  vs  -->

## 👀 cv-model-training-pt Overview  1/3  
<h1 align="center">Computer Vision AI model training for rasberry pi 5 and interactive command-line application for filtering, merging, splitting, validating, reorganizing, and packaging YOLO datasets</h1>  


## 🚀 Features
- [x] Create empty YOLO label files for background and negative images
- [x] Filter datasets using one or multiple selected classes
- [x] Keep selected classes or process all unselected classes
- [x] Copy filtered image-label pairs into a separate dataset
- [x] Delete selected image-label pairs after explicit confirmation
- [x] Limit the number of images separately for each class
- [x] Preserve the existing train, val, and test distribution during filtering
- [x] Choose whether images containing unselected objects are allowed
- [x] Automatically remove unwanted bounding boxes from copied labels
- [x] Merge multiple classes into a single class
- [x] Convert all label class IDs to a user-selected class
- [x] Automatically detect and display classes from data.yaml
- [x] Reindex remaining class IDs starting from 0
- [x] Automatically update data.yaml after class changes
- [x] Merge multiple YOLO datasets into one main dataset
- [x] Match classes automatically by their names in data.yaml
- [x] Remap source class IDs to the correct destination IDs
- [x] Preserve existing destination classes while adding new classes
- [x] Prevent class ID conflicts that could corrupt labels
- [x] Rename image-label pairs together when filenames conflict
- [x] Redistribute datasets using customizable train, val, and test percentages
- [x] Use an 80/10/10 train, val, and test split by default
- [x] Support datasets containing either single or multiple classes
- [x] Approximately preserve class distribution in multi-class datasets
- [x] Prioritize images containing rare classes during dataset splitting
- [x] Preserve empty labels used for negative images
- [x] Detect missing image-label pairs and offer repair options
- [x] Detect orphan label files that do not have matching images
- [x] Convert datasets to the images/split + labels/split directory structure
- [x] Support legacy valid directories and convert them to val
- [x] Validate YOLO detection and segmentation/polygon labels
- [x] Validate class IDs, coordinates, image-label counts, and duplicate filenames
- [x] Report image, label, bounding-box, negative-image, and per-class statistics
- [x] Automatically back up labels and data.yaml before modifying datasets
- [x] Display progress information during long-running operations
- [x] Browse and select directories across different disks and locations
- [x] Use the same interactive directory browser for every source and destination selection
- [x] Create ZIP archives containing only data.yaml, images, and labels
- [x] Preserve empty split directories inside generated ZIP archives
- [x] Verify ZIP integrity after archive creation
- [x] Windows and Linux support


En basta ana bir main klasorun ve icinde bir data.yaml dosyan olsun.Icinde kullanacagin tum classlarin idleri dursun.Cunku classlari aktarirken ana data.yaml icinde yazan class id'sine gore yeni eklenen datalarin labellarinin ilk karakteri degistirilerek eklenir.  

Dikkat edecegin sey ana data.yaml dosyandaki class adin ile ekleyecegin class datasinin data.yaml'daki adi ayni olmali  

En iyi dataset icinde yeterli ve 80 10 10 oranina yakin bir train/test/val degilimina sahip, ardi ardina cekilmis olmayan goruntulerden olusan bir datasettir  


<details>
<summary>Veri yayma yolu:</summary>

TRAIN
Şunların çoğu train içinde olmalı:

Farklı ayı türleri ve görünümleri
Gece, gündüz ve IR görüntüleri
Yakın, uzak, küçük ve büyük ayılar
Kısmen ağaç arkasında kalan hayvanlar
Farklı mevsimler ve hava koşulları
Farklı kamera açıları
Boş orman gibi negatif görüntüler

Örneğin 3.000 ayı görüntüsünün yaklaşık 2.400’ü train içine konur.

VAL
Model val görüntülerinden öğrenmez. Her eğitim turundan sonra bunlarda ölçüm yapılır.

val, şu kararları etkiler:

En iyi checkpoint’in seçilmesi
Early stopping
Precision, recall ve mAP ölçümleri
Modelin ezberlemeye başlayıp başlamadığının görülmesi
Confidence threshold gibi ayarların belirlenmesi

3.000 görüntüde yaklaşık 300 görüntü val içine konur.

val içinde de mutlaka ayı bulunmalıdır. Ancak bunlar train içindeki ayı görüntülerinin kopyaları veya komşu video kareleri olmamalıdır.

TEST
test, model tamamen bittikten sonra yalnızca son değerlendirme için kullanılır.

Test sonucuna bakarak sürekli eğitim ayarı değiştirirsen test artık tarafsız olmaz ve pratikte ikinci bir val setine dönüşür.

3.000 görüntüde yaklaşık 300 görüntü test içine konur.

NEGATIF GORUNTULER(modelde kullanilan classlar harici olan arkaplandir orman sokak cadde gibi)
Minimum:  %10
Önerilen: %20
Üst sınır: %25–30
</details>


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





Eksik train, valid, test, images ve labels klasörlerini gerektiğinde oluşturur.

Ana menü
(1) Fotograflar icin bos/negatif label olustur
(2) Class'lari filtrele/azalt
(3) Datasetleri ana dataset icine birlestir
(4) Train/val/test oranlariyla yeniden bolustur
(5) Ana dataseti images/split + labels/split duzenine cevir
(6) Datasetleri yalnizca kontrol et
(7) Ana dataseti ZIP dosyasi yap
(0) Cikis

# YOLO Dataset Yönetim Aracı

Bu araç YOLO formatındaki datasetlerde negatif label oluşturma, class filtreleme, dataset birleştirme, train/val/test bölüştürme, klasör düzenini dönüştürme, doğrulama ve ZIP oluşturma işlemlerini yapar.

## Ana menü

(1) Fotoğraflar için boş/negatif label oluştur
(2) Class'ları filtrele/azalt
(3) Datasetleri ana dataset içine birleştir
(4) Train/val/test oranlarıyla yeniden bölüştür
(5) Ana dataseti images/split + labels/split düzenine çevir
(6) Datasetleri yalnızca kontrol et
(7) Ana dataseti ZIP dosyası yap
(0) Çıkış

## Klasör seçme sistemi

Programdaki bütün kaynak ve hedef klasör seçimlerinde aynı gezilebilir klasör seçici kullanılır.

Yukarı/aşağı: Listede hareket et
Sağ ok veya Space: Seçili klasöre gir
Sol ok veya Esc: Bir üst klasöre çık
Enter: Bulunduğun klasörü seç
Ctrl+C: İşlemi iptal et

Dataset, kaynak, hedef ve ZIP kayıt klasörü seçimlerinde farklı diskler ve dizinler arasında gezilebilir.

Windows'ta `C:\`, `D:\` gibi kullanılabilir diskler listelenir.

Linux'ta dosya sistemindeki erişilebilir dizinler arasında gezilebilir.

## Desteklenen dataset düzeni

Düzenleme işlemlerinde kullanılan çalışma düzeni:

```text
dataset/  
├── data.yaml  
├── train/  
│   ├── images/  
│   └── labels/  
├── val/  
│   ├── images/  
│   └── labels/  
└── test/  
    ├── images/  
    └── labels/  
```

Program eski datasetlerde bulunan `valid` klasörünü okuyabilir ancak oluşturduğu standart klasörün adı `val` olur.

## 1. Fotoğraflar için boş/negatif label oluştur

Bu seçenek, içerisinde hedef obje bulunmayan arka plan görüntüleri için boş YOLO label dosyaları oluşturur.

Önce fotoğrafların bulunduğu gerçek `images` klasörü seçilir.

Program aynı seviyedeki `labels` klasörünü otomatik olarak belirler.

Örneğin:

```text
train/  
├── images/  
└── labels/  
```

Her resim için aynı kök ada sahip boş bir `.txt` dosyası oluşturulur.

Örneğin:

```text
forest001.jpg → forest001.txt  
forest002.png → forest002.txt  
```

Oluşturulan `.txt` dosyaları tamamen boş olur. Bu dosyalar görüntünün negatif, yani hedef obje içermeyen bir görüntü olduğunu belirtir.

`labels` klasörü yoksa otomatik oluşturulur.

Klasörün içinde daha önceden oluşturulmuş label dosyaları varsa program şu şekilde açık onay ister:

```text
Bu labels klasörü dolu. Mevcut label dosyalarının üzerine yazmak ister misiniz?  
```

Onay verilirse işlemden önce mevcut `.txt` dosyalarının yedeği oluşturulur:

```text
labels_empty_backup_....zip  
```

Yalnızca seçilen `images` klasöründeki resimlerle aynı ada sahip label dosyaları boşaltılır. İlgisiz label dosyaları değiştirilmez.

Aynı kök ada sahip birden fazla resim bulunursa işlem durdurulur. Örneğin aynı klasörde hem `forest01.jpg` hem de `forest01.png` bulunması tek bir `forest01.txt` label’ına karşılık geleceği için güvenli değildir.

İşlem sırasında ilerleme durumu gösterilir. İşlem sonunda her resim için boş label oluşturulup oluşturulmadığı kontrol edilir.

Bu özellik yalnızca gerçekten boş görüntülerde kullanılmalıdır. Görüntüde hayvan bulunduğu hâlde boş label oluşturulursa model bu hayvanı arka plan olarak öğrenebilir.

Negatif görüntü oranı sabit bir kural değildir. Başlangıç için toplam train+val+test görüntülerinin yaklaşık `%10–20` kadarı negatif görüntü olarak kullanılabilir. Fotokapanın gerçek kullanım ortamına göre bu oran doğrulama sonuçlarıyla ayarlanmalıdır.

## 2. Class'ları filtrele/azalt

Bu menü üç farklı işlem içerir:

```text
(1) Class'ları tut veya kaldır  
(2) Birden fazla class'ı tek class'ta birleştir  
(3) Bütün label class ID'lerini tek bir değere çevir  
```

### 2.1. Class'ları tut veya kaldır

Bir veya birden fazla class seçilebilir.

İşlem sırası şöyledir:

```text
Class'ları filtrelenecek dataset  
Filtrelenmesini istediğiniz class'ları seçin  
Nasıl resimler filtrelenmeli?  
Yapmak istediğiniz işlem  
Hangi tür için geçerli olsun?  
Her class için kaç tane filtrelensin?  
```

Resimlerin nasıl filtreleneceği için iki yöntem bulunur.

Birinci yöntem:

```text
Yeni data.yaml'da class'i bulunmayacak bir obje, modeli seçilen class için eğiten resimde bulunabilir (önerilen)  
```

Bu yöntemde seçilen class kutuları korunur.

Seçilmeyen class kutuları oluşturulan label dosyasından kaldırılır.

Hedef kutusu kalmayan bir görüntü negatif görüntü olarak tutulabilir ve label dosyası boş bırakılır.

Ancak resimde görünen fakat label’dan kaldırılan objeler model açısından arka plan olarak değerlendirilebilir. Bu nedenle oluşturulan dataset görsel olarak kontrol edilmelidir.

İkinci yöntem:

```text
Sadece seçtiğimiz class'ların box'ının bulunduğu resimler kullanılsın  
```

Bu yöntemde resimde seçilmeyen herhangi bir class kutusu bulunuyorsa görüntü kullanılmaz.

Bu seçenek daha temiz bir class dataseti oluşturur ancak kullanılabilecek resim sayısını azaltabilir.

İşlemin hangi class’lara uygulanacağı seçilebilir:

```text
Seçilen class'lar  
Seçilmeyen class'lar  
```

Program etkili olan her class için ayrı ayrı kaç görüntü istendiğini sorar.

```text
0 = Uygun olan bütün görüntüleri kullan  
```

Örneğin filtreleme sonrasında bir class için mevcut dağılım şu şekildeyse:

```text
train: 10  
val:    4  
test:   3  
```

Toplam 8 görüntü istendiğinde program görüntüleri split oranını koruyarak yaklaşık şu şekilde seçer:

```text
train: 5  
val:   2  
test:  1  
```

Benzer şekilde mevcut dağılım `10000/4000/3000` ve istenen sayı 800 ise yaklaşık `471/188/141` görüntü alınır.

Hangi görüntülerin seçileceği sabit seed kullanılarak belirlenir. Aynı dataset ve aynı ayarlarla işlem tekrarlandığında mümkün olduğunca aynı sonuç elde edilir.

Bir görüntü birden fazla seçilen class içeriyorsa aynı görüntü birden fazla kotaya katkı sağlayabilir ancak yalnızca bir defa kopyalanır veya silinir. Bu nedenle çok class’lı seçimlerde toplam benzersiz görüntü sayısı class başına girilen sayıların toplamından farklı olabilir.

Yapılabilecek işlemler:

```text
Başka dosyaya kopyalama  
Silmek  
```

#### Başka dosyaya kopyalama

Önce kopyalamanın yapılacağı dizin seçilir.

Ardından program şunu sorar:

```text
Yeni hedef dataset klasörünün adı:  
```

Bu alan boş bırakılırsa seçilen dizinin kendisi hedef dataset olarak kullanılır. Gerekli `train`, `val`, `test` ve `data.yaml` dosya ve klasörleri doğrudan bu dizinde oluşturulur veya mevcut olanlar kullanılır.

Bir isim yazılırsa seçilen dizinin altında bu isimde yeni bir dataset klasörü oluşturulur veya aynı isimdeki mevcut dataset kullanılır.

Filtrelenmiş görüntüler mevcut split’leri korunarak kopyalanır:

```text
Kaynak train → Hedef train  
Kaynak val   → Hedef val  
Kaynak test  → Hedef test  
```

Görüntüyle birlikte aynı ada sahip label dosyası da kopyalanır.

Hedefte `data.yaml` bulunuyorsa kaynak ve hedef class isimleri karşılaştırılır.

Aynı class ismi hedefte farklı bir ID ile bulunuyorsa program hedefteki ID’yi kullanmayı teklif eder.

Örneğin:

```text
Kaynak: 0 = bear  
Hedef:  2 = bear  
```

Bu durumda kopyalanan label dosyalarındaki `bear` ID’si `0 → 2` olarak değiştirilir.

Kullanıcı bu eşleştirmeyi reddederse datasetin bozulmasını önlemek için kopyalama işlemi iptal edilir.

Kaynak class ID’si hedefte başka bir class tarafından kullanılıyorsa program en yakın boş ve güvenli ID’yi önerir.

Örneğin:

```text
Hedef:  
0 = bear  
1 = boar  
  
Kaynak:  
1 = wolf  
```

Bu durumda program filtrelenmiş `wolf` label değerlerini `2` yapmak isteyip istemediğinizi sorar.

Hedef datasetin mevcut class’ları ve label dosyaları korunur. Yalnızca yeni kopyalanan label dosyalarının class ID’leri gerekli olduğunda dönüştürülür.

Hedefte label dosyaları bulunduğu hâlde `data.yaml` yoksa mevcut ID’lerin hangi class anlamına geldiği bilinemez. Datasetin bozulmasını önlemek için işlem durdurulur.

Dosya adı çakışırsa görüntü ve label birlikte yeniden adlandırılır:

```text
bear_dataset__image001.jpg  
bear_dataset__image001.txt  
```

Kopyalama öncesinde hedef datasetin mevcut label ve YAML dosyaları yedeklenir.

#### Silmek

Bu işlem uygun bulunan resim ve label çiftlerini datasetten kaldırır.

Bir görüntüde birden fazla class varsa program diğer class’ların da bu görüntüyle birlikte kaybedilebileceğini belirtir.

Silme işlemi açık onay alınmadan başlatılmaz.

Oluşturulan otomatik yedekler çoğunlukla label ve YAML dosyalarını içerir. Silinen büyük görüntü dosyaları bu metadata yedeğinden geri getirilemez.

### 2.2. Birden fazla class'ı tek class'ta birleştir

Bu seçenek aynı dataset içerisindeki birden fazla class’ı tek bir class altında birleştirir.

Örneğin:

```text
polar_bear  
black_bear  
brown_bear  
```

class’ları `bear` class’ı altında birleştirilebilir.

Önce birleştirilecek en az iki class seçilir.

Ardından seçilen class’lardan hangisinin hedef class olarak kullanılacağı belirlenir.

Seçilen bütün class ID’leri hedef class ID’sine dönüştürülür.

Aynı görüntüde birden fazla kutu varsa kutular silinmez. Her kutu korunur ancak class ID’leri aynı class’ı gösterecek şekilde değiştirilir.

Birleştirme sonrasında kalan class ID’leri `0` değerinden başlayarak yeniden sıralanır ve `data.yaml` güncellenir.

Boş negatif label dosyaları boş olarak korunur.

İşlem öncesinde label ve `data.yaml` yedeği oluşturulur. İşlem sırasında ilerleme durumu gösterilir ve sonuç doğrulanır.

### 2.3. Bütün label class ID'lerini tek bir değere çevir

Bu seçenek bütün dolu label satırlarının ilk değerini seçilen tek bir class ID’sine dönüştürür.

`data.yaml` içinde class isimleri bulunursa program tespit ettiği class’ları listeler ve hedef class bu listeden seçilir. Böylece class adının birebir elle yazılması gerekmez.

Program `data.yaml` veya class listesi bulamazsa şu durumu bildirir:

```text
Class bilgisi bulunamadı. Bu nedenle hedef class ID ve class adını elle girmeniz gerekiyor.  
```

Yalnızca label satırındaki ilk değer değiştirilir. Koordinatlar, detection kutuları ve polygon noktaları korunur.

Boş negatif label dosyaları boş kalır.

İşlem sonunda `data.yaml` tek hedef class’a göre güncellenir.

Tek class’lı doğrudan YOLO eğitimi için genellikle class ID değerinin `0` olması beklenir. Farklı bir değer seçilirse program uyarı gösterir.

İşlem öncesinde label ve YAML yedeği oluşturulur.

## 3. Datasetleri ana dataset içine birleştir

Bu seçenek birden fazla YOLO datasetini tek bir ana dataset içinde birleştirir.

Kaynak datasetler klasör seçiciyle sırayla seçilir. Seçim sırası yeni ana datasette eklenecek class’ların sırasını etkiler.

Class isimleri klasör adından değil, her datasetin `data.yaml` dosyasındaki `names` alanından okunur.

Program yeni bir hedef dataset oluşturmayı veya mevcut bir ana dataseti kullanmayı sorar.

Yeni dataset oluşturulacaksa ana dizin seçilir ve yeni dataset klasörünün adı girilir.

Mevcut dataset kullanılacaksa herhangi bir dizindeki hedef dataset seçilebilir.

Önerilen seçenek class’ları `data.yaml` içindeki isimlere göre eşleştirir.

Örneğin kaynak dataset:

```yaml
names:  
  0: polar_bear  
  1: black_bear  
  2: brown_bear  
```

Ana dataset:

```yaml
names:  
  0: bear  
  1: boar  
  2: deer  
  3: wolf  
  4: cow  
  5: person  
  6: dog  
```

Birleştirme sonucu:

```yaml
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
```

Kaynak label dosyalarındaki dönüşüm otomatik yapılır:

```text
0 → 7  
1 → 8  
2 → 9  
```

Kaynak dosyalar değiştirilmez. Dönüşüm yalnızca ana datasete kopyalanan label dosyalarına uygulanır.

İki datasette aynı isimli class varsa yeni class oluşturulmaz. Hedef dataset içindeki mevcut class ID kullanılır.

Örneğin kaynakta ve hedefte `brown_bear` varsa ikinci bir `brown_bear` class’ı eklenmez.

Class isimleri karşılaştırılırken gereksiz büyük/küçük harf ve boşluk farklılıkları normalize edilir.

Dosya adı çakışırsa hem görüntü hem label birlikte yeniden adlandırılır:

```text
bear_dataset__image001.jpg  
bear_dataset__image001.txt  
```

Çok sınıflı bir dataseti class ID’lerini değiştirmeden kopyalamak yalnızca iki datasetin ID anlamları tamamen aynıysa güvenlidir.

```text
Kaynak 0 = bear  
Hedef  0 = bear  
```

Kaynakta `0=polar_bear`, hedefte `0=bear` ise doğrudan kopyalamak yanlıştır. Bu nedenle normal kullanımda `data.yaml adlarına göre eşleştir` seçeneği kullanılmalıdır.

Hedef datasette mevcut `data.yaml` bulunursa program class listesini koruma veya kaynak seçim sırasına göre yeniden kurma seçeneklerini sunabilir. Yeniden sıralama yapılırsa mevcut hedef label dosyaları da class isimlerine göre dönüştürülür.

İşlem öncesinde hedef datasetin label ve YAML yedeği oluşturulur.

Kopyalama ve class ID dönüştürme sırasında ilerleme durumu gösterilir. İşlem tamamlandığında ana dataset doğrulanır.

## 4. Train/val/test oranlarıyla yeniden bölüştür

Bu seçenek bir veya birden fazla datasetteki görüntü-label çiftlerini yeniden train, val ve test klasörlerine dağıtır.

Program önce şunları sorar:

```text
Kaç dataset klasörünüz var?  
Tek class içeren dataset klasörlerini seçin  
Çok class içeren dataset klasörlerini seçin  
Train yüzdesi  
Val yüzdesi  
Test yüzdesi  
```

Burada sorulan sayı class sayısı değil, işleme alınacak dataset klasörü sayısıdır.

Seçilen tek class ve çok class datasetlerinin toplamı girilen dataset sayısıyla aynı olmalıdır.

Varsayılan oranlar:

```text
train: %80  
val:   %10  
test:  %10  
```

Oranların toplamı `%100` olmalıdır.

Bir split için `%0` girilebilir. Bu durumda ilgili split klasörü boş olarak oluşturulur.

`val` veya `test` klasörü kaynak datasette bulunmasa bile gerekli klasörler otomatik oluşturulur.

Program mevcut train, val ve test klasörlerindeki bütün resim-label çiftlerini toplar ve yeniden dağıtır.

Tek class içeren datasetlerde dağıtım görüntü sayısı temel alınarak yapılır.

Çok class içeren datasetlerde label içerikleri okunur ve mevcut class dağılımı yaklaşık olarak korunmaya çalışılır.

Örneğin mevcut kutu dağılımı:

```text
polar_bear : black_bear : brown_bear  
3          : 5          : 8  
```

ise her class’tan zorla eşit sayıda görüntü seçilmez. Yaklaşık `3:5:8` dağılımı korunmaya çalışılır.

Nadir class içeren görüntülere yerleştirme sırasında öncelik verilir.

Aynı görüntüde üç class varsa bu görüntü üç class’ın hedefine de katkı sağlar. Bu nedenle çok etiketli datasetlerde her class hedefini matematiksel olarak tam tutturmak her zaman mümkün değildir.

Toplam train, val ve test görüntü sayıları ise belirlenen oranlara tam olarak dağıtılır.

Train, val ve test içindeki görüntüler birbirinden farklı olmalıdır. Aynı görüntü birden fazla split içinde bulunmamalıdır.

Bir class’ın yalnızca train içinde bulunması yeterli değildir. Mümkün olan durumlarda bear gibi eğitilen her class’ın val ve test içinde de örnekleri bulunmalıdır.

Train modelin ağırlıkları öğrenmesi için kullanılır.

Val eğitim sırasında modelin görmediği görüntüler üzerindeki başarısını takip etmek ve ayarları değerlendirmek için kullanılır.

Test ise eğitim ve ayarlama tamamlandıktan sonra modelin son performansını tarafsız biçimde ölçmek için kullanılır.

Aynı videodan veya aynı fotokapan olayından gelen birbirine çok benzeyen kareler farklı split’lere dağıtılmamalıdır. Aksi hâlde model aynı olayın benzer görüntülerini önceden görmüş olur ve sonuçlar gerçekte olduğundan daha iyi çıkabilir.

Program görsel benzerlikten aynı video veya fotokapan olayını otomatik olarak tanıyamaz. Bu karelerin önceden aynı olay grubu altında tutulması gerekir.

Bölüştürme sırasında dosyaların geçici alana alınması ve yeni split’lere yerleştirilmesi için ilerleme durumu gösterilir.

İşlem öncesinde label ve `data.yaml` yedeği oluşturulur. Büyük datasetlerde yüzlerce GB’lık gereksiz yedek oluşmaması için bütün görüntüler yedeğe eklenmez.

## Eksik label denetimi

Bir resmin aynı isimli label dosyası bulunamazsa program şu bilgiyi gösterir:

```text
image001.jpg isimli dosyaya ait label bulunamadı  
```

Ardından şu seçenekler sunulur:

```text
(1) Resmi aç  
(2) Devam et / şimdilik atla  
(3) Programı sonlandır  
```

Resim açıldıktan sonra:

```text
(1) Resmi sil  
(2) Boş label oluştur  
(3) Hiçbir şey yapma ve programı bitir  
```

Boş label oluşturma seçeneği yalnızca negatif görüntüler içindir.

Resmi bulunmayan bağımsız `.txt` label dosyaları otomatik olarak silinmez veya değiştirilmez. Program hata vererek bu dosyaların konumlarını gösterir.

Windows'ta görüntü işletim sisteminin varsayılan görüntüleyicisiyle açılır.

Linux'ta görüntü açmak için `xdg-open` kullanılır. Gerekirse `xdg-utils` paketinin kurulması istenir.

## 5. Ana dataseti images/split + labels/split düzenine çevir

Bu seçenek çalışma düzenindeki ana dataseti eğitim, paylaşım veya ZIP hazırlığı için son klasör düzenine dönüştürür.

Kaynak düzen:

```text
dataset/  
├── train/  
│   ├── images/  
│   └── labels/  
├── val/  
│   ├── images/  
│   └── labels/  
└── test/  
    ├── images/  
    └── labels/  
```

Dönüştürülen düzen:

```text
dataset/  
├── data.yaml  
├── images/  
│   ├── train/  
│   ├── val/  
│   └── test/  
└── labels/  
    ├── train/  
    ├── val/  
    └── test/  
```

Eski `valid` klasörü bulunursa standart `val` klasörüne dönüştürülür.

Dosyalar kopyalanmaz, yeni düzene taşınır.

Dataset kökünde daha önceden `images` veya `labels` klasörü bulunuyorsa yanlışlıkla üzerine yazmayı önlemek için işlem durdurulur.

`data.yaml` içindeki train, val ve test yolları yeni klasör düzenine göre güncellenir.

Dönüştürmeden önce `data.yaml` yedeği datasetin yanına kaydedilir.

İşlem sonunda yeni klasör düzeni ve bütün resim-label eşleşmeleri kontrol edilir.

Bu işlem son paketleme adımı olarak düşünülmelidir. Filtreleme, birleştirme ve yeniden bölüştürme işlemleri çalışma düzeninde yapılmalı, klasör dönüşümü bunlardan sonra uygulanmalıdır.

## 6. Datasetleri yalnızca kontrol et

Bu seçenek dosyaları değiştirmeden bir veya birden fazla dataseti doğrular.

Datasetler farklı disk ve klasörlerden seçilebilir.

Kontrol sırasında ilerleme durumu gösterilir.

Program şu denetimleri yapar:

```text
Her resmin aynı isimli .txt label dosyası var mı?  
Her label’ın karşılık gelen resmi var mı?  
Aynı kök ada sahip birden fazla resim var mı?  
Class ID negatif mi?  
Class ID data.yaml içinde tanımlı mı?  
Koordinatlar 0–1 aralığında mı?  
Detection satırları geçerli mi?  
Polygon satırları geçerli mi?  
Bounding box genişlik ve yükseklik değerleri pozitif mi?  
Boş negatif label dosyaları korunmuş mu?  
Split başına resim ve label sayıları eşit mi?  
Her class için kaç kutu var?  
```

Normal detection satırlarında class ID ile birlikte dört koordinat değeri bulunmalıdır.

Polygon satırlarında class ID’den sonra geçerli sayıda ve çiftler hâlinde koordinat bulunmalıdır.

Örnek doğrulama çıktısı:

```text
train: resim=7000, label=7000, kutu=8450, negatif=900 [OK]  
val  : resim=1000, label=1000, kutu=1210, negatif=125 [OK]  
test : resim=1000, label=1000, kutu=1195, negatif=125 [OK]  
```

Resim sayısıyla kutu sayısının eşit olması gerekmez.

Bir resimde sıfır, bir veya birden fazla kutu bulunabilir.

Eşit olması gereken değer, resim sayısıyla karşılık gelen label dosyası sayısıdır.

## 7. Ana dataseti ZIP dosyası yap

Bu seçenek son klasör düzenine dönüştürülmüş ana dataseti ZIP dosyası hâline getirir.

Önce ZIP yapılacak dataset seçilir.

Ardından ZIP dosyasının kaydedileceği klasör ayrıca seçilir. ZIP dosyası kaynak datasetle aynı dizine kaydedilmek zorunda değildir.

ZIP içine yalnızca gerekli dataset içerikleri eklenir:

```text
data.yaml  
images/  
labels/  
```

ZIP içinde fazladan bir dış dataset klasörü oluşturulmaz.

Örneğin ZIP açıldığında doğrudan şu yapı görülür:

```text
data.yaml  
images/train/  
images/val/  
images/test/  
labels/train/  
labels/val/  
labels/test/  
```

Boş train, val veya test klasörleri korunur.

Büyük datasetler için ZIP64 desteği kullanılır.

İşlem sırasında dosya ve işlenen veri miktarına göre ilerleme gösterilir.

ZIP önce geçici `.zip.part` dosyasına yazılır. İşlem başarıyla tamamlandıktan sonra gerçek `.zip` adına çevrilir.

Aynı isimde ZIP dosyası bulunursa üzerine yazmadan önce açık onay istenir.

ZIP oluşturulduktan sonra arşiv bütünlük testi yapılır ve gerekli klasörlerin arşiv içinde bulunup bulunmadığı doğrulanır.

## Son doğrulama

Filtreleme, class birleştirme, dataset birleştirme, yeniden bölüştürme ve klasör dönüşümü gibi işlemler tamamlandığında sonuç dataset otomatik olarak kontrol edilir.

Kontrol sonucunda eksik resim, eksik label, geçersiz class ID veya bozuk koordinat bulunursa konumları kullanıcıya gösterilir.

İşlem başarılı olsa bile eğitimden önce özellikle val ve test görüntülerinin görsel olarak incelenmesi önerilir.

Kontrol edilmesi gereken temel noktalar:

```text
Aynı görüntü farklı split’lerde bulunuyor mu?  
Aynı olaydan gelen çok benzer kareler farklı split’lere dağılmış mı?  
Her önemli class val ve test içinde yeterince temsil ediliyor mu?  
Negatif görüntüler gerçekten hedef obje içermiyor mu?  
Filtreleme sonrasında görünür fakat etiketsiz bırakılmış objeler var mı?  
```

## Yedekleme

Veri değiştiren işlemlerden önce otomatik yedek oluşturulur.

Örnek yedek adları:

```text
bear_filter_backup_....zip  
bear_filter_copy_backup_....zip  
bear_class_merge_backup_....zip  
bear_class_id_backup_....zip  
bear_merge_backup_....zip  
bear_split_backup_....zip  
labels_empty_backup_....zip  
```

Yedekler ilgili datasetin yanında oluşturulur.

Filtreleme, class birleştirme, dataset birleştirme ve yeniden bölüştürme yedekleri çoğunlukla label dosyalarıyla `data.yaml` dosyasını içerir.

Bütün görüntüler yedeğe eklenmez. Bu sayede büyük datasetlerde çok büyük ve gereksiz yedek dosyaları oluşmaz.

Görüntü silme işlemlerinde otomatik metadata yedeğinin silinen görüntüleri geri getiremeyeceği unutulmamalıdır.

## Windows kurulumu

```powershell
py -m pip install questionary PyYAML  
py "merge_yolo_datasets(1).py"  
```

Program başlangıçta Windows veya Linux ortamını seçmenizi ister.

## Linux kurulumu

Sanal ortam kullanılması önerilir:

```bash
python3 -m venv .venv  
source .venv/bin/activate  
python -m pip install --upgrade pip  
python -m pip install questionary PyYAML  
python "merge_yolo_datasets(1).py"  
```

Eksik label kontrolünde resmi sistem görüntüleyicisiyle açmak için gerekirse `xdg-utils` kurulmalıdır.

Debian veya Ubuntu:

```bash
sudo apt install xdg-utils  
```

Fedora:

```bash
sudo dnf install xdg-utils  
```

Arch Linux:

```bash
sudo pacman -S xdg-utils  
```

Programın klasör seçme, dosya yolu oluşturma ve dataset işlemleri hem Windows hem de Linux yol yapıları dikkate alınarak çalışır.

