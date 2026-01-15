# 🌐 QoS Odaklı Çok Amaçlı Ağ Rotalama Optimizasyonu

![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![Library](https://img.shields.io/badge/Library-NetworkX-green) ![Status](https://img.shields.io/badge/Status-Completed-success)

Bu proje, karmaşık ağ topolojileri üzerinde **Hizmet Kalitesi (QoS)** parametrelerini (Gecikme, Güvenilirlik, Bant Genişliği) optimize etmek amacıyla geliştirilmiş kapsamlı bir simülasyon ve analiz çerçevesidir. 

Proje, gerçek dünya ağ problemlerini simüle etmek için aynı topoloji üzerinde üç farklı yapay zeka yaklaşımını kıyaslar:

1.  **🧬 Genetik Algoritma (GA):** Doğal seleksiyon ve mutasyon prensiplerine dayalı evrimsel rota optimizasyonu.
2.  **🐜 Karınca Kolonisi Optimizasyonu (ACO):** Sürü zekası (Swarm Intelligence) ve feromon izi mantığıyla en kısa yol analizi.
3.  **🤖 Q-Learning (RL):** Pekiştirmeli öğrenme (Reinforcement Learning) kullanarak dinamik ortamda ajan tabanlı rota keşfi.

---

## 📂 Proje İçeriği ve Dosya Yapısı

* `ag.py`: CSV verilerini okuyarak düğüm (node) ve kenar (edge) yapılarıyla `networkx` grafını oluşturur.
* `deney_duzenegi.py`: Otomasyonun merkezidir. Algoritmaları belirli parametrelerle yarıştırır ve raporlar.
* `BSM307_317_*.csv`: Ağ topolojisi ve talep (demand) verilerini içeren veri setleri.

---

## 🚀 Kurulum ve Gereksinimler

Proje **Python 3.10+** ile uyumludur. Gerekli bağımlılıkları yüklemek için aşağıdaki komutu kullanabilirsiniz:

bash
"python3 -m pip install pandas networkx matplotlib"

hızlı başlangıç örnek tablosu
python3 deney_duzenegi.py \
  --demands 20 \
  --repeats 5 \
  --algorithms ga aco qlearning \
  --weights 0.4 0.4 0.2 \
  --seed 42

Parametre,Açıklama,Örnek
--demands,CSV dosyasından işlenecek toplam talep (rota isteği) sayısı.,20
--repeats,İstatistiksel doğruluk için her algoritmanın kaç kez çalıştırılacağı.,5
--algorithms,Kıyaslamaya dahil edilecek algoritmalar.,ga aco qlearning
--weights,"QoS öncelik ağırlıkları (Sırasıyla: Gecikme, Güvenilirlik, Maliyet).",0.4 0.4 0.2
--seed,Tekrarlanabilirlik: Sabit bir çekirdek değer vererek sonuçların her çalışmada aynı olmasını sağlar.,42
--output,Sonuç raporu için özel dosya adı tanımlar.,sonuc.txt

📊 Raporlama ve Sonuçlar
Simülasyon tamamlandığında, deney_detay_YYYYMMDD_HHMMSS.txt formatında zaman damgalı bir teknik rapor üretilir.

Bu raporda şunlar bulunur:

Başarı Metrikleri: Algoritmaların geçerli bir rota bulma başarısı (Success Rate).

Performans: Ortalama hesaplama süresi (ms) ve bellek kullanımı.

Yol Kalitesi: Bulunan rotaların toplam gecikmesi, darboğaz bant genişliği ve güvenilirlik skorları.

Hata Analizi: Başarısız denemelerin (örn. yetersiz bant genişliği, döngü oluşumu) teknik nedenleri.

⚖️ Tekrarlanabilirlik (Seed Mantığı)
Bilimsel kıyaslamanın tutarlılığı için tüm algoritmalar merkezi bir rastgelelik (Seed) mekanizması kullanır.

--seed parametresi verildiğinde, algoritmaların (özellikle Q-Learning keşif süreci ve GA mutasyonları) kararları deterministik hale gelir.

Bu sayede farklı bilgisayarlarda aynı sonuçlar elde edilebilir ve algoritmalar adil bir şekilde kıyaslanabilir.

Geliştirici
Eyüphan Altuntaş - Bilgisayar Teknolojileri ve Bilişim Sistemleri öğrencisi.
