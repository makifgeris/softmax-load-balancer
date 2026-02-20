# Softmax Load Balancer Simülasyonu

Client-side load balancing için Softmax action selection algoritmasının non-stationary ortamda performans analizi.

## Özellikler

- **Softmax Action Selection**: Temperature parametresi (τ) ile exploration-exploitation dengesi
- **Non-Stationary Ortam**: Sunucu latency'leri dinamik olarak değişir (drift, trend, rejim değişimleri)
- **Karşılaştırmalı Analiz**: Random, Round-Robin ve Softmax algoritmalarının performans karşılaştırması
- **Görselleştirme**: Kümülatif ödül, Q-değerleri, latency değişimi ve seçim frekansları grafikleri
- **Runtime Analizi**: Algoritmaların hesaplama maliyeti analizi

## Kurulum

```bash
pip install -r requirements.txt
```

## Kullanım

### Temel Kullanım

```bash
python softmax.py
```

### Parametrelerle Çalıştırma

```bash
python softmax.py --servers 5 --requests 5000 --tau 0.5 --shift-interval 200 --seed 7
```

### Parametreler

- `--servers`: Sunucu sayısı (K) [varsayılan: 5]
- `--requests`: İstek sayısı (N) [varsayılan: 5000]
- `--tau`: Softmax temperature parametresi (τ > 0) [varsayılan: 0.5]
- `--shift-interval`: Rejim değişim periyodu (adım) [varsayılan: 200]
- `--seed`: Random seed [varsayılan: 7]
- `--no-plot`: Grafikleri gösterme
- `--save-plots`: Grafikleri belirtilen klasöre kaydet

### Örnek Komutlar

```bash
# Daha fazla sunucu ile test
python softmax.py --servers 10 --requests 10000

# Farklı temperature değerleri
python softmax.py --tau 0.1  # Daha fazla exploitation
python softmax.py --tau 2.0  # Daha fazla exploration

# Grafikleri kaydet
python softmax.py --save-plots ./results

# Grafik gösterme, sadece konsol çıktısı
python softmax.py --no-plot
```

## Algoritma Detayları

### Softmax Action Selection

Softmax, her sunucunun seçilme olasılığını Q-değerlerine göre hesaplar:

```
P(server_i) = exp(Q_i / τ) / Σ_j exp(Q_j / τ)
```

- **τ (tau)**: Temperature parametresi
  - Düşük τ: Daha fazla exploitation (en iyi sunucuyu seç)
  - Yüksek τ: Daha fazla exploration (tüm sunucuları dene)

### Q-Değer Güncelleme

Incremental mean yöntemi ile:

```
Q(server) ← Q(server) + (1/n) * (reward - Q(server))
```

- reward = -latency (düşük latency = yüksek ödül)
- n: Sunucunun seçilme sayısı

### Non-Stationary Ortam

Her sunucu için latency modeli:

- **Base Latency**: Temel gecikme süresi
- **Drift**: Sinüzoidal değişim (periyodik)
- **Trend**: Uzun vadeli artış/azalış
- **Noise**: Gaussian gürültü
- **Regime Shifts**: Periyodik rejim değişimleri

## Çıktılar

### Konsol Çıktısı

- Kümülatif ödül (cumulative reward)
- Ortalama latency
- Sunucu seçim frekansları
- Algoritma başına ortalama seçim süresi (µs)

### Grafikler

1. **Kümülatif Ödül Karşılaştırması**: Algoritmaların zaman içinde performansı
2. **Softmax Seçim Frekansları**: Her sunucunun ne sıklıkla seçildiği
3. **Sunucu Latency Değişimi**: Gerçek latency gözlemleri
4. **Q-Değerleri Evrimi**: Softmax'in sunucular hakkında öğrendikleri

## Performans

- **Softmax**: O(K) seçim, O(1) güncelleme → O(N·K) toplam
- **Random**: O(1) seçim
- **Round-Robin**: O(1) seçim

Tipik seçim süreleri: ~1-5 µs/seçim

## Gereksinimler

- Python 3.8+
- NumPy 1.24+
- Matplotlib 3.7+

## Lisans

MIT
