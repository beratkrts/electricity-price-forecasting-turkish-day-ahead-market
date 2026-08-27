# Yerel LLM Kurulumu — Kriz & Olay İstihbarat Sistemi

**Tarih:** 17 Ağustos 2026
**Kısıt:** Sıfır bütçe. Tüm çıkarım yerel, M1 MacBook Air 16 GB üzerinde.
**Backend kararı:** MLX + `outlines`. llama.cpp yedekte.
**Bağlam:** `CRISIS_ANALYSIS_PLAN.md` §6. Şema: `src/crisis/labeling_schema.py`.

---

## 0. Ölçülmüş kısıtlar

| | Değer | Sonucu |
|---|---|---|
| RAM | 16 GB (unified) | 8B 6-bit + KV cache ≈ 7,5 GB — sığar, tarayıcıyı kapat |
| **Boş disk** | **12 GB** | Tek backend, tek model. İkisini birden kuramayız. |
| CPU/GPU | Apple M1, fansız | Sürekli yükte %30-50 yavaşlama beklenir |
| macOS | 26.3.1 | Metal var |
| Kurulu | `brew`, `git`, `cmake`, Python 3.11.8, `torch` | `mlx`, `mlx-lm`, `outlines` yok |

Model boyutları (HuggingFace'ten HEAD ile ölçüldü):

| MLX | Boyut | | GGUF (yedek) | Boyut |
|---|---|---|---|---|
| `mlx-community/Qwen3-8B-4bit` | 4,29 GB | | `Qwen3-8B-Q4_K_M` | 4,68 GB |
| **`mlx-community/Qwen3-8B-6bit`** | **6,20 GB** | | `Qwen3-8B-Q5_K_M` | 5,45 GB |
| `mlx-community/Qwen3-8B-8bit` | 8,11 GB | | `Qwen3-8B-Q6_K` | 6,26 GB |

---

## 1. Neden MLX (ve neden llama.cpp değil)

Plan §6.3 "MLX de denenmeli, Apple Silicon'da genelde llama.cpp'den hızlı" diyordu.
İlk taslakta llama.cpp'yi önermiştim; iki argümanım da incelemeye dayanmadı:

**Disk argümanı geçersizdi.** 12 GB kısıtı *iki backend'i birden* kurmaya karşıydı,
llama.cpp'yi MLX'e tercih etmeye değil. Tek backend kuracaksak disk eşit.

**"Sunucu kullan, CLI değil" argümanı MLX'te hiç doğmuyor.** llama.cpp'de haber başına
`llama-cli` çağırmak 28.000 kez 6 GB model yükler; bu yüzden `llama-server` + HTTP
gerekiyordu. MLX süreç içi çalışıyor: `mlx_lm.load()` bir kez, sonra 28.000 çağrı aynı
Python sürecinde. Sunucu yaşam döngüsü yok, HTTP hop'u yok, port çakışması yok.
**Mimari olarak daha basit.**

Lehine iki şey daha:

**Yapısal üretim tam destekli.** `outlines[mlxlm]` resmî entegrasyon ve dokümantasyonu
JSON Schema, regex, çoktan seçmeli ve CFG'yi listeliyor. `labeling_schema.JSON_SCHEMA`
zaten enum'lardan üretiliyor — yeniden yazılacak bir şey yok.

**Açık önek cache kontrolü.** `mlx_lm`'de `prompt_cache` ilgili 42 referans var
(`make_prompt_cache`, diske kaydet/yükle). Sabit ~3.000 token'lık önek 28.000 kez
kullanılacak; cache'i **bir kez kurup diske yazmak**, sunucunun sezgisel cache'ine
güvenmekten daha belirlenimli.

### Karşı taraf — dürüst maliyet

| Maliyet | Ağırlık |
|---|---|
| `outlines` + `mlx-lm`, GBNF-in-llama.cpp'ye göre daha az yol kat etmiş bir hat | Gerçek risk. Azaltıcı: `parse_output()` kısıt arızasını **anında** yakalar ve 5 altın örnekle duman testi dakikalar içinde belli eder — boşa geçmiş bir geceden sonra değil. |
| Yazdığım GBNF kullanılmıyor | Bedelsiz. Aynı enum'lardan üretiliyor, llama.cpp yedeğe geçerse hazır. |
| `prob ::= "0"\|"1"\|"0.NN"` daraltması kayboluyor | Kozmetik. JSON Schema 0-1 arası her ondalığı kabul ediyor, `validate()` 2 haneye yuvarlıyor. |
| Kısıtlı üretim **batch ile çalışmıyor** (outlines dokümanı) | Kayıp yok — `--parallel 1` seri işleme zaten planlıydı (16 GB'da paralel slot KV cache'i çoğaltıp takasa sokar). |

**Yedek planı:** outlines entegrasyonu direnirse `brew install llama.cpp` + zaten yazılmış
`src/crisis/grammar/news_event.gbnf` — 5 dakikalık geçiş.

---

## 2. Kuantizasyon: 6-bit

MLX'te 5-bit yok; seçenekler 4 / 6 / 8-bit.

Plan Q5_K_M'yi "Türkçe'de kuantizasyon kaybı daha belirgin" diye seçmişti ve κ > 0,6
hedefi koymuştu. 6-bit (6,20 GB) Q5_K_M'den (5,45 GB) **daha iyi** kalite, 0,75 GB fazla
yer. 4-bit 1,9 GB tasarruf ediyor ama projeyi öldürebilecek tek şeyin (Türkçe etiket
kalitesi) üzerinde tasarruf yapmak yanlış yer.

**6-bit öneriyorum.** Disk sıkışırsa modeli küçültmek yerine yer aç:
`docker system prune`, `~/Library/Caches`, İndirilenler.

Kurulum sonrası kalan: 12 − 6,2 − ~0,4 (paketler) ≈ **5,4 GB.** Sıkı ama çalışır.
3 GB'ın altına düşme.

---

## 3. Kurulum

### 3.1 Paketler

```bash
cd /Users/beratkaratasoglu/etkb_intern_project/enerji_fiyat_tahmini
pip install "outlines[mlxlm]"
```

`mlx`, `mlx-lm`, `transformers`, `datasets` bunun içinde geliyor. `torch` kurulu ama
çakışmıyor — MLX bağımsız.

Doğrula:

```bash
python -c "import mlx.core as mx, mlx_lm, outlines; print(mx.default_device())"
# Device(gpu, 0) görmelisin — cpu görürsen Metal devrede değil
```

### 3.2 Model

Model repoya girmez. HF cache'i varsayılan olarak `~/.cache/huggingface`:

```bash
python - <<'PY'
import mlx_lm
model, tok = mlx_lm.load("mlx-community/Qwen3-8B-6bit")   # ilk çağrı indirir
print("yüklendi:", sum(p.size for _, p in mlx_lm.utils.tree_flatten(model.parameters()))/1e9, "milyar parametre")
PY
```

İndirme kesilirse aynı komut kaldığı yerden devam eder (HF cache blok bazlı).

> **Not:** yer sorunu olursa modelin nereye indiğini `HF_HOME` ile değiştir.
> `du -sh ~/.cache/huggingface` ile izle.

### 3.3 Qwen3 düşünme modu — atlanmaması gereken adım

Qwen3 varsayılan olarak `<think>...</think>` üretir. Kısıtlı üretim ilk token'ı `{` olmaya
zorladığı için o blok fiziksel olarak çıkamaz — **ama model kısıtla güreşir ve kalite
düşer.** Prompt sonuna soft anahtar ekle:

```
/no_think
```

Etiketleme bir sınıflandırma işi; zincirleme akıl yürütmeye ihtiyacı yok ve token
maliyetini üç katına çıkarır.

---

## 4. Duman testi

Gerçek şema, gerçek haberler. İran penceresi artık DB'de (440/440 haber, id 46432-46892).

```bash
python -m src.crisis.labeling_schema --check      # şema öz-kontrolü (LLM'siz)
python scripts/smoke_test_labeling.py             # yazılacak
```

Test şunu yapacak:

1. `mlx_lm.load("mlx-community/Qwen3-8B-6bit")` — bir kez
2. `outlines.from_mlxlm(model, tok)`
3. `labeling_schema.GOLD_EXAMPLES`'taki 5 örneğin **girdisini** DB'den çeker
4. Modeli `output_type=<JSON_SCHEMA>` ile koşturur
5. Çıktıyı `parse_output()` ile doğrular, sonra **elle yazılmış altın cevapla alan alan
   karşılaştırır**

**Kabul kriterleri:**

| Kontrol | Eşik | Başarısızsa |
|---|---|---|
| `parse_output()` hatasız | 5/5 | Kısıt devrede değil — `output_type` gerçekten geçiyor mu? |
| `kanal` eşleşmesi | 5/5 | Kanal tanımları promptta net değil |
| `ileriye_donuk` eşleşmesi | 5/5 | **En kritik alan.** Few-shot'taki 4. örnek yetersiz. |
| `alaka_skoru` yönü doğru | negatifler < 0,3, pozitifler > 0,6 | Skor rehberi promptta yeniden yazılmalı |
| `etki_baslangici` | 31 Ocak örneğinde `2022-01-31T08:00` | Yürürlük anı açıklaması yetersiz |

Bunlar 5 örnekte tutuyorsa 200'lük altın kümeye geçilir. Tutmuyorsa promptu düzeltmek
5 dakikalık iş — bir geceyi çöpe atmadan önce.

---

## 5. Hız ölçümü

```bash
python scripts/bench_labeling.py --n 30      # yazılacak
```

Ölçülecekler:

- **Soğuk prefill** — sabit önek (~3.000 token) ilk kez, cache yok
- **Sıcak prefill** — önek cache'lenmişken haber başına (~800 token hedef haber)
- **Üretim** — 8 alanlı JSON ≈ 120-180 token
- **Haber başına toplam** — gecelik bütçenin girdisi

```
haber/gece = (kullanılabilir saat × 3600) / (haber başına saniye)
```

Fansız M1 Air'da ilk 15 dakikaya göre değil, **1. saate göre** hesapla — termal kısılma
%30-50 alır.

---

## 6. Sıra

1. `pip install "outlines[mlxlm]"` + model indirme
2. **Duman testi** — 5 altın örnek, yukarıdaki kabul kriterleri
3. **Hız ölçümü** — gecelik bütçe netleşir
4. **Kural filtresi kalibrasyonu** — korpus bitince `keywords` + `section` + başlık
   kalıbı. Günlük fiyat raporu (`Spot elektrik fiyatı DD.MM.YYYY`) tek başına korpusun
   ~%7'si, regex'le bedava gider. Bu adım gecelik koşunun kaç gece olduğunu belirliyor.
5. **`silver.news_events` şeması** + gecelik koşu scripti — toplayıcıyla aynı kesinti
   disiplini (parti bazlı commit, anti-join ile devam, SIGINT'te açık partiyi kapat)
6. **200 haberlik altın küme + kappa** — kalite çıpası. κ > 0,6 hedef; 0,4-0,6 çıkarsa
   sistem "aday üretici" olur, otomatik etiketleyici olmaz (plan §6)

---

## 7. Riskler

| Risk | Belirti | Ne yapılır |
|---|---|---|
| **Disk** | Kurulum sonrası ~5,4 GB kalıyor | Önce yer aç. `HF_HOME` ile modeli başka diske alabilirsin. |
| **Bellek baskısı** | Takas başlar, hız çöker | Tek istek seri, tarayıcı kapalı. `vm_stat` ile izle. |
| **Termal kısılma** | İlk saat hızlı, sonra yavaş | Beklenen. ETA'yı 1. saate göre kur. |
| **outlines/mlx-lm uyumu** | Kurulumda veya ilk çağrıda patlama | `mlx` 0.32.0 / `mlx-lm` 0.31.3 / `outlines` 1.3.3. Sürüm çakışırsa `mlx-lm` sürümünü outlines'ın istediğine sabitle. Çözülmezse llama.cpp yedeği. |
| **Türkçe kalite** | κ < 0,4 | 8-bit'e çık (8,11 GB — disk gerekir) veya few-shot'ı büyüt. Çözülmezse LLM'i **filtre** olarak kullan, etiketleyici olarak değil. |
| **Kısıt devrede değil** | Çıktı serbest metin | `parse_output()` anında yakalar. `output_type` argümanı gerçekten geçiyor mu kontrol et. |

---

## Ek: llama.cpp yedeği

MLX yolu tıkanırsa, 5 dakika:

```bash
brew install llama.cpp
mkdir -p ~/models/qwen3 && cd ~/models/qwen3
curl -L --continue-at - -o Qwen3-8B-Q5_K_M.gguf \
  https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q5_K_M.gguf
llama-server -m ~/models/qwen3/Qwen3-8B-Q5_K_M.gguf -c 8192 -ngl 99 --port 8081 --parallel 1
```

Gramer hazır: `python -m src.crisis.labeling_schema --write-grammar` →
`src/crisis/grammar/news_event.gbnf`. Sunucu isteğinde `grammar` (GBNF içeriği string
olarak), `cache_prompt: true`, `temperature: 0`, `n_predict: 200`.

Bu yolda `llama-cli`'yi haber başına **çağırma** — her çağrı modeli yeniden yükler.
`llama-server` kalıcı süreç olarak çalışır.
