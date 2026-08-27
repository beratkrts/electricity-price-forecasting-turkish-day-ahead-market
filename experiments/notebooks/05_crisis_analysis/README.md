# Kriz Analizi Notebook'ları

`CRISIS_ANALYSIS_PLAN.md` §8'in analiz tarafı. Her notebook kendi başına koşar
(veri DB'den gelir, aralarında dosya bağımlılığı yoktur) ama **sırayla okunmalıdır**.

| # | Notebook | Ne anlatır |
|---|---|---|
| 01 | `01_counterfactual_model.ipynb` | Aletin kendisi: kontrafaktüel nasıl kuruldu, ne kadar doğru, "anormal" eşiği ne, İran penceresi ve 2024 karşılaştırması |
| 02 | `02_model_bias_investigation.ipynb` | Aletin iki dönemde neden bozuk olduğu — **elenen hipotezler dahil** |
| 03 | `03_gas_tariff_discovery.ipynb` | Sapmanın kaynağı ve çözümü: BOTAŞ tarifesi ≠ GRF, `silver.gas_tariff_electricity`, ve **v5 testi** |
| 04 | `04_event_attribution_attempt.ipynb` | **Üç negatif sonuç**: haberden günlük olay atfetme çalışmıyor. Kural filtresi, elle etiketleme (182 haber altın küme), korpus teşhisi |

İlgili, ama canlı model tarafında:
`03_ptf_model_comparison/05_gas_tariff_feature_backtest.ipynb` — BOTAŞ tarifesi canlı
tahmin modelini iyileştiriyor mu (hayır: rMAE 0,817 → 0,817).

## Koşmadan önce

- PostgreSQL ayakta olmalı; `gold.crisis_counterfactual` dolu olmalı.
  Değilse: `python scripts/build_analysis_model.py --variant both --write`
- `02` §4 model eğitir (~30 sn). Diğer hücreler sorgu ve grafik.
- Tümünü baştan koşmak: `python -m nbconvert --to notebook --execute --inplace <dosya>`

## Sorgu yazarken

`gold.crisis_counterfactual` **beş sürüm** içerir. Yeni analizde daima:

    variant = 'fundamental'   AND   model_name = 'crisis_cf_v5'

`crisis_cf_v5` = GRF yerine BOTAŞ elektrik üretim tarifesi. MAE 9,54 (v1: 10,29),
sapması $10'u aşan ay 2 (v1: 7), ve İran olay profili ilk kez tutarlı. Gerekçe: `03` §7.

Filtreyi atlarsan sonuçlar sessizce karışır. Özellikle v2 İran sinyalini +$16,5'ten
+$0,2'ye düşürüyor — yani yanlış filtre "olay yok" dedirtir.

`01` ve `02` bilerek v1 üzerinden yazılmıştır: biri aletin kuruluşunu, diğeri
v1'in sapmasının nasıl teşhis edildiğini anlatır. Oradaki v1 rakamları tarihsel
kayıttır, güncel ölçüm değildir.

## Bu klasörün kuralı

Kriz hattına yeni bir analiz eklendiğinde **aynı oturumda buraya da eklenir**.
Negatif sonuçlar da yazılır — `04` tamamen negatif sonuçlardan oluşuyor ve en değerli
notebook'lardan biri: bir sonraki oturumun aynı üç yolu tekrar denemesini engelliyor.
Notebook'lar sonradan yazılan bir özet değil, analizin kendisinin kaydıdır:
elenen hipotezler de kalır, çünkü bir sonraki oturumun aynı yolu tekrar yürümemesi
onlara bağlı.
