#!/usr/bin/env python3
"""Dashboard'ın okuduğu türetilmiş tabloları doldur.

Canlı dashboard (`../enerji_fiyat_tahmini`) `/api/event-transmission` endpoint'i
üç tabloyu okur:
  gold.event_decomposition_v2   — olay ayrıştırması (event_decomposition.py yazar;
                                  bu script LABEL/DESCRIPTION/END_DATE kolonlarını ekler)
  gold.transmission_summary     — Faz C özet metrikleri (bu script yazar)
  silver.news_attention_daily   — dikkat endeksi (build_attention_index.py yazar)

Olay metinleri OLAY_EVRENI.md §5 ile aynı. Faz C sayıları
reports/OLAY_YAPISAL_GECIS_BULGULARI.md ile aynı.

    .venv/bin/python experiments/scripts/dashboard_tables.py
"""
from __future__ import annotations
import sys
from pathlib import Path

from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from db.connection import get_db_engine

# event_id -> (label, onset, end_date|None, affected, description)
CATALOG = {
 "G1": ("2021 Avrupa gaz fiyatı tırmanışı", "2021-09-01", None, "Küresel gaz",
        "Düşük depo, düşük Rus akışı ve Asya LNG rekabetiyle Avrupa gaz fiyatları "
        "Eylül–Aralık 2021 boyunca tırmandı; TTF aylık artışı %40'a ulaştı."),
 "G2": ("Rusya-Ukrayna işgali", "2022-02-24", "2022-12-31", "Küresel gaz",
        "24 Şubat 2022 işgali boyunca Avrupa gaz fiyatları rekor kırdı; TTF Mart "
        "2022'de %59 arttı. Türkiye MCP'si dolar bazında 2022 boyunca tarihi zirvesine çıktı."),
 "G3": ("Freeport LNG tesisi patlaması", "2022-06-08", None, "Küresel gaz",
        "8 Haziran 2022'de ABD Freeport LNG tesisindeki patlama ABD ihracat "
        "kapasitesini yaklaşık %20 düşürdü, tesis Kasım'a dek kapalı kaldı; TTF Temmuz'da %60 arttı."),
 "G4": ("Nord Stream gaz akışının durması", "2022-09-02", None, "Küresel gaz",
        "Eylül 2022'de Gazprom Avrupa'ya gaz akışını tamamen kesti, 26 Eylül'de "
        "boru hattı sabote edildi. TTF zaten yaz zirvesindeydi."),
 "G5": ("2022-23 Avrupa gazının normalleşmesi", "2022-10-15", None, "Küresel gaz",
        "Ilıman kış, dolu depolar ve talep düşüşüyle Avrupa gaz fiyatları Ekim "
        "2022 – Ocak 2023 arasında %45 geriledi."),
 "G6": ("2026 Hürmüz krizi ve hat tehdidi", "2026-02-25", None, "Küresel gaz + petrol",
        "Şubat-Mart 2026'da İran Hürmüz Boğazı'nı kapattığını açıkladı, "
        "Türkiye-Rusya gaz hatlarına saldırı iddiaları çıktı; TTF Mart'ta %63 arttı, Brent 80 doları aştı."),
 "S1": ("İran doğal gaz kesintisi", "2022-01-19", "2022-02-05", "Türkiye gaz arzı",
        "19 Ocak 2022'de İran'ın Türkiye'ye gaz sevkiyatını yaklaşık 10 gün "
        "kesmesiyle gazdan üretim %28,6 düştü; sanayiye gaz kısıtı ve elektrik kesintileri uygulandı."),
 "O1": ("ABD-İsrail-İran çatışması", "2025-06-13", "2025-06-26", "Petrol / jeopolitik",
        "13 Haziran 2025 çatışmasında Brent 78 doları aştı, İran Meclisi Hürmüz "
        "kararı aldı. Küresel gaz fiyatları büyük ölçüde tepkisiz kaldı."),
}

# metric -> (value, label, unit)
FINDINGS = {
 "link1_pass_up":    (0.36,  "TTF → BOTAŞ tarifesi kümülatif geçiş (yukarı, 3 ay)", "EUR/EUR"),
 "link1_pass_down":  (0.07,  "TTF → BOTAŞ tarifesi kümülatif geçiş (aşağı, 3 ay)", "EUR/EUR"),
 "link1_asym_p":     (0.008, "Asimetri Wald testi p-değeri", "p"),
 "link3_delta":      (1.77,  "Marjinal yakıt maliyeti → MCP geçiş katsayısı", "USD/USD"),
 "link3_delta_se":   (0.07,  "Link 3 katsayı standart hatası", "USD/USD"),
 "tobit_supp_2021":  (61.0,  "Tavanın saatlik fiyat varyansını bastırma oranı — 2021", "%"),
 "tobit_supp_2024":  (60.0,  "Tavanın saatlik fiyat varyansını bastırma oranı — 2024", "%"),
 "tobit_supp_2025":  (69.0,  "Tavanın saatlik fiyat varyansını bastırma oranı — 2025", "%"),
 "tobit_supp_2026":  (29.0,  "Tavanın saatlik fiyat varyansını bastırma oranı — 2026", "%"),
 "tobit_clip_usd":   (19.9,  "Üst-sansürlü saatlerde ort. kırpılan miktar", "USD/MWh"),
 "tobit_cens_share": (17.2,  "Üst-sansürlü saat payı (2021-2026)", "%"),
}


def main() -> int:
    eng = get_db_engine()
    with eng.begin() as c:
        for col, typ in [("label", "text"), ("end_date", "date"),
                         ("affected", "text"), ("description", "text")]:
            c.execute(text(f"ALTER TABLE gold.event_decomposition_v2 ADD COLUMN IF NOT EXISTS {col} {typ}"))
        for eid, (label, _onset, end, aff, desc) in CATALOG.items():
            c.execute(text("""UPDATE gold.event_decomposition_v2
                SET label=:l, end_date=:e, affected=:a, description=:d WHERE event_id=:id"""),
                {"l": label, "e": end, "a": aff, "d": desc, "id": eid})

        c.execute(text("""CREATE TABLE IF NOT EXISTS gold.transmission_summary (
            metric text PRIMARY KEY, value numeric, label text, unit text,
            source text DEFAULT 'passthrough_model.py + tobit_hourly.py',
            updated_at timestamptz DEFAULT now())"""))
        for m, (v, l, u) in FINDINGS.items():
            c.execute(text("""INSERT INTO gold.transmission_summary (metric, value, label, unit)
                VALUES (:m,:v,:l,:u) ON CONFLICT (metric) DO UPDATE
                SET value=EXCLUDED.value, label=EXCLUDED.label, unit=EXCLUDED.unit, updated_at=now()"""),
                {"m": m, "v": v, "l": l, "u": u})
    print(f"→ gold.event_decomposition_v2: {len(CATALOG)} olaya label/description eklendi")
    print(f"→ gold.transmission_summary: {len(FINDINGS)} metrik")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
