#!/usr/bin/env python3
"""silver.market_events — küratörlenmiş olay kataloğu.

Girdi: market_events_candidates.json (54 aday, build_market_events_candidates.py).
Çıktı: silver.market_events (23 satır) + market_events_curated.json (drop/merge izi).

İş bölümü MARKET_EVENTS_BRIEF.md'deki gibi: retrieval + tarihleme otomatik,
seçim/açıklama/dedupe insan. Aday dosyasındaki auto_context aynı-tarihli
çiftlerde birebir aynı olduğu için (pencere aynı) ölçümler burada FINAL
pencereden yeniden hesaplanıyor.

İki metin alanı bilinçli olarak ayrı:
  description    — nötr. Ne oldu + hangi haberde. Dashboard bunu gösterir.
  interpretation — mekanizma iddiası. Sunum/tez bunu kullanır.

    .venv/bin/python experiments/scripts/curate_market_events.py [--dry-run]
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from db.connection import get_db_engine

OUT_DIR = ROOT / "experiments/notebooks/05_crisis_analysis"
CF_MODEL, CF_VARIANT = "crisis_cf_v5", "fundamental"
BEFORE_DAYS = 30          # olay öncesi referans penceresi
POINT_EVENT_DAYS = 30     # nokta olaylar için "esna" penceresi

DDL = """
CREATE SCHEMA IF NOT EXISTS silver;
DROP TABLE IF EXISTS silver.market_events;
CREATE TABLE silver.market_events (
    event_id           serial PRIMARY KEY,
    start_date         date NOT NULL,
    end_date           date,                    -- NULL = nokta olay
    label              text NOT NULL,
    mechanism          text NOT NULL,           -- maliyet|arz|regülasyon|jeopolitik|talep|kur
    direction          text NOT NULL,           -- yukari|asagi|karisik
    description        text NOT NULL,           -- NÖTR: ne oldu (dashboard)
    interpretation     text NOT NULL,           -- mekanizma iddiası (sunum/tez)
    confidence         text NOT NULL,           -- kesin|muhtemel
    source_article_ids bigint[] NOT NULL,
    source_urls        text[] NOT NULL,
    candidate_ids      int[] NOT NULL,          -- hangi adaylardan birleşti
    mcp_usd_before     numeric,
    mcp_usd_during     numeric,
    pct_change         numeric,
    cf_residual_usd    numeric,                 -- + = fiyat temellerin üstünde
    cf_at_cap_share    numeric,
    n_hours            integer,
    curated_by         text DEFAULT 'opus',
    created_at         timestamptz DEFAULT now(),
    CONSTRAINT me_mechanism_ck CHECK (mechanism IN
        ('maliyet','arz','regülasyon','jeopolitik','talep','kur')),
    CONSTRAINT me_direction_ck CHECK (direction IN ('yukari','asagi','karisik')),
    CONSTRAINT me_confidence_ck CHECK (confidence IN ('kesin','muhtemel')),
    CONSTRAINT me_window_ck CHECK (end_date IS NULL OR end_date >= start_date)
);
CREATE INDEX ON silver.market_events (start_date);
"""

E = lambda **kw: kw

EVENTS = [
 E(start="2021-07-01", end="2021-07-31", cand=[3, 4],
   label="Temmuz 2021 — gaz tarifesi +%20, fiyat tavana dayandı",
   mechanism="maliyet", direction="yukari", confidence="kesin",
   arts=[43481, 43475, 43863, 44082],
   description="1 Temmuz 2021'de BOTAŞ'ın elektrik üretimine uyguladığı doğal gaz tarifesi "
     "%20 artışla 2.056 TL/1000m³'e yükseldi. Aynı dönemde spot elektrik fiyatı azami fiyat "
     "limitine dayandı ve EPİAŞ tavanı ay içinde 636 TL/MWh'e çıkarıldı.",
   interpretation="Tavanın fiilen bağlayıcı hale geldiği ilk dönem. Gaz tarifesindeki artış "
     "maliyet tabanını yukarı iterken fiyat tavana çarptı; bu noktadan sonra gözlenen MCP artık "
     "serbest bir piyasa dengesi değil, idari sınırın kendisi."),

 E(start="2021-09-01", end="2021-12-31", cand=[6, 7, 9, 10],
   label="Sonbahar 2021 tarife rampası (2.369 → 4.800 TL)",
   mechanism="maliyet", direction="yukari", confidence="kesin",
   arts=[44348, 44725, 45165, 45690, 44520],
   description="Eylül–Aralık 2021 arasında elektrik üretimine verilen doğal gaz tarifesi dört kez "
     "art arda zamlandı: 2.369 → 2.724 → 3.999 → 4.800 TL/1000m³. Kasım artışı tek başına %47'ydi. "
     "Aynı dönemde Avrupa spot gaz fiyatları rekor seviyelere çıkmıştı.",
   interpretation="Küresel gaz şokunun Türkiye'ye idari kanaldan aktarıldığı dönem. Tarife piyasa "
     "fiyatını takip etti, ama bir karar olarak: gecikmeli ve kademeli. Şokun MCP'ye geçişi de aynı "
     "ölçüde gecikmeli — bu gecikme çalışmanın ana mekanizması."),

 E(start="2021-10-15", end=None, cand=[8],
   label="AFL tavanı +%50 → 1.078 TL, formüle bağlandı",
   mechanism="regülasyon", direction="yukari", confidence="kesin",
   arts=[44932, 44931, 44992, 45102],
   description="15 Ekim 2021'de azami fiyat limiti yaklaşık %50 artışla 1.078 TL/MWh'e yükseltildi. "
     "EMO'nun değerlendirmesine göre yeni düzenlemeyle tavan bundan sonra otomatik artacak biçimde "
     "formüle bağlandı; Kasım tavanı 1.131 TL olarak açıklandı.",
   interpretation="Tavanın sabit bir sınır olmaktan çıkıp maliyet endeksli bir mekanizmaya "
     "dönüştüğü an. Bu tarihten sonra tavan, fiyatı sınırlayan dışsal bir kısıt değil, maliyeti "
     "gecikmeli izleyen içsel bir değişken — analizde tavan serisi bu yüzden dışsal muamelesi "
     "göremez. Metodolojik olarak kataloğun en önemli satırı."),

 E(start="2022-01-01", end=None, cand=[12, 13],
   label="Ocak 2022 — tavan ve gaz tarifesi eşzamanlı zam",
   mechanism="regülasyon", direction="yukari", confidence="muhtemel",
   arts=[46117, 46181, 46182, 45954],
   description="1 Ocak 2022'de azami fiyat limiti 1.345 TL/MWh'e, elektrik üretimine verilen gaz "
     "tarifesi %15 artışla 5.520 TL/1000m³'e çıktı. Basında tavanın Ocak ayı için %135 arttığı "
     "bildirildi; silver.price_cap_official serisi aynı tarihte %11'lik bir basamak kaydediyor — "
     "iki rakam farklı tanımlara ait, kaynak uyuşmazlığı çözülmedi.",
   interpretation="Tavan ve yakıt tarifesinin aynı gün hareket etmesi bu dönemde kural haline "
     "geliyor; ikisinin etkisi ayrıştırılamaz. Olay bazlı değil rejim bazlı analiz gerektiğinin "
     "ilk açık işareti."),

 E(start="2022-01-18", end="2022-02-05", cand=[14],
   label="İran doğal gaz kesintisi",
   mechanism="arz", direction="yukari", confidence="kesin",
   arts=[46538, 46565, 46682, 46594, 46643],
   description="Ocak 2022'de İran'dan gelen doğal gaz sevkiyatının kesilmesi üzerine TEİAŞ elektrik "
     "kesintileri duyurdu, sanayi üretimine üç gün 'elektrik molası' verildi, sanayiye gaz kısıtı "
     "uygulandı ve arz açığının bir kısmı yerli kömürden karşılandı. Haber arşivinde olay 'İran' "
     "adıyla değil, sonuçlarıyla yer alıyor.",
   interpretation="Kataloğun en temiz arz şoku: kısa, tanımlı, idari kararla değil fiziksel kısıtla "
     "oluşmuş. Fiyat temellerin belirgin biçimde üstüne çıktı ve saatlerin büyük kısmı tavanda "
     "gerçekleşti — tavan bağlayıcıyken şokun tam etkisi gözlenemez, ölçülen değer bir ALT SINIR. "
     "Ayrıntılı zaman çizelgesi: CRISIS_CASE_IRAN_2022.md."),

 E(start="2022-02-01", end=None, cand=[15, 16],
   label="Şubat 2022 — tavan ve tarife zammı, tavan baskısı zirveye yakın",
   mechanism="regülasyon", direction="yukari", confidence="kesin",
   arts=[46702, 46596, 46689, 46533],
   description="1 Şubat 2022'de azami fiyat limiti 1.524 TL/MWh'e, elektrik üretimi amaçlı gaz "
     "tarifesi %14 artışla 6.300 TL/1000m³'e yükseldi. Spot elektrik aya tavandan girdi; basın "
     "tavanın Şubat'ta 'üçe katlandığını' aktardı.",
   interpretation="Ocak–Nisan 2022, tavanın en sık bağladığı dönem: bu pencerede saatlerin yarıdan "
     "fazlası tavan fiyatta gerçekleşti. Gözlenen fiyatın önemli kısmı piyasa dengesi değil idari "
     "sınır; böyle pencerelerde MCP'den çıkarılan her arz/talep esnekliği tahmini aşağı yanlıdır."),

 E(start="2022-02-24", end="2022-12-31", cand=[17],
   label="Rusya-Ukrayna savaşı",
   mechanism="jeopolitik", direction="yukari", confidence="muhtemel",
   arts=[47100, 47321, 49978, 50023, 49548],
   description="24 Şubat 2022'de başlayan savaş boyunca Avrupa gaz fiyatları rekor seviyelere çıktı; "
     "Kuzey Akım-2'nin sigortası iptal edildi, Eylül 2022'de Gazprom Avrupa'ya gaz akışını tamamen "
     "kesti. Türkiye'de MCP aynı dönemde dolar bazında tarihi zirvesine ulaştı.",
   interpretation="Savaşın başlangıcı ile MCP'nin yükselişi arasında dört-beş aylık bir mesafe var: "
     "enerji basınında anlatı yoğunluğu 24 Şubat'ta dikey sıçrarken fiyat Temmuz 2022'ye kadar tepki "
     "vermiyor (tfidf_signal.png). Şok piyasaya doğrudan değil, BOTAŞ tarifesi üzerinden geçiyor. "
     "Confound ağır — aynı dönemde kur, tavan ve AUF de hareket ediyor; savaşa atfedilen büyüklük "
     "bir ÜST SINIR. 'ABD-İsrail-İran savaşı' satırıyla birlikte okunmalı: o olay bu mekanizmanın "
     "kontrolü."),

 E(start="2022-03-01", end=None, cand=[18, 19],
   label="Mart 2022 — tavan bağlayıcılığı zirvede",
   mechanism="regülasyon", direction="yukari", confidence="kesin",
   arts=[47113, 47184, 47185, 47105],
   description="1 Mart 2022'de azami fiyat limiti 1.745 TL/MWh'e, elektrik üretimi amaçlı gaz "
     "tarifesi %18 artışla 7.450 TL/1000m³'e çıktı. Aynı hafta yük alma/atma talimatlarında "
     "santrallerin korunmasına dair düzenleme yayımlandı.",
   interpretation="Aylık ayarlama ritminin bir basamağı olarak seçildi, ama ölçüm onu ayrı bir "
     "rejim penceresi yapıyor: saatlerin üçte ikisi tavanda geçti — İran kesintisi penceresinden "
     "sonra kataloğun en yüksek tavan bağlayıcılığı. Tavanın bastırdığı oynaklığı tahmin etmek için "
     "en bilgilendirici pencere."),

 E(start="2022-04-01", end=None, cand=[20, 21, 22],
   label="1 Nisan 2022 — AUF mekanizması + AFL 1.745 → 2.500 TL",
   mechanism="regülasyon", direction="karisik", confidence="kesin",
   arts=[47685, 47689, 47615, 47495],
   description="1 Nisan 2022'de azami fiyat limiti %43 artışla 2.500 TL/MWh'e çıkarıldı ve aynı "
     "tarihte azami uzlaştırma fiyatı (AUF) mekanizması devreye girdi. Düzenleme basında 'elektrikte "
     "tavan fiyat devrede' ve 'elektrik fiyatları desteklenecek' başlıklarıyla yer aldı. AUF, "
     "AFL'den farklı bir araçtır: teklif tavanı değil, üreticinin elde ettiği gelirin geri alınması "
     "esasına dayanır.",
   interpretation="İki aracın aynı gün devreye girmesi Türkiye modelini Avrupa'daki gelir tavanı "
     "uygulamalarından ayırıyor: fiyat hem yukarıdan kesiliyor hem üretici geliri geri alınıyor. Bu "
     "tarihten sonra MCP, üreticinin fiilen aldığı fiyatın üst sınırı olmaktan çıkıyor — teklif "
     "davranışı modellenecekse ikisi ayrılmak zorunda. AFL≠AUF ayrımı literatürde de karıştırılıyor."),

 E(start="2022-05-19", end="2022-06-30", cand=[23, 24, 26],
   label="Mayıs–Haziran 2022 — tavan üç kez yükseldi, bağlayıcılığı düştü",
   mechanism="regülasyon", direction="karisik", confidence="kesin",
   arts=[48414, 48558, 48424, 49021],
   description="19 Mayıs 2022'de azami fiyat limiti 2.750 TL/MWh'e, 1 Haziran'da 3.200 TL'ye, "
     "30 Haziran'da 3.750 TL'ye yükseltildi — altı haftada üç artış. Aynı pencerede MCP dolar bazında "
     "yükseldi ve tavanda geçen saatlerin payı %8,6'ya geriledi.",
   interpretation="Tavan hızla yükseltilirken bağlayıcılığı düştü: sınır fiyatın önünden çekildi. "
     "METODOLOJİK UYARI — aday üretiminde bu olay 'tavan arttı, fiyat %32 düştü' diye görünüyordu; o "
     "rakam artış gününün tek günlük ortalamasıydı (25 saat). Altı haftalık pencerede fiyat %20 arttı. "
     "Tavan etkisi ölçülen pencereye aşırı duyarlı ve tek günlük karşılaştırmalar yanıltıyor; "
     "kataloğun her satırı bu yüzden final penceresinden yeniden ölçüldü."),

 E(start="2022-06-01", end="2022-09-30", cand=[25, 27, 28, 29, 30],
   label="Yaz 2022 — gaz tarifesi zirvesi (20.625 TL) ve MCP rekoru",
   mechanism="maliyet", direction="yukari", confidence="kesin",
   arts=[48595, 49461, 49459, 49974, 49427],
   description="Haziran–Eylül 2022 arasında elektrik üretimine verilen gaz tarifesi 12.525 TL'den "
     "20.625 TL/1000m³'e çıktı; Eylül artışı tek başına %50'ydi. Azami fiyat limiti aynı dönemde "
     "3.200 → 4.800 TL/MWh'e yükseldi. MCP dolar bazında serinin zirvesine ulaştı.",
   interpretation="Maliyet şokunun en büyük olduğu pencere: tarife dört ayda %65 arttı, MCP dolar "
     "zirvesine ulaştı. Dört aylık ortalamada kalıntı küçük ve pozitif — toplamda fiyat temellerle "
     "uyumlu hareket etti. Ama Eylül'deki %50'lik tarife sıçramasının dar çevresinde kalıntı sert "
     "biçimde negatife dönüyor (adaylar #29/#30, ~−32 USD): büyük tarife basamakları MCP'ye anında "
     "geçmiyor, aylar içinde emiliyor. UYARI: crisis_cf_v5 gaz maliyetini güçlü girdi olarak "
     "kullandığı için dar pencere kalıntıları kısmen model kaynaklı olabilir; geçişkenlik katsayısı "
     "gecikme yapısıyla ve düzgün standart hatalarla ayrıca tahmin edilmeli."),

 E(start="2023-01-01", end="2023-04-30", cand=[32, 33, 35, 37, 38, 39, 41],
   label="2023 geri sarımı — tavan 4.800 → 2.600, tarife 20.625 → 10.000 TL",
   mechanism="regülasyon", direction="asagi", confidence="kesin",
   arts=[52175, 52217, 52617, 53075, 53500, 53499],
   description="Ocak–Nisan 2023 arasında hem azami fiyat limiti hem gaz tarifesi art arda indirildi: "
     "tavan 4.800 → 4.200 → 3.650 → 3.050 → 2.600 TL/MWh; gaz tarifesi 20.625 → 18.000 → 15.000 → "
     "12.000 → 10.000 TL/1000m³. Sanayi elektriğine %16 indirim açıklandı.",
   interpretation="Geri sarım penceresinde kontrafaktüel kalıntı ısrarla POZİTİF: fiyat, maliyetteki "
     "düşüşün ima ettiği kadar düşmedi. Büyük tarife ARTIŞLARININ da fiyata tam geçmediği "
     "gözlemiyle ('Ekim 2023' satırı ve Eylül 2022 basamağı) birlikte okununca ortaya asimetri değil "
     "YAPIŞKANLIK çıkıyor: MCP idari maliyetten her iki yönde de daha az oynak. Aynı model uyarısı "
     "burada da geçerli."),

 E(start="2023-01-15", end="2023-03-30", cand=[34, 40],
   label="AUF'a yargı freni ve mekanizmanın uzatılması",
   mechanism="regülasyon", direction="karisik", confidence="kesin",
   arts=[52468, 52429, 52503, 52529, 53483],
   description="Ocak 2023'te EPDK'nin azami uzlaştırma fiyatı uygulamasına yargı freni geldi; "
     "üreticilerin mekanizma kapsamında 1 milyar TL geri ödeme yaptığı bildirildi ve hidroelektrik "
     "kaynak katkı payı hesabında AUF etkisi tartışmaya açıldı. 30 Mart 2023'te mekanizma buna "
     "rağmen uzatıldı; doğalgaz ve kömür santrallerine üretim desteği açıklandı.",
   interpretation="AUF'un hukuken tartışmalı hale geldiği ama uygulanmaya devam ettiği dönem. "
     "Üretici davranışı modellenecekse önemli: mekanizmanın kalıcılığına dair belirsizlik teklif "
     "stratejisini etkiler ve bu belirsizlik fiyat serisinde doğrudan görünmez. Aday üretiminde "
     "hedeflenmemişti — retrieval'ın kendiliğinden çıkardığı tek yeni olay."),

 E(start="2023-02-06", end="2023-02-28", cand=[36],
   label="Kahramanmaraş depremi",
   mechanism="talep", direction="asagi", confidence="kesin",
   arts=[52763, 52829, 52760, 52777, 52833],
   description="6 Şubat 2023 depremlerinde üç il ve dokuz ilçede doğal gaz akışı durdu, bölgede "
     "elektrik kesintileri yaşandı. Akkuyu ve büyük üretim tesislerinde hasar bildirilmedi. Deprem "
     "bölgesinde avans ödemeleri ve teminat yükümlülükleri ertelendi.",
   interpretation="Talep tarafı şoku: bölgesel tüketimin düşmesi fiyatı aşağı çekti. Kontrafaktüel "
     "kalıntı sıfıra yakın — fiyat hareketinin tamamı temeller (düşen yük) tarafından açıklanıyor. "
     "Çok büyük bir olayın fiyatta açıklanmamış bir sapma bırakmaması, olay-etki analizinde negatif "
     "sonucun da bulgu olduğunu gösteren en iyi örnek."),

 E(start="2023-06-01", end="2023-06-30", cand=[42],
   label="Haziran 2023 — fiyatın rejim değiştirmesi",
   mechanism="arz", direction="asagi", confidence="muhtemel",
   arts=[54712, 54701, 54321, 54308],
   description="Haziran 2023'te MCP dolar bazında %28 geriledi. Aynı dönemde Avrupa gaz fiyatları "
     "düşüyordu, lisanssız kurulu güç 9 bin MW'ı aşmıştı ve Türkiye kurulu gücünün %54'ü yenilenebilir "
     "kaynaklara ulaşmıştı. EPDK, PTF tavan fiyatındaki artışın faturalara yansımayacağını açıkladı.",
   interpretation="Tek bir belgelenmiş olaya bağlanamıyor; yenilenebilir payının artışı ve gaz "
     "maliyetindeki gerileme birlikte etkili. Haber retrieval'ı bu pencerede zayıf. Kataloğa olay "
     "olarak değil REJİM GEÇİŞİ olarak girdi — 2022 zirvesinden 2024 platosuna inişin kırılma noktası."),

 E(start="2023-10-01", end=None, cand=[43],
   label="Ekim 2023 — tarife +%20, fiyat temellerin altında kaldı",
   mechanism="maliyet", direction="karisik", confidence="kesin",
   arts=[55873, 55874, 55899],
   description="30 Eylül 2023'te elektrik üretimine verilen doğal gaz tarifesi %20 zamlandı "
     "(12.000 TL/1000m³) ve elektrik fiyatlarına zam açıklandı. Takip eden ayda MCP dolar bazında "
     "%8 arttı.",
   interpretation="%20'lik maliyet artışına karşılık fiyat %8 arttı ve kontrafaktüel kalıntı negatife "
     "döndü — temellerin ima ettiğinden az. Tarife artışının fiyata KISMİ geçtiği en temiz tekil "
     "örnek; 'Nisan 2026' satırıyla birlikte, idari araçların hareketinin fiyat hareketiyle bire bir "
     "eşleşmediğini gösteren çift."),

 E(start="2024-07-01", end="2024-07-31", cand=[45, 46],
   label="Temmuz 2024 — tavan 3.000 TL ve yaz zirvesi",
   mechanism="talep", direction="yukari", confidence="kesin",
   arts=[59281, 59181],
   description="29 Haziran 2024'te azami fiyat limiti 3.000 TL/MWh'e yükseltildi. Temmuz ayında MCP "
     "dolar bazında %22 arttı ve saatlerin dörtte birine yakını tavan fiyatta gerçekleşti.",
   interpretation="Mevsimsel talep zirvesi. Kontrafaktüel kalıntı sıfıra yakın — artış temellerle "
     "açıklanıyor, ayrıca anlatılacak bir olay yok. Yaz zirvelerinin 'olay' sanılmaması için kataloğa "
     "bilinçli olarak bu etiketle girdi."),

 E(start="2025-04-05", end=None, cand=[47, 48],
   label="Nisan 2025 — tavan 3.400 TL ve gecikmeli tarife zammı",
   mechanism="regülasyon", direction="yukari", confidence="kesin",
   arts=[62872, 62869],
   description="5 Nisan 2025'te azami fiyat limiti 3.400 TL/MWh'e çıkarıldı; elektrik üretimine "
     "verilen gaz tarifesi %24 artışla 14.904 TL/1000m³'e yükseldi. Basında zam 'gecikmeli' olarak "
     "nitelendirildi.",
   interpretation="Tavan-tarife eşzamanlılığı 2025'te de sürüyor. 'Gecikmeli zam' nitelemesi idari "
     "kanalın tanımlayıcı özelliğini özetliyor: maliyet değiştiğinde tarife hemen değil, karar "
     "alındığında hareket ediyor."),

 E(start="2025-06-12", end="2025-06-26", cand=[49],
   label="ABD-İsrail-İran savaşı",
   mechanism="jeopolitik", direction="karisik", confidence="muhtemel",
   arts=[63794, 63790, 63789, 63723, 63754],
   description="13 Haziran 2025'te başlayan çatışmada Brent 78 doları aştı, İran Meclisi Hürmüz "
     "Boğazı'na ilişkin karar aldı ve Goldman Sachs jeopolitik riskin petrolü 10 dolar "
     "artırabileceğini bildirdi. Küresel LNG fiyatları %3 arttı.",
   interpretation="2022 Rusya-Ukrayna satırının kontrolü. Enerji basınında anlatı yoğunluğu dönemin "
     "en yüksek seviyesine çıktı (z ≈ +4) ama takip eden haftalarda kontrafaktüel kalıntı yükselmedi "
     "ve BOTAŞ tarifesi hareket etmedi. DİKKAT: pencere içi ortalama kalıntı pozitif, gecikmeli "
     "korelasyon negatif — iki ölçüm farklı şeyler söylüyor, güven bu yüzden 'muhtemel'. Net okuma: "
     "idari vana açılmadığı için jeopolitik şok MCP'ye geçmedi."),

 E(start="2025-07-01", end="2025-07-31", cand=[50],
   label="Temmuz 2025 — yaz zirvesi",
   mechanism="talep", direction="yukari", confidence="kesin",
   arts=[63869, 63760, 63879],
   description="Temmuz 2025'te MCP dolar bazında %32 arttı. Son 6 aylık YEKDEM maliyet öngörüsü "
     "307,17 lira olarak açıklandı; aynı dönemde elektrik borsasına 'esnek tavan' önerisi tartışıldı.",
   interpretation="Bir önceki satırın hemen ardından gelen bu yükseliş savaşa değil yaz talebine ait: "
     "kontrafaktüel kalıntı sıfıra yakın. İkisinin yan yana durması, jeopolitik şoka atfedilebilecek "
     "yükselişin aslında mevsimsel olduğunu gösteriyor — kataloğun kendi içinde kurduğu kontrol."),

 E(start="2026-02-01", end="2026-05-31", cand=[51],
   label="2026 çöküşü — hidro bolluğu ve sıfır fiyatlı saatler",
   mechanism="arz", direction="asagi", confidence="kesin",
   arts=[68154, 68309, 67409, 68478],
   description="Şubat–Mayıs 2026 arasında MCP dolar bazında %57 geriledi. Mayıs'ta gündüz "
     "saatlerinde spot elektrik fiyatının sıfır TL olacağı duyuruldu; artan yağışlarla hidroelektrik "
     "üretimi rekor kırdı. Sektörde 'sıfır fiyatlı saatler' ve negatif fiyat tartışması açıldı.",
   interpretation="Sürücü güneş değil HİDRO (04_zero_price_crisis/01_low_price_regime_analysis.ipynb). "
     "Tavanın bağlamadığı, dolayısıyla idari kanalın fiilen devre dışı kaldığı bir rejim — fiyat "
     "oluşumunun tamamen arz tarafından belirlendiği nadir pencere. Tavan-bağlayıcı 2022 "
     "pencereleriyle doğal karşılaştırma grubu; çalışmanın kimlik stratejisi buna dayanabilir."),

 E(start="2026-04-04", end=None, cand=[52, 53],
   label="Nisan 2026 — tavan +%32 → 4.500 TL, iz bırakmadı",
   mechanism="regülasyon", direction="karisik", confidence="kesin",
   arts=[67633, 67710, 67654, 67642],
   description="2 Nisan 2026'da piyasa takas fiyatı tavanı %32 artışla 4.500 TL/MWh'e yükseltildi; "
     "3 Nisan'da doğal gaz ve elektrik zamlandı (gaz tarifesi 18.000 TL/1000m³). Basında tavan "
     "artışının ardından spot fiyatın yükseldiği bildirildi, ancak aylık ortalama dolar bazında "
     "gerilemeye devam etti.",
   interpretation="Tavan artışının etkisine dair kataloğun EN TEMİZ testi: düşük fiyat rejiminin "
     "ortasında tavan %32 yükseltildi, takip eden ayda fiyat %44 geriledi ve tavanda geçen saat payı "
     "%3'te kaldı. Günlük bazda bildirilen kısa yükselişle aylık ortalamanın ayrışması, tavanın "
     "yalnızca bağlayıcı olduğu saatlerde çalıştığı tezini doğrudan destekliyor — ve tavan etkisini "
     "'tavan değişti mi' üzerinden ölçen çalışmaların (Energy Policy 2022) neden sıfır bulduğunu "
     "açıklıyor: sorun yöntemde değil, tavanın o pencerelerde bağlamamasında."),

 E(start="2026-06-01", end="2026-07-31", cand=[54],
   label="Haziran–Temmuz 2026 — hidro normalleşmesi ve toparlanma",
   mechanism="arz", direction="yukari", confidence="kesin",
   arts=[68399, 68553, 69377, 69107],
   description="Haziran–Temmuz 2026'da MCP dolar bazında dip seviyeden üç katına yakın toparlandı. "
     "Hidroelektrik üretiminin normale dönmesi ve yaz talebi etkili oldu; Temmuz'da elektrik "
     "tarifesine %5,8 zam yapıldı.",
   interpretation="2026 çöküşünün simetrik karşılığı. Kalıntının sıfıra yakın olması hem çöküşün hem "
     "toparlanmanın temellerle (hidro payı, mevsimsel talep) açıklandığını, idari müdahaleye ihtiyaç "
     "duymadığını gösteriyor. Arz kaynaklı rejimin oynaklığı idari rejimden yüksek ama daha öngörülebilir."),
]

# Kataloğa girmeyen adaylar ve gerekçeleri.
DROPPED = {
  1: "Mayıs 2021 tarife +%12 — enflasyon kaynaklı aylık drift, fiyatta karşılığı yok (%+4,5).",
  2: "Haziran 2021 tarife +%5 — drift.",
  5: "Ağustos 2021 tarife +%0 — tarife değişmedi, olay değil.",
  11: "Aralık 2021 kur şoku — retrieval çuvalladı (ilk 10 haberin hiçbiri kur şokuyla ilgili değil) "
      "ve hedef değişken USD bazlı olduğu için TL değer kaybı zaten mekanik olarak içeriliyor.",
  31: "Kasım 2022 tarife +%0 — tarife değişmedi, olay değil.",
  44: "Nisan 2024 MCP −%20 — gerçek bir hareket ama retrieval tamamen gürültü (Çin kömür üretimi). "
      "Belgelenebilir bir olaya bağlanamadığı için kataloğa alınmadı; bahar yenilenebilir dibi olarak "
      "'Haziran 2023' rejim satırıyla aynı aileden.",
}


def measure(eng, start: str, end: str | None) -> dict:
    """Final pencereden ölçüm. Aday dosyasındaki auto_context aynı-tarihli
    çiftlerde birebir aynıydı (pencere aynı); birleştirilmiş olaylar için
    yeniden hesaplamak şart."""
    TZ = "Europe/Istanbul"
    s = pd.Timestamp(start, tz=TZ)
    e = pd.Timestamp(end, tz=TZ) + pd.Timedelta(days=1) if end else s + pd.Timedelta(days=POINT_EVENT_DAYS)
    b0 = s - pd.Timedelta(days=BEFORE_DAYS)

    with eng.connect() as c:
        mcp = pd.read_sql(text("""
            SELECT ts, price_usd, at_cap FROM silver.mcp_with_cap
            WHERE ts >= :b0 AND ts < :e"""), c, params={"b0": b0, "e": e})
        cf = pd.read_sql(text("""
            SELECT ts, residual_usd FROM gold.crisis_counterfactual
            WHERE model_name = :m AND variant = :v AND ts >= :s AND ts < :e"""),
            c, params={"m": CF_MODEL, "v": CF_VARIANT, "s": s, "e": e})

    mcp["ts"] = pd.to_datetime(mcp.ts, utc=True).dt.tz_convert(TZ)
    before = mcp[mcp.ts < s].price_usd.astype(float)
    during = mcp[mcp.ts >= s]
    dur_p = during.price_usd.astype(float)

    r = lambda x, n=1: None if pd.isna(x) else round(float(x), n)
    mb, md = before.mean(), dur_p.mean()
    return dict(
        mcp_usd_before=r(mb), mcp_usd_during=r(md),
        pct_change=r((md - mb) / mb * 100) if mb and not pd.isna(mb) and mb != 0 else None,
        cf_residual_usd=r(cf.residual_usd.astype(float).mean()) if len(cf) else None,
        cf_at_cap_share=r(during.at_cap.mean(), 3) if len(during) else None,
        n_hours=int(len(during)),
    )


def main() -> int:
    dry = "--dry-run" in sys.argv
    eng = get_db_engine()

    # --- article_id doğrulama: eksik kaynak sessizce geçmesin ---
    all_ids = sorted({a for ev in EVENTS for a in ev["arts"]})
    with eng.connect() as c:
        found = pd.read_sql(text("SELECT article_id, url, title, published_at FROM bronze.news_raw "
                                 "WHERE article_id = ANY(:ids)"), c, params={"ids": all_ids})
    missing = set(all_ids) - set(found.article_id.tolist())
    if missing:
        print(f"HATA: bronze.news_raw'da yok: {sorted(missing)}")
        return 1
    url_of = dict(zip(found.article_id, found.url))
    print(f"{len(all_ids)} kaynak haber doğrulandı.\n")

    # --- aday kapsamı: her aday ya bir olayda ya DROPPED'ta olmalı ---
    cands = json.loads((OUT_DIR / "market_events_candidates.json").read_text())
    used = {c for ev in EVENTS for c in ev["cand"]}
    unaccounted = {c["candidate_id"] for c in cands} - used - set(DROPPED)
    if unaccounted:
        print(f"HATA: hiçbir olaya veya drop listesine girmeyen aday: {sorted(unaccounted)}")
        return 1
    overlap = used & set(DROPPED)
    if overlap:
        print(f"HATA: hem kullanılmış hem drop edilmiş aday: {sorted(overlap)}")
        return 1

    rows = []
    for ev in EVENTS:
        m = measure(eng, ev["start"], ev["end"])
        rows.append({**ev, **m, "urls": [url_of[a] for a in ev["arts"]]})

    w = 62
    print(f"{'olay':<{w}} {'önce':>6} {'esna':>6} {'%':>7} {'kalıntı':>8} {'tavan%':>7} {'saat':>6}")
    print("-" * (w + 45))
    for r in rows:
        f = lambda v, d="—": d if v is None else v
        print(f"{r['label'][:w]:<{w}} {f(r['mcp_usd_before']):>6} {f(r['mcp_usd_during']):>6} "
              f"{f(r['pct_change']):>7} {f(r['cf_residual_usd']):>8} {f(r['cf_at_cap_share']):>7} "
              f"{r['n_hours']:>6}")

    if dry:
        print("\n--dry-run: DB'ye yazılmadı.")
        return 0

    with eng.begin() as c:
        for stmt in DDL.strip().split(";\n"):
            if stmt.strip():
                c.execute(text(stmt))
        for r in rows:
            c.execute(text("""
                INSERT INTO silver.market_events
                  (start_date, end_date, label, mechanism, direction, description,
                   interpretation, confidence, source_article_ids, source_urls, candidate_ids,
                   mcp_usd_before, mcp_usd_during, pct_change, cf_residual_usd,
                   cf_at_cap_share, n_hours)
                VALUES
                  (:start, :end, :label, :mechanism, :direction, :description,
                   :interpretation, :confidence, :arts, :urls, :cand,
                   :mcp_usd_before, :mcp_usd_during, :pct_change, :cf_residual_usd,
                   :cf_at_cap_share, :n_hours)"""), r)
    print(f"\n→ silver.market_events: {len(rows)} satır yazıldı.")

    # --- küratörleme izi: hangi aday nereye gitti ---
    trace = {
        "curated_at": pd.Timestamp.now(tz="Europe/Istanbul").isoformat(),
        "n_candidates": len(cands), "n_events": len(rows), "n_dropped": len(DROPPED),
        "events": [{k: r[k] for k in ("start", "end", "label", "mechanism", "direction",
                                      "confidence", "cand", "arts", "description",
                                      "interpretation")} for r in rows],
        "dropped": DROPPED,
    }
    out = OUT_DIR / "market_events_curated.json"
    out.write_text(json.dumps(trace, ensure_ascii=False, indent=2))
    print(f"→ {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
