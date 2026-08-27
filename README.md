# Enerji Fiyat Tahmini — Deney & Araştırma

`enerji_fiyat_tahmini` canlı reposunun kardeşi. Model deneyleri, literatür
taraması ve kriz/olay istihbarat araştırması burada; canlı dashboard/ETL/model
orada. İkisi de aynı PostgreSQL veritabanına bağlanıyor.

Detay için `CLAUDE.md`'ye bak.

## Kurulum

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`.env` canlı repodakiyle aynı DB credential'larını içeriyor (bu repoya özel
kopyalandı, git'e girmiyor).
