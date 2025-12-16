# config.py
import os

CACHE_TTL_SECONDS = 3 * 3600  # 3 hours

TOP_N_DEFAULT = 5
LOOKBACK_DAYS_NEWS_DEFAULT = 7
LOOKBACK_DAYS_PRICE_DEFAULT = 30
PREFILTER_N_DEFAULT = 25
GNEWS_MAX_TICKERS_DEFAULT = 12
MIN_AVG_DAILY_VOLUME_DEFAULT = 2_000_000

WEIGHT_TONE_DEFAULT = 0.55
WEIGHT_NEWS_VOL_DEFAULT = 0.15
WEIGHT_MOMENTUM_DEFAULT = 0.30

GDELT_WORKERS_DEFAULT = 10

WATCHLIST = [
    "BBCA","BBRI","BMRI","BBNI","BBTN","BRIS","BNGA","BTPS","MEGA","NISP",
    "TLKM","ISAT","EXCL","MTEL","TBIG","TOWR",
    "ASII","UNVR","ICBP","INDF","MYOR","AMRT","SIDO","KLBF","MAPI","ERAA",
    "ADRO","PTBA","ITMG","HRUM","INDY","MEDC","PGAS","PGEO",
    "ANTM","TINS","INCO","MDKA","NCKL",
    "JSMR","WIKA","PTPP","ADHI","WSKT",
    "BSDE","CTRA","PWON","SMRA","ASRI",
    "GOTO","BUKA","SRTG","BRPT","EMTK",
    "MIKA","SILO","HEAL",
]

# Keep aliases specific. Avoid generic single-word aliases like "Astra".
# Acronyms like BCA/BRI/BNI are allowed (all caps).
ALIASES = {
    "BBCA": ["Bank Central Asia", "BCA", "PT Bank Central Asia"],
    "BBRI": ["Bank Rakyat Indonesia", "BRI", "PT Bank Rakyat Indonesia"],
    "BMRI": ["Bank Mandiri", "Mandiri", "PT Bank Mandiri"],
    "BBNI": ["Bank Negara Indonesia", "BNI", "PT Bank Negara Indonesia"],
    "BBTN": ["Bank Tabungan Negara", "BTN", "PT Bank Tabungan Negara"],
    "BRIS": ["Bank Syariah Indonesia", "BSI", "PT Bank Syariah Indonesia"],
    "BNGA": ["CIMB Niaga", "PT Bank CIMB Niaga"],
    "BTPS": ["Bank BTPN Syariah", "BTPN Syariah"],
    "MEGA": ["Bank Mega", "PT Bank Mega"],
    "NISP": ["OCBC NISP", "PT Bank OCBC NISP"],

    "TLKM": ["Telkom Indonesia", "PT Telkom Indonesia", "Telkom"],
    "ISAT": ["Indosat Ooredoo Hutchison", "Indosat", "PT Indosat Ooredoo Hutchison"],
    "EXCL": ["XL Axiata", "PT XL Axiata"],
    "MTEL": ["Dayamitra Telekomunikasi", "Mitratel", "PT Dayamitra Telekomunikasi"],
    "TBIG": ["Tower Bersama Infrastructure", "PT Tower Bersama Infrastructure"],
    "TOWR": ["Sarana Menara Nusantara", "PT Sarana Menara Nusantara"],

    "ASII": ["Astra International", "PT Astra International"],  # do not add "Astra"
    "UNVR": ["Unilever Indonesia", "PT Unilever Indonesia"],
    "ICBP": ["Indofood CBP Sukses Makmur", "PT Indofood CBP Sukses Makmur"],
    "INDF": ["Indofood Sukses Makmur", "PT Indofood Sukses Makmur"],
    "MYOR": ["Mayora Indah", "PT Mayora Indah"],
    "AMRT": ["Sumber Alfaria Trijaya", "PT Sumber Alfaria Trijaya", "Alfamart"],
    "SIDO": ["Sido Muncul", "PT Industri Jamu dan Farmasi Sido Muncul", "Sidomuncul"],
    "KLBF": ["Kalbe Farma", "PT Kalbe Farma", "Kalbe"],
    "MAPI": ["Mitra Adiperkasa", "PT Mitra Adiperkasa"],
    "ERAA": ["Erajaya Swasembada", "PT Erajaya Swasembada", "Erajaya"],

    "ADRO": ["Adaro Energy Indonesia", "PT Adaro Energy Indonesia", "Adaro"],
    "PTBA": ["Bukit Asam", "PT Bukit Asam"],
    "ITMG": ["Indo Tambangraya Megah", "PT Indo Tambangraya Megah"],
    "HRUM": ["Harum Energy", "PT Harum Energy"],
    "INDY": ["Indika Energy", "PT Indika Energy"],
    "MEDC": ["Medco Energi Internasional", "PT Medco Energi Internasional", "Medco"],
    "PGAS": ["Perusahaan Gas Negara", "PT Perusahaan Gas Negara", "PGN"],
    "PGEO": ["Pertamina Geothermal Energy", "PT Pertamina Geothermal Energy"],

    "ANTM": ["Aneka Tambang", "PT Aneka Tambang", "Antam"],
    "TINS": ["Timah", "PT Timah"],
    "INCO": ["Vale Indonesia", "PT Vale Indonesia"],
    "MDKA": ["Merdeka Copper Gold", "PT Merdeka Copper Gold"],
    "NCKL": ["Trimegah Bangun Persada", "PT Trimegah Bangun Persada", "Harita Nickel"],

    "JSMR": ["Jasa Marga", "PT Jasa Marga", "Jasa Marga"],
    "WIKA": ["Wijaya Karya", "PT Wijaya Karya", "WIKA"],
    "PTPP": ["Pembangunan Perumahan", "PT Pembangunan Perumahan", "PT PP"],
    "ADHI": ["Adhi Karya", "PT Adhi Karya", "Adhikarya"],
    "WSKT": ["Waskita Karya", "PT Waskita Karya", "Waskita"],

    "BSDE": ["Bumi Serpong Damai", "PT Bumi Serpong Damai"],
    "CTRA": ["Ciputra Development", "PT Ciputra Development", "Ciputra"],
    "PWON": ["Pakuwon Jati", "PT Pakuwon Jati", "Pakuwon"],
    "SMRA": ["Summarecon Agung", "PT Summarecon Agung", "Summarecon"],
    "ASRI": ["Alam Sutera Realty", "PT Alam Sutera Realty", "Alam Sutera"],

    "GOTO": ["GoTo Gojek Tokopedia", "PT GoTo Gojek Tokopedia", "Gojek", "Tokopedia", "GOTO"],
    "BUKA": ["Bukalapak", "PT Bukalapak"],
    "SRTG": ["Saratoga Investama Sedaya", "PT Saratoga Investama Sedaya", "Saratoga"],
    "BRPT": ["Barito Pacific", "PT Barito Pacific"],
    "EMTK": ["Elang Mahkota Teknologi", "PT Elang Mahkota Teknologi", "Emtek"],

    "MIKA": ["Mitra Keluarga Karyasehat", "PT Mitra Keluarga Karyasehat", "Mitra Keluarga"],
    "SILO": ["Siloam International Hospitals", "PT Siloam International Hospitals", "Siloam"],
    "HEAL": ["Medikaloka Hermina", "PT Medikaloka Hermina", "Hermina"],
}

# Optional Google Sheet source for watchlists
GOOGLE_SHEET_ID = os.getenv("GOOGLE_SHEET_ID", "").strip()
GOOGLE_SERVICE_ACCOUNT_FILE = os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE", "").strip()
GOOGLE_SERVICE_ACCOUNT_JSON = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "").strip()
