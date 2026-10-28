import inspect
import os
import textwrap
import time

import numpy as np
import pandas as pd
import requests
import streamlit as st
import folium
import altair as alt
from sklearn.cluster import KMeans
from streamlit_folium import st_folium

import algoritmalar as alg
from algoritmalar import (
    ALGORITMALAR,
    TesisModeli,
    acgozlu_ekleme,
    calistir,
    haversine_matrisi,
)

# ---------------------------------------------------------------
# Sayfa ayarları
# ---------------------------------------------------------------
st.set_page_config(page_title="Ankara Hastane Yerleşimi", page_icon="🏥", layout="wide")

st.title("🏥 Ankara Hastane Yerleşimi Optimizasyonu")
st.write(
    "Tesis yerleştirme problemi: aday boş arazilerden hangilerinde hastane açılırsa "
    "**yapım maliyeti + ulaşım maliyeti** toplamı en düşük olur? "
    "Kapasitesiz ve kapasiteli model, GA / DE / PSO algoritmalarıyla çözülür."
)

ANKARA_CENTER = [39.9334, 32.8597]
KIZILAY = (39.9208, 32.8541)          # "şehir merkezi" olarak kabul edilen nokta

# Yaklaşık Ankara kentsel sınırları (güney, batı, kuzey, doğu)
SOUTH, WEST, NORTH, EAST = 39.75, 32.45, 40.10, 33.10
BBOX = f"{SOUTH},{WEST},{NORTH},{EAST}"

ENDPOINTS = [
    "https://overpass.private.coffee/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    "https://overpass-api.de/api/interpreter",
]
HEADERS = {"User-Agent": "AnkaraHospitalOptimization/1.0"}

# İndirilen ham veriler bu klasöre kaydedilir; bir kez inince bir daha indirilmez
VERI_KLASORU = os.path.join(os.path.dirname(os.path.abspath(__file__)), "veri_onbellek")
os.makedirs(VERI_KLASORU, exist_ok=True)


# ---------------------------------------------------------------
# Veri çekme
# ---------------------------------------------------------------
def overpass_sorgu(query, timeout=120, tur=2):
    """Sunucuları sırayla dener (tur kez), ilk çalışanın JSON cevabını döndürür."""
    hatalar = []
    for deneme in range(tur):
        for ep in ENDPOINTS:
            try:
                r = requests.post(ep, data={"data": query}, headers=HEADERS, timeout=timeout)
                r.raise_for_status()
                return r.json()
            except (requests.RequestException, ValueError) as e:
                hatalar.append(f"{ep}: {str(e)[:120]}")
        if deneme < tur - 1:
            time.sleep(3)
    raise RuntimeError("Veri alınamadı. Denenen sunucular:\n" + "\n".join(hatalar))


def bbox_parcalari(n_lat=2, n_lon=2):
    """Büyük bölgeyi küçük kutulara böler (her sorgu hafif kalsın diye)."""
    lats = np.linspace(SOUTH, NORTH, n_lat + 1)
    lons = np.linspace(WEST, EAST, n_lon + 1)
    return [f"{lats[i]},{lons[j]},{lats[i + 1]},{lons[j + 1]}"
            for i in range(n_lat) for j in range(n_lon)]


def diskte_sakla(dosya_adi, uretici):
    """Dosya varsa diskten okur, yoksa üretip kaydeder."""
    yol = os.path.join(VERI_KLASORU, dosya_adi)
    if os.path.exists(yol):
        try:
            return pd.read_csv(yol)
        except Exception:
            pass
    df = uretici()
    df.to_csv(yol, index=False)
    return df


def polygon_alan_m2(geom):
    """Overpass geometry listesinden yaklaşık alanı m² olarak hesaplar."""
    lat = np.array([p["lat"] for p in geom])
    lon = np.array([p["lon"] for p in geom])
    lat0 = np.radians(lat.mean())
    y = (lat - lat.mean()) * 111_320
    x = (lon - lon.mean()) * 111_320 * np.cos(lat0)
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def _aday_parca_indir(kutu):
    """Bir kutudaki boş arazi poligonları (alan >= 100 m²)."""
    query = f"""
    [out:json][timeout:60];
    (
      way["landuse"~"^(brownfield|greenfield|meadow)$"]({kutu});
      way["natural"~"^(grassland|scrub|heath)$"]({kutu});
    );
    out geom tags;
    """
    data = overpass_sorgu(query, timeout=75, tur=1)
    rows = []
    for el in data.get("elements", []):
        geom = el.get("geometry")
        if not geom or len(geom) < 4:
            continue
        alan = polygon_alan_m2(geom)
        if alan < 100:
            continue
        tags = el.get("tags", {})
        rows.append({
            "Enlem": float(np.mean([p["lat"] for p in geom])),
            "Boylam": float(np.mean([p["lon"] for p in geom])),
            "Alan_m2": int(round(alan)),
            "Tür": tags.get("landuse") or tags.get("natural", "-"),
        })
    return pd.DataFrame(rows, columns=["Enlem", "Boylam", "Alan_m2", "Tür"])


def _yerlesim_parca_indir(kutu):
    """Bir kutudaki yerleşim alanları: sadece sınır kutusu istenir (hafif sorgu).
    Alan ~ kutu alanı x 0.6 (poligon kutuyu tam doldurmaz)."""
    query = f"""
    [out:json][timeout:60];
    way["landuse"="residential"]({kutu});
    out bb qt;
    """
    data = overpass_sorgu(query, timeout=75, tur=1)
    rows = []
    for el in data.get("elements", []):
        b = el.get("bounds")
        if not b:
            continue
        lat0 = (b["minlat"] + b["maxlat"]) / 2
        lon0 = (b["minlon"] + b["maxlon"]) / 2
        dy = (b["maxlat"] - b["minlat"]) * 111_320
        dx = (b["maxlon"] - b["minlon"]) * 111_320 * np.cos(np.radians(lat0))
        alan = dx * dy * 0.6
        if alan < 2000:
            continue
        rows.append({"Enlem": float(lat0), "Boylam": float(lon0), "Alan_m2": float(alan)})
    return pd.DataFrame(rows, columns=["Enlem", "Boylam", "Alan_m2"])


def parcali_ham_veri(onek, indir_fonksiyonu):
    """Bölgeyi 4 parçaya bölüp her parçayı ayrı kaydeder. Bir parça hata verirse
    sayfayı yenileyince indirilmiş parçalar tekrar indirilmez, kalan yerden devam edilir."""
    parcalar = []
    for k, kutu in enumerate(bbox_parcalari(2, 2)):
        parcalar.append(diskte_sakla(f"{onek}_{k}.csv", lambda kutu=kutu: indir_fonksiyonu(kutu)))
    df = pd.concat(parcalar, ignore_index=True)
    df = df.drop_duplicates(subset=["Enlem", "Boylam", "Alan_m2"]).reset_index(drop=True)
    if df.empty:
        raise RuntimeError("Veri alınamadı (sonuç boş geldi).")
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def get_candidate_lands(esik_m2, n_aday):
    """Boş arazileri alan eşiğine göre filtreler, n_aday adede indirir."""
    ham = parcali_ham_veri("aday", _aday_parca_indir)
    df = ham[ham["Alan_m2"] >= esik_m2].reset_index(drop=True)
    if df.empty:
        raise RuntimeError("Eşiği geçen boş arazi bulunamadı. Alan eşiğini düşür.")
    toplam_uygun = len(df)

    if toplam_uygun > n_aday:
        km = KMeans(n_clusters=n_aday, random_state=42, n_init=10)
        km.fit(df[["Enlem", "Boylam"]])
        secilen = []
        for c in range(n_aday):
            idx = np.where(km.labels_ == c)[0]
            d = ((df.iloc[idx][["Enlem", "Boylam"]].values - km.cluster_centers_[c]) ** 2).sum(axis=1)
            secilen.append(idx[d.argmin()])
        df = df.iloc[secilen]

    df = df.reset_index(drop=True)
    df.index.name = "Aday_ID"
    return df, toplam_uygun


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def get_demand_points(n_talep):
    """Yerleşim alanlarını alan ağırlıklı KMeans ile n_talep talep noktasına indirger."""
    raw = parcali_ham_veri("yerlesim", _yerlesim_parca_indir)
    if len(raw) > n_talep:
        km = KMeans(n_clusters=n_talep, random_state=42, n_init=3)
        km.fit(raw[["Enlem", "Boylam"]], sample_weight=raw["Alan_m2"])
        agirlik = np.bincount(km.labels_, weights=raw["Alan_m2"], minlength=n_talep)
        df = pd.DataFrame(km.cluster_centers_, columns=["Enlem", "Boylam"])
        df["Agirlik"] = agirlik
    else:
        df = raw.rename(columns={"Alan_m2": "Agirlik"})[["Enlem", "Boylam", "Agirlik"]].copy()
    df = df[df["Agirlik"] > 0].reset_index(drop=True)
    df.index.name = "Talep_ID"
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def get_hospitals():
    """Mevcut hastaneler (sadece karşılaştırma için)."""
    query = f"""
    [out:json][timeout:60];
    (
      node["amenity"="hospital"]({BBOX});
      way["amenity"="hospital"]({BBOX});
      relation["amenity"="hospital"]({BBOX});
    );
    out center tags;
    """
    data = overpass_sorgu(query, timeout=60)
    rows, seen = [], set()
    for el in data.get("elements", []):
        tags = el.get("tags", {})
        name = tags.get("name", "İsimsiz hastane")
        if el["type"] == "node":
            lat, lon = el.get("lat"), el.get("lon")
        else:
            c = el.get("center", {})
            lat, lon = c.get("lat"), c.get("lon")
        if lat is None or lon is None:
            continue
        key = (name.strip().lower(), round(lat, 5), round(lon, 5))
        if key in seen:
            continue
        seen.add(key)
        rows.append({"Hastane": name, "Enlem": lat, "Boylam": lon})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------
# Kenar çubuğu
# ---------------------------------------------------------------
sb = st.sidebar
sb.header("Ayarlar")

sb.subheader("1) Aday araziler")
esik = sb.number_input("Hastane için gereken arazi (m²)", 100, 50_000, 1000, step=100)
n_aday = sb.slider("Aday nokta sayısı", 20, 400, 200, step=10)

sb.subheader("2) Talep noktaları (nüfus)")
n_talep = sb.slider("Talep noktası sayısı", 50, 600, 300, step=10)
toplam_nufus = sb.number_input("Toplam nüfus (kişi)", 100_000, 20_000_000, 5_000_000, step=100_000)

sb.subheader("3) Maliyet modeli (milyon TL)")
insaat = sb.number_input("Sabit inşaat bedeli (milyon TL)", 1.0, 100_000.0, 100.0, step=10.0)
arsa_taban = sb.number_input("Arsa taban fiyatı (TL/m²)", 100, 1_000_000, 20_000, step=1000)
k_merkez = sb.slider("Merkez etkisi k (0 = hepsi aynı fiyat)", 0.0, 10.0, 3.0, step=0.5)
ulasim_kats = sb.number_input(
    "Ulaşım maliyeti (TL / kişi·km, planlama süresi boyunca)", 1.0, 100_000.0, 500.0, step=50.0
)

sb.subheader("4) Model türü")
model_turu = sb.radio("Kapasite", ["Kapasitesiz", "Kapasiteli"], horizontal=True)
kapasite = None
if model_turu == "Kapasiteli":
    kapasite = sb.number_input("Hastane kapasitesi (kişi)", 10_000, 5_000_000, 300_000, step=10_000)

sb.subheader("5) Algoritma ayarları")
sb.caption("Algoritmaları 'Optimizasyon' sekmesindeki düğmelerle tek tek çalıştırırsın.")
pop_boyut = sb.slider("Popülasyon / sürü büyüklüğü", 10, 100, 40, step=5)
iterasyon = sb.slider("İterasyon sayısı", 20, 400, 150, step=10)
n_calisma = sb.slider("Tekrar sayısı (rastgele başlangıç)", 1, 10, 3)

with sb.expander("Algoritma parametreleri"):
    ga_mut = st.slider("GA: mutasyon oranı", 0.001, 0.05, 0.01, step=0.001, format="%.3f")
    ga_cx = st.slider("GA: çaprazlama oranı", 0.5, 1.0, 0.9, step=0.05)
    de_F = st.slider("DE: F (ölçek)", 0.1, 1.5, 0.6, step=0.05)
    de_CR = st.slider("DE: CR (çaprazlama)", 0.1, 1.0, 0.9, step=0.05)
    pso_c1 = st.slider("PSO: c1 (bireysel)", 0.5, 2.5, 1.5, step=0.1)
    pso_c2 = st.slider("PSO: c2 (sürü)", 0.5, 2.5, 1.5, step=0.1)
    seed = st.number_input("Rastgelelik tohumu (seed)", 0, 10_000, 1)

def yerel_mi():
    """Site kendi bilgisayarında (localhost) mı çalışıyor? Yayındaki sitede veri silme düğmesi gizlenir."""
    try:
        url = str(st.context.url)
        return "localhost" in url or "127.0.0.1" in url
    except Exception:
        return True


if yerel_mi():
    sb.divider()
    if sb.button("Verileri yeniden indir"):
        for f in os.listdir(VERI_KLASORU):
            os.remove(os.path.join(VERI_KLASORU, f))
        st.cache_data.clear()
        st.rerun()
    sb.caption("Veriler bir kez indirilip 'veri_onbellek' klasörüne kaydedilir.")

ALGO_PARAMS = {
    "Genetik Algoritma (GA)": dict(mutasyon=ga_mut, caprazlama=ga_cx),
    "Diferansiyel Evrim (DE)": dict(F=de_F, CR=de_CR),
    "Parçacık Sürü (PSO)": dict(c1=pso_c1, c2=pso_c2),
}

# ---------------------------------------------------------------
# Veriyi hazırla
# ---------------------------------------------------------------
try:
    with st.spinner("Boş araziler hazırlanıyor (ilk seferde indirilir, birkaç dakika sürebilir)..."):
        adaylar, toplam_uygun = get_candidate_lands(int(esik), int(n_aday))
except Exception as error:
    st.error("Aday arazi verisi alınamadı (OpenStreetMap sunucuları yoğun olabilir). "
             "Sayfayı yenile (indirilen parçalar kaydedildi, kaldığı yerden devam eder).")
    st.code(str(error))
    st.stop()

try:
    with st.spinner("Yerleşim alanları (nüfus) hazırlanıyor (ilk seferde indirilir)..."):
        talep = get_demand_points(int(n_talep))
except Exception as error:
    st.error("Yerleşim alanı (nüfus) verisi alınamadı (OpenStreetMap sunucuları yoğun olabilir). "
             "Sayfayı yenile (indirilen parçalar kaydedildi, kaldığı yerden devam eder).")
    st.code(str(error))
    st.stop()

nufus = toplam_nufus * talep["Agirlik"].values / talep["Agirlik"].sum()
talep = talep.assign(Nufus=np.round(nufus).astype(int))

D = haversine_matrisi(adaylar["Enlem"], adaylar["Boylam"], talep["Enlem"], talep["Boylam"])
merkeze_km = haversine_matrisi(adaylar["Enlem"], adaylar["Boylam"], [KIZILAY[0]], [KIZILAY[1]])[:, 0]
yakinlik = 1.0 - merkeze_km / max(merkeze_km.max(), 1e-9)          # 1 = merkez, 0 = en uzak aday
arsa_fiyat = arsa_taban * (1.0 + k_merkez * yakinlik)               # TL/m²
yapim = insaat + esik * arsa_fiyat / 1e6                            # milyon TL

model = TesisModeli(D, talep["Nufus"].values.astype(float), yapim, ulasim_kats, kapasite)

adaylar_tablo = adaylar.copy()
adaylar_tablo["Merkeze_uzaklik_km"] = np.round(merkeze_km, 2)
adaylar_tablo["Arsa_fiyati_TL_m2"] = np.round(arsa_fiyat).astype(int)
adaylar_tablo["Yapim_maliyeti_milyonTL"] = np.round(yapim, 1)
adaylar_tablo["Tek_basina_ulasim_milyonTL"] = np.round((D @ model.nufus) * model.c, 0)

model_imza = dict(
    esik=int(esik), n_aday=int(n_aday), n_talep=int(n_talep), toplam_nufus=int(toplam_nufus),
    insaat=float(insaat), arsa_taban=int(arsa_taban), k=float(k_merkez), ulasim=float(ulasim_kats),
    kapasite=None if kapasite is None else int(kapasite),
)


# ---------------------------------------------------------------
# Yardımcı: harita çizimleri
# ---------------------------------------------------------------
def aday_haritasi(goster_talep, goster_mevcut):
    m = folium.Map(location=ANKARA_CENTER, zoom_start=11, tiles="OpenStreetMap")
    if goster_talep:
        for tid, t in talep.iterrows():
            folium.CircleMarker(
                [t["Enlem"], t["Boylam"]], radius=float(min(9, max(2, np.sqrt(t["Nufus"]) / 40))),
                color="#e8890c", weight=1, fill=True, fill_color="#f5a623", fill_opacity=0.5,
                tooltip=f"Talep {int(tid)} | {int(t['Nufus']):,} kişi",
            ).add_to(m)
    for i, r in adaylar_tablo.iterrows():
        folium.CircleMarker(
            [r["Enlem"], r["Boylam"]], radius=6, color="darkred", weight=1,
            fill=True, fill_color="red", fill_opacity=0.85,
            tooltip=f"Aday {i} | yapım {r['Yapim_maliyeti_milyonTL']:,.0f} milyon TL",
            popup=folium.Popup(
                f"<b>Aday {i}</b><br>Alan: {r['Alan_m2']:,} m²<br>Tür: {r['Tür']}<br>"
                f"Merkeze: {r['Merkeze_uzaklik_km']} km<br>"
                f"Yapım: {r['Yapim_maliyeti_milyonTL']:,.1f} milyon TL", max_width=250),
        ).add_to(m)
    if goster_mevcut:
        try:
            for _, h in get_hospitals().iterrows():
                folium.Marker([h["Enlem"], h["Boylam"]], tooltip=h["Hastane"],
                              icon=folium.Icon(color="blue", icon="plus-sign")).add_to(m)
        except Exception as e:
            st.warning(f"Mevcut hastaneler alınamadı: {e}")
    return m


def sonuc_haritasi(sadaylar, stalep, d, goster_talep, goster_cizgi):
    m = folium.Map(location=ANKARA_CENTER, zoom_start=11, tiles="OpenStreetMap")
    acik = d["idx"]
    acik_set = set(acik.tolist())

    for j in range(len(sadaylar)):
        if j in acik_set:
            continue
        folium.CircleMarker([sadaylar["Enlem"].iat[j], sadaylar["Boylam"].iat[j]], radius=3,
                            color="gray", weight=1, fill=True, fill_opacity=0.6,
                            tooltip=f"Aday {j} (kapalı)").add_to(m)

    if goster_cizgi:
        for i in range(len(stalep)):
            f = int(d["flow"][:, i].argmax())
            j = int(acik[f])
            folium.PolyLine(
                [[stalep["Enlem"].iat[i], stalep["Boylam"].iat[i]],
                 [sadaylar["Enlem"].iat[j], sadaylar["Boylam"].iat[j]]],
                weight=1, color="#4c78a8", opacity=0.35).add_to(m)

    if goster_talep:
        for i in range(len(stalep)):
            nf = stalep["Nufus"].iat[i]
            folium.CircleMarker([stalep["Enlem"].iat[i], stalep["Boylam"].iat[i]],
                                radius=float(min(9, max(2, np.sqrt(nf) / 40))),
                                color="#e8890c", weight=1, fill=True, fill_color="#f5a623",
                                fill_opacity=0.5, tooltip=f"Talep {i} | {int(nf):,} kişi").add_to(m)

    for f, j in enumerate(acik):
        metin = (f"<b>Aday {j}</b><br>Yapım: {d['yapim_kalem'][f]:,.1f} milyon TL<br>"
                 f"Hizmet: {d['hizmet_nufus'][f]:,.0f} kişi<br>"
                 f"Ort. uzaklık: {d['ort_uzaklik_km'][f]:.2f} km")
        if d["doluluk"] is not None:
            metin += f"<br>Doluluk: %{100 * d['doluluk'][f]:.0f}"
        folium.CircleMarker([sadaylar["Enlem"].iat[j], sadaylar["Boylam"].iat[j]], radius=9,
                            color="darkgreen", weight=2, fill=True, fill_color="#2ecc71",
                            fill_opacity=0.95, tooltip=f"Aday {j} (AÇIK)",
                            popup=folium.Popup(metin, max_width=260)).add_to(m)
    return m


def sonuc_tablosu(sadaylar, d):
    acik = d["idx"]
    tablo = pd.DataFrame({
        "Aday_ID": acik,
        "Enlem": sadaylar["Enlem"].values[acik].round(5),
        "Boylam": sadaylar["Boylam"].values[acik].round(5),
        "Yapim_maliyeti_milyonTL": d["yapim_kalem"].round(1),
        "Hizmet_verilen_nufus": d["hizmet_nufus"].round(0).astype(int),
        "Ortalama_uzaklik_km": d["ort_uzaklik_km"].round(2),
        "Ulasim_maliyeti_milyonTL": d["ulasim_kalem"].round(1),
    })
    if d["doluluk"] is not None:
        tablo["Doluluk_%"] = (100 * d["doluluk"]).round(1)
    return tablo.sort_values("Hizmet_verilen_nufus", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------
# Sekmeler
# ---------------------------------------------------------------
sekme1, sekme2, sekme3 = st.tabs(["1) Adaylar ve maliyetler", "2) Optimizasyon", "3) Yöntem ve varsayımlar"])

# ===================== SEKME 1 =====================
with sekme1:
    st.subheader("Aday boş araziler (kırmızı)")
    c1, c2 = st.columns(2)
    goster_talep1 = c1.checkbox("Talep noktalarını göster (turuncu, büyüklük = nüfus)", value=True)
    goster_mevcut = c2.checkbox("Mevcut hastaneleri göster (mavi)", value=False)

    st_folium(aday_haritasi(goster_talep1, goster_mevcut), use_container_width=True,
              height=600, returned_objects=[], key="aday_harita")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Eşiği geçen toplam arazi", toplam_uygun)
    m2.metric("Aday sayısı", len(adaylar))
    m3.metric("Talep noktası", len(talep))
    m4.metric("Toplam nüfus", f"{int(talep['Nufus'].sum()):,}")

    st.subheader("Tablo 1: Aday arazilerin maliyetleri")
    st.caption(
        "Yapım maliyeti adaya özeldir (merkeze yaklaştıkça arsa pahalıdır). "
        "'Tek başına ulaşım maliyeti', yalnızca o hastane açık olsaydı tüm nüfusun toplam "
        "ulaşım maliyetidir (kapasite ihmal edilir); düşük değer = nüfusa göre merkezi konum. "
        "Gerçek ulaşım maliyeti hangi hastanelerin açık olduğuna bağlıdır, Sekme 2'de görünür."
    )
    st.dataframe(adaylar_tablo)
    st.download_button("Tablo 1'i CSV olarak indir", adaylar_tablo.to_csv().encode("utf-8-sig"),
                       file_name="aday_noktalar_maliyet.csv", mime="text/csv")

# ===================== SEKME 2 =====================
with sekme2:
    st.subheader(f"Optimizasyon ({model_turu})")

    kapasite_ok = model.kapasite_yeterli_mi()
    if not kapasite_ok:
        st.error(
            f"Kapasite yetersiz: {len(adaylar)} hastane × {int(kapasite):,} kişi = "
            f"{len(adaylar) * int(kapasite):,} < toplam nüfus {int(model.toplam_nufus):,}. "
            "Kapasiteyi artır ya da aday sayısını yükselt."
        )

    st.caption(
        "Algoritmaları tek tek çalıştırabilirsin. Çalıştırdığın her algoritmanın sonucu saklanır ve "
        "aşağıda karşılaştırılır. Aynı algoritmayı yeniden çalıştırırsan yalnızca onun sonucu güncellenir."
    )

    kisa = {ad: ad.split("(")[-1].rstrip(")") for ad in ALGORITMALAR}      # GA, DE, PSO
    kolonlar = st.columns(len(ALGORITMALAR) + 2)
    hedef = []
    for kol, ad in zip(kolonlar, ALGORITMALAR):
        if kol.button(f"▶ {kisa[ad]} çalıştır", key=f"btn_{ad}", disabled=not kapasite_ok,
                      help=f"Sadece {ad} çalışır"):
            hedef = [ad]
    if kolonlar[-2].button("▶▶ Hepsini çalıştır", key="btn_hepsi", type="primary", disabled=not kapasite_ok,
                           help="Üç algoritmayı sırayla çalıştırır"):
        hedef = list(ALGORITMALAR)
    if kolonlar[-1].button("🗑 Sonuçları temizle", key="btn_temizle"):
        st.session_state.pop("sonuc", None)
        st.rerun()

    if hedef:
        sonuc = st.session_state.get("sonuc")
        if sonuc is None or sonuc["model_imza"] != model_imza:
            with st.spinner("Referans çözümler (açgözlü ekleme) hesaplanıyor..."):
                ac_maske, _ = acgozlu_ekleme(model)
            sonuc = {
                "model_imza": model_imza, "model": model, "adaylar": adaylar_tablo, "talep": talep,
                "algo": {}, "hepsi": np.ones(model.n, bool), "acgozlu": ac_maske, "tur": model_turu,
            }
            st.session_state["sonuc"] = sonuc

        toplam_adim = len(hedef) * n_calisma
        ilerleme = st.progress(0.0, text="Başlıyor...")
        adim = 0
        for ad in hedef:
            kosular = []
            for r in range(n_calisma):
                ilerleme.progress(adim / toplam_adim, text=f"{ad}: tekrar {r + 1}/{n_calisma}")
                kosular.append(calistir(ALGORITMALAR[ad], model, int(seed) + r,
                                        pop=int(pop_boyut), iters=int(iterasyon), **ALGO_PARAMS[ad]))
                adim += 1
            sonuc["algo"][ad] = {
                "kosular": kosular,
                "ayar": {"pop": int(pop_boyut), "iter": int(iterasyon), "tekrar": int(n_calisma)},
            }
        ilerleme.empty()

    sonuc = st.session_state.get("sonuc")
    if not sonuc or not sonuc["algo"]:
        st.info("Başlamak için yukarıdan bir algoritma seç (ya da hepsini çalıştır).")
    else:
        if sonuc["model_imza"] != model_imza:
            st.warning(
                "Aşağıdaki sonuçlar önceki veri / maliyet / model ayarlarıyla hesaplandı. "
                "Güncel ayarlarla yeni bir algoritma çalıştırırsan eski sonuçlar silinir."
            )
        sm, sad, stalep = sonuc["model"], sonuc["adaylar"], sonuc["talep"]
        calisan = [ad for ad in ALGORITMALAR if ad in sonuc["algo"]]
        st.caption(f"Çalıştırılan algoritmalar: **{', '.join(kisa[a] for a in calisan)}** · "
                   f"Model: **{sonuc['tur']}**")

        ayarlar = {(v["ayar"]["pop"], v["ayar"]["iter"]) for v in sonuc["algo"].values()}
        if len(ayarlar) > 1:
            st.info("Algoritmalar farklı popülasyon / iterasyon ayarlarıyla çalıştırıldı. "
                    "Adil bir karşılaştırma için hepsini aynı ayarlarla çalıştır.")

        # ---------- Özet tablo ----------
        satirlar, en_iyi_detay = [], {}
        for ad in calisan:
            kosular = sonuc["algo"][ad]["kosular"]
            ayar = sonuc["algo"][ad]["ayar"]
            maliyetler = np.array([k["maliyet"] for k in kosular])
            b = int(maliyetler.argmin())
            d = sm.detay(kosular[b]["maske"])
            en_iyi_detay[ad] = d
            satirlar.append({
                "Yöntem": ad,
                "Toplam maliyet (milyon TL)": round(maliyetler.min(), 1),
                "Ort. toplam maliyet": f"{maliyetler.mean():,.1f}",
                "Std. sapma": f"{maliyetler.std():,.1f}",
                "Açılan hastane (en iyi)": int(kosular[b]["maske"].sum()),
                "Yapım (en iyi)": round(d["yapim"], 1),
                "Ulaşım (en iyi)": round(d["ulasim"], 1),
                "Ort. süre (sn)": f"{np.mean([k['sure'] for k in kosular]):.2f}",
                "Ayarlar": f"pop {ayar['pop']} × iter {ayar['iter']}, {ayar['tekrar']} tekrar",
            })
        for etiket, anahtar in [("Açgözlü ekleme (referans)", "acgozlu"), ("Hepsini aç (referans)", "hepsi")]:
            d = sm.detay(sonuc[anahtar])
            satirlar.append({
                "Yöntem": etiket,
                "Toplam maliyet (milyon TL)": round(d["toplam"], 1),
                "Ort. toplam maliyet": "-", "Std. sapma": "-",
                "Açılan hastane (en iyi)": int(len(d["idx"])),
                "Yapım (en iyi)": round(d["yapim"], 1),
                "Ulaşım (en iyi)": round(d["ulasim"], 1),
                "Ort. süre (sn)": "-", "Ayarlar": "-",
            })
        ozet = pd.DataFrame(satirlar)[[
            "Yöntem", "Açılan hastane (en iyi)", "Yapım (en iyi)", "Ulaşım (en iyi)",
            "Toplam maliyet (milyon TL)", "Ort. toplam maliyet", "Std. sapma",
            "Ort. süre (sn)", "Ayarlar",
        ]]

        st.subheader("Tablo 2: Sonuçlar ve karşılaştırma")
        st.caption("Toplam maliyet = Yapım + Ulaşım (milyon TL). Yapım, Ulaşım ve Toplam sütunları "
                   "her yöntemin en iyi çözümüne aittir; ortalama ve standart sapma tekrarların toplam maliyetidir.")
        st.dataframe(ozet, hide_index=True)
        st.download_button("Karşılaştırmayı CSV olarak indir", ozet.to_csv(index=False).encode("utf-8-sig"),
                           file_name="algoritma_karsilastirma.csv", mime="text/csv")

        hepsi_toplam = sm.detay(sonuc["hepsi"])["toplam"]
        en_iyi_ad = min(calisan, key=lambda a: min(k["maliyet"] for k in sonuc["algo"][a]["kosular"]))
        en_iyi_maliyet = min(k["maliyet"] for k in sonuc["algo"][en_iyi_ad]["kosular"])
        tasarruf = 100 * (hepsi_toplam - en_iyi_maliyet) / hepsi_toplam
        onek = "En iyi çözüm" if len(calisan) > 1 else "Sonuç"
        st.success(
            f"{onek}: **{en_iyi_ad}**. Toplam maliyet **{en_iyi_maliyet:,.0f} milyon TL**, "
            f"{len(en_iyi_detay[en_iyi_ad]['idx'])} hastane. Tüm adaylarda hastane açmaya göre "
            f"**%{tasarruf:.1f}** daha ucuz."
        )

        # ---------- Grafikler ----------
        st.subheader("Grafikler")
        g1, g2 = st.columns(2)

        with g1:
            st.markdown("**Yakınsama eğrisi** (tekrarların ortalaması)")
            log_eksen = st.checkbox("Y eksenini logaritmik yap", value=False)
            parcalar = []
            for ad in calisan:
                ort = np.mean([k["gecmis"] for k in sonuc["algo"][ad]["kosular"]], axis=0)
                parcalar.append(pd.DataFrame({"Iterasyon": np.arange(len(ort)), "Maliyet": ort, "Yöntem": ad}))
            ac_maliyet = sm.detay(sonuc["acgozlu"])["toplam"]
            n_nokta = max(len(p) for p in parcalar)
            parcalar.append(pd.DataFrame({"Iterasyon": np.arange(n_nokta), "Maliyet": ac_maliyet,
                                          "Yöntem": "Açgözlü (referans)"}))
            uzun = pd.concat(parcalar, ignore_index=True)
            grafik = alt.Chart(uzun).mark_line().encode(
                x=alt.X("Iterasyon:Q"),
                y=alt.Y("Maliyet:Q", title="Toplam maliyet (milyon TL)",
                        scale=alt.Scale(zero=False, type="log" if log_eksen else "linear")),
                color="Yöntem:N",
            ).properties(height=340)
            st.altair_chart(grafik)

        with g2:
            st.markdown("**Tekrarlar arası dağılım** (son maliyetler)")
            yeterli = [ad for ad in calisan if len(sonuc["algo"][ad]["kosular"]) >= 3]
            if yeterli:
                nokta = pd.DataFrame([{"Yöntem": ad, "Maliyet": k["maliyet"]}
                                      for ad in yeterli for k in sonuc["algo"][ad]["kosular"]])
                kutu = alt.Chart(nokta).mark_boxplot(extent="min-max").encode(
                    x=alt.X("Yöntem:N", title=None),
                    y=alt.Y("Maliyet:Q", title="Toplam maliyet (milyon TL)", scale=alt.Scale(zero=False)),
                    color="Yöntem:N",
                ).properties(height=340)
                st.altair_chart(kutu)
            else:
                st.info("Kutu grafiği için bir algoritmayı en az 3 tekrarla çalıştır.")

        # Yapım / ulaşım dağılımı
        dagilim = pd.DataFrame([
            {"Yöntem": r["Yöntem"], "Kalem": kalem, "Maliyet": r[f"{kalem} (en iyi)"]}
            for _, r in ozet.iterrows() for kalem in ["Yapım", "Ulaşım"]
        ])
        st.markdown("**Maliyetin yapım ve ulaşım olarak dağılımı** (referanslar dahil)")
        yigin = alt.Chart(dagilim).mark_bar().encode(
            x=alt.X("Yöntem:N", title=None, sort=None),
            y=alt.Y("Maliyet:Q", title="milyon TL"),
            color=alt.Color("Kalem:N"),
        ).properties(height=320)
        st.altair_chart(yigin)
        st.caption("'Hepsini aç' çubuğu yapım maliyetinin ne kadar şiştiğini gösterir; "
                   "algoritmalar yapım ve ulaşım kalemlerini dengeler.")

        # ---------- Haritalar ve ayrıntılı tablolar ----------
        st.subheader("Seçilen hastaneler")
        h1, h2 = st.columns(2)
        goster_talep2 = h1.checkbox("Talep noktalarını göster", value=False, key="gt2")
        goster_cizgi = h2.checkbox("Talep → hastane çizgilerini göster", value=False, key="gc2")
        st.caption("🟢 yeşil: açılan hastane  ·  ⚪ gri: açılmayan aday  ·  🟠 turuncu: talep noktası")

        sekmeler = st.tabs(list(en_iyi_detay.keys()))
        for sek, ad in zip(sekmeler, en_iyi_detay):
            with sek:
                d = en_iyi_detay[ad]
                a, b, c, e = st.columns(4)
                a.metric("Toplam maliyet", f"{d['toplam']:,.0f}")
                b.metric("Yapım", f"{d['yapim']:,.0f}")
                c.metric("Ulaşım", f"{d['ulasim']:,.0f}")
                e.metric("Açılan hastane", len(d["idx"]))
                st_folium(sonuc_haritasi(sad, stalep, d, goster_talep2, goster_cizgi),
                          use_container_width=True, height=560, returned_objects=[],
                          key=f"sonuc_harita_{ad}")
                st.markdown("**Tablo 3: Açılan hastaneler**")
                tablo = sonuc_tablosu(sad, d)
                st.dataframe(tablo, hide_index=True)
                st.download_button("CSV indir", tablo.to_csv(index=False).encode("utf-8-sig"),
                                   file_name=f"sonuc_{kisa[ad]}.csv", mime="text/csv", key=f"dl_{ad}")

# ===================== SEKME 3 =====================
def kaynak(nesne):
    """Fonksiyon / metodun kaynak kodunu girintisiz döndürür (koddan canlı okunur)."""
    return textwrap.dedent(inspect.getsource(nesne))


with sekme3:
    st.header("Yöntem")

    # ------------------------------------------------------------
    st.subheader("1) Problem")
    st.markdown(
        "Aday araziler arasından hangilerinde hastane açılacağına karar verilir. Amaç, **yapım maliyeti ile "
        "ulaşım maliyetinin toplamını** en aza indirmektir. Bu, **tesis yerleştirme problemidir** "
        "(Facility Location Problem): kapasite sınırı yoksa **kapasitesiz (UFLP)**, varsa **kapasiteli (CFLP)**."
    )

    # ------------------------------------------------------------
    st.subheader("2) Matematiksel model")
    st.markdown(
        "**Kümeler:** $I$ = aday araziler ($i = 1,\\dots,n$), $J$ = talep noktaları ($j = 1,\\dots,m$)  \n"
        "**Parametreler:** $f_i$ = $i$ adayında hastane yapım maliyeti, $d_j$ = $j$ noktasının nüfusu, "
        "$D_{ij}$ = $i$ ile $j$ arası uzaklık (km), $\\kappa$ = ulaşım maliyeti katsayısı (TL / kişi·km), "
        "$Q$ = hastane kapasitesi (kişi)  \n"
        "**Karar değişkenleri:** $y_i = 1$ ise $i$ adayında hastane açılır, değilse $0$. "
        "$x_{ij}$ = $j$ noktasının talebinin $i$ hastanesinden karşılanan oranı."
    )

    st.markdown("**Kapasitesiz model (UFLP)**")
    st.latex(r"\min \; Z = \sum_{i \in I} f_i \, y_i \;+\; \kappa \sum_{i \in I} \sum_{j \in J} d_j \, D_{ij} \, x_{ij}")
    st.latex(r"\text{kısıtlar:}\quad \sum_{i \in I} x_{ij} = 1 \;\; \forall j \in J, \qquad x_{ij} \le y_i \;\; \forall i, j")
    st.latex(r"y_i \in \{0,1\}, \qquad x_{ij} \ge 0")

    st.markdown("**Kapasiteli model (CFLP)** — aynı amaç fonksiyonu, ek olarak kapasite kısıtı:")
    st.latex(r"\sum_{j \in J} d_j \, x_{ij} \;\le\; Q \, y_i \quad \forall i \in I")
    st.latex(r"\sum_{i \in I} x_{ij} = 1 \;\; \forall j \in J, \qquad 0 \le x_{ij} \le 1, \qquad y_i \in \{0,1\}")
    st.caption("Kapasiteli modelde bir talep noktası birden fazla hastaneye bölünebilir. "
               "Kapasitenin yeterli olması için ayrıca $Q \\sum_i y_i \\ge \\sum_j d_j$ olmalıdır.")

    # ------------------------------------------------------------
    st.subheader("3) Maliyet bileşenleri")
    st.markdown("**Yapım maliyeti** (aday $i$ için, milyon TL): sabit inşaat bedeli $F$ ve gereken arazi $A$ (m²) "
                "için arsa bedeli. Arsa fiyatı merkeze yaklaştıkça artar.")
    st.latex(r"f_i = F + \frac{A \cdot p_i}{10^{6}}, \qquad p_i = p_0 \,\bigl(1 + k \, s_i\bigr), \qquad s_i = 1 - \frac{r_i}{r_{\max}}")
    st.markdown("$p_0$ = arsa taban fiyatı (TL/m²), $k$ = merkez etkisi, $r_i$ = adayın şehir merkezine (Kızılay) "
                "uzaklığı, $r_{\\max}$ = adaylar içindeki en büyük uzaklık. Merkezde $s_i = 1$, en uzak adayda $s_i = 0$.")

    st.markdown("**Ulaşım maliyeti** (milyon TL):")
    st.latex(r"T = \frac{\kappa}{10^{6}} \sum_{i \in I} \sum_{j \in J} d_j \, D_{ij} \, x_{ij}")

    st.markdown("**Mesafe** (Haversine, kuş uçuşu, $R = 6371$ km; $\\varphi$ enlem, $\\lambda$ boylam):")
    st.latex(r"D_{ij} = 2R \arcsin \sqrt{\sin^2\!\Bigl(\tfrac{\varphi_j-\varphi_i}{2}\Bigr) + \cos\varphi_i \cos\varphi_j \sin^2\!\Bigl(\tfrac{\lambda_j-\lambda_i}{2}\Bigr)}")

    # ------------------------------------------------------------
    st.subheader("4) Algoritmalarda çözüm gösterimi ve amaç fonksiyonu")
    st.markdown(
        "Algoritmalar yalnızca hangi adaylarda hastane açılacağını ($y$ vektörünü) arar. "
        "Verilen bir $y$ için atama ($x_{ij}$) ayrıca hesaplanır ve amaç fonksiyonu şudur:"
    )
    st.latex(r"Z(y) = \sum_{i \in I} f_i \, y_i + T(y)")
    st.markdown(
        "- **Kapasitesiz:** Her talep noktası en yakın **açık** hastaneye gider.  \n"
        "- **Kapasiteli:** Talep noktaları nüfusu büyükten küçüğe sıralanır; her biri en yakın, kapasitesi kalan "
        "hastaneden başlayarak karşılanır (bölünebilir). Bu açgözlü bir **sezgisel atamadır**, kesin optimum değildir.  \n"
        "- **Ceza:** Açık hastanelerin toplam kapasitesi nüfusa yetmiyorsa ($Q\\sum_i y_i < \\sum_j d_j$), "
        "çözüm cezalandırılır. Hiç hastane açılmazsa daha büyük ceza verilir:"
    )
    st.latex(r"Z(y) = P \Bigl(1 + \frac{\sum_j d_j - Q \sum_i y_i}{\sum_j d_j}\Bigr), \qquad P = 2\,\bigl(\textstyle\sum_i f_i + T(\text{hepsi açık}) + \overline{T}_{\text{tek}}\bigr)")
    st.caption("$\\overline{T}_{\\text{tek}}$: adaylardan yalnızca biri açık olsaydı ulaşım maliyetinin ortalaması.")

    # ------------------------------------------------------------
    st.subheader("5) Algoritmalar")
    st.markdown(
        "| Yöntem | Kodlama | Karar kuralı |\n|---|---|---|\n"
        "| GA | İkili vektör $y \\in \\{0,1\\}^n$ | Doğrudan $y$ |\n"
        "| DE | Reel vektör $x \\in [0,1]^n$ | $y_i = 1$ eğer $x_i > 0{,}5$ |\n"
        "| PSO | Reel vektör $x \\in [0,1]^n$ | $y_i = 1$ eğer $x_i > 0{,}5$ |\n\n"
        "Üç yöntem de aynı amaç fonksiyonunu kullanır; başlangıç popülasyonu seyrek çözümlerle "
        "(her bireyde adayların %2-30'u açık) üretilir. Kodlar `algoritmalar.py` dosyasından okunmaktadır."
    )

    st.markdown("#### 5.1) Genetik Algoritma (GA)")
    st.markdown(
        "1. Başlangıç popülasyonunu rastgele üret, her bireyin $Z$ değerini hesapla.  \n"
        "2. Her nesilde: en iyi `elit` bireyi aynen aktar; kalan bireyleri **turnuva seçimi** ile ebeveyn seçerek, "
        "olasılık $p_c$ ile **tekdüze çaprazlama** (her bit için yazı-tura), sonra her biti olasılık $p_m$ ile "
        "çevirerek (**mutasyon**) üret.  \n"
        "3. Yeni nesli değerlendir, en iyi çözümü güncelle; iterasyon bitene kadar tekrarla."
    )
    with st.expander("GA kodu (algoritmalar.py)"):
        st.code(kaynak(alg.genetik_algoritma), language="python")

    st.markdown("#### 5.2) Diferansiyel Evrim (DE/rand/1/bin)")
    st.markdown("Her birey $x_i$ için rastgele farklı üç birey $x_a, x_b, x_c$ seçilir:")
    st.latex(r"v = \operatorname{clip}\bigl(x_a + F\,(x_b - x_c),\, 0,\, 1\bigr)")
    st.latex(r"u_j = \begin{cases} v_j & \text{eğer } \text{rand}_j \le CR \;\text{ veya }\; j = j_{\text{rastgele}} \\ x_{i,j} & \text{aksi halde} \end{cases}")
    st.latex(r"x_i \leftarrow u \quad \text{eğer } Z(u > 0{,}5) \le Z(x_i > 0{,}5)")
    with st.expander("DE kodu (algoritmalar.py)"):
        st.code(kaynak(alg.diferansiyel_evrim), language="python")

    st.markdown("#### 5.3) Parçacık Sürü Optimizasyonu (PSO)")
    st.markdown("Her parçacığın konumu $x$, hızı $v$; kendi en iyisi $p$, sürünün en iyisi $g$:")
    st.latex(r"v \leftarrow w \, v + c_1 r_1 \,(p - x) + c_2 r_2 \,(g - x)")
    st.latex(r"x \leftarrow \operatorname{clip}(x + v,\, 0,\, 1), \qquad |v| \le v_{\max}")
    st.markdown("$r_1, r_2 \\sim U(0,1)$; atalet ağırlığı $w$ iterasyonla **0,9'dan 0,4'e doğrusal azalır**, "
                "$v_{\\max} = 0{,}4$.")
    with st.expander("PSO kodu (algoritmalar.py)"):
        st.code(kaynak(alg.parcacik_suru), language="python")

    st.markdown("#### 5.4) Referans: Açgözlü ekleme")
    st.markdown(
        "Hiç hastane açık değilken başlanır. Her adımda kapalı adaylardan **toplam maliyeti en çok azaltan** "
        "açılır; hiçbir aday maliyeti azaltmıyorsa durulur. GA/DE/PSO'nun kalitesini ölçmek için "
        "karşılaştırma referansı olarak kullanılır. Bir hastane açıldıktan sonra kapatılmaz; bu yüzden "
        "optimum garantisi yoktur."
    )
    with st.expander("Açgözlü ekleme kodu (algoritmalar.py)"):
        st.code(kaynak(alg.acgozlu_ekleme), language="python")

    st.markdown("#### 5.5) Maliyet fonksiyonu ve atama (amaç fonksiyonu)")
    with st.expander("Maliyet fonksiyonu: Z(y) (algoritmalar.py)"):
        st.code(kaynak(alg.TesisModeli.maliyet), language="python")
    with st.expander("Ulaşım maliyeti ve talep ataması (algoritmalar.py)"):
        st.code(kaynak(alg.TesisModeli._ulasim), language="python")
    with st.expander("Mesafe matrisi: Haversine (algoritmalar.py)"):
        st.code(kaynak(alg.haversine_matrisi), language="python")

    st.markdown("#### 5.6) Varsayılan algoritma parametreleri")
    st.markdown(
        "| Parametre | Varsayılan |\n|---|---|\n"
        "| Popülasyon / sürü büyüklüğü | 40 |\n"
        "| İterasyon sayısı | 150 |\n"
        "| GA: çaprazlama $p_c$, mutasyon $p_m$, turnuva, elit | 0,9 · 0,01 · 3 · 2 |\n"
        "| DE: $F$, $CR$ | 0,6 · 0,9 |\n"
        "| PSO: $c_1$, $c_2$, $w$, $v_{\\max}$ | 1,5 · 1,5 · 0,9→0,4 · 0,4 |\n\n"
        "Değerler kenar çubuğundan değiştirilebilir. Algoritmalar rastgele olduğu için birkaç tekrar yapılır; "
        "en iyi, ortalama ve standart sapma raporlanır."
    )

    # ------------------------------------------------------------
    st.subheader("6) Veri ve varsayımlar")
    st.markdown(
        """
- **Adaylar:** OpenStreetMap'te boş arazi olarak etiketli alanlar (brownfield, greenfield, meadow,
  grassland, scrub, heath). Alanı eşikten küçük olanlar elenir, çok ise KMeans ile şehre dengeli
  yayılan seçilen sayıda aday alınır. Arazilerin gerçekte boş olduğu varsayılmıştır.
- **Talep noktaları:** OSM yerleşim alanları (landuse=residential). Nüfus, yerleşim alanının
  büyüklüğüyle orantılı varsayılır ve toplam nüfusa ölçeklenir. Alan, sınır kutusu alanının 0,6 katı
  olarak yaklaşık hesaplanır. Alan ağırlıklı KMeans ile talep noktası sayısına indirilir.
- **Mesafe:** Kuş uçuşu (Haversine), yol ağı mesafesi değil.
- **Maliyet parametreleri** ($F$, $p_0$, $k$, $\\kappa$) varsayımdır; kenar çubuğundan değiştirilerek
  duyarlılık analizi yapılabilir.
- **Kapasiteli atama** açgözlü sezgiseldir; bulunan maliyetler gerçek kapasiteli optimumun üst sınırıdır.
- Algoritmalar **optimum garantisi vermez**; rastgele çalıştıkları için sonuçlar tekrarlar arasında değişir.
        """
    )
    