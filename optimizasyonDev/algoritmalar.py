"""
Tesis yerleştirme (Facility Location) modeli ve optimizasyon algoritmaları.

- Kapasitesiz (UFLP): her talep noktası en yakın AÇIK hastaneye gider.
- Kapasiteli (CFLP): her hastanenin kapasitesi var; talep açgözlü yöntemle
  (büyük talepten küçüğe, en yakın boş kapasiteli hastane) paylaştırılır.
  Bir talep noktası birden fazla hastaneye bölünebilir.

Toplam maliyet = Σ yapım maliyeti (açık hastaneler) + Σ ulaşım maliyeti
Birim: milyon TL
"""
import time
import numpy as np

DUNYA_YARICAPI_KM = 6371.0


def haversine_matrisi(lat1, lon1, lat2, lon2):
    """(n1 x n2) kuş uçuşu mesafe matrisi, km."""
    la1 = np.radians(np.asarray(lat1, float))[:, None]
    lo1 = np.radians(np.asarray(lon1, float))[:, None]
    la2 = np.radians(np.asarray(lat2, float))[None, :]
    lo2 = np.radians(np.asarray(lon2, float))[None, :]
    a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * DUNYA_YARICAPI_KM * np.arcsin(np.sqrt(a))


class TesisModeli:
    """
    D        : (n_aday x n_talep) mesafe matrisi (km)
    nufus    : (n_talep,) her talep noktasının nüfusu
    yapim    : (n_aday,) her adayın yapım maliyeti (milyon TL)
    ulasim_katsayisi : TL / (kişi·km)  (planlama süresi boyunca toplam)
    kapasite : None -> kapasitesiz, sayı -> her hastanenin kişi kapasitesi
    """

    def __init__(self, D, nufus, yapim, ulasim_katsayisi, kapasite=None):
        self.D = np.asarray(D, float)
        self.nufus = np.asarray(nufus, float)
        self.yapim = np.asarray(yapim, float)
        self.c = float(ulasim_katsayisi) / 1e6          # milyon TL
        self.kapasite = None if kapasite is None else float(kapasite)
        self.n, self.m = self.D.shape
        self.toplam_nufus = float(self.nufus.sum())
        self.talep_sirasi = np.argsort(-self.nufus).tolist()
        self.nufus_liste = self.nufus.tolist()
        self.n_degerlendirme = 0

        # ceza değeri (kapasite yetersizse): "hepsini aç" maliyetinin katı
        self.ceza = 0.0
        hepsi = np.arange(self.n)
        yap = self.yapim.sum()
        ulas, _ = self._ulasim(hepsi)
        ort_tek = float((self.D @ self.nufus).mean()) * self.c
        self.ceza = 2.0 * (yap + ulas + ort_tek)

    # ------------------------------------------------------------------
    def kapasite_yeterli_mi(self):
        return self.kapasite is None or self.n * self.kapasite >= self.toplam_nufus

    def _ulasim(self, idx, flow_istenir=False):
        """Açık hastane indeksleri için ulaşım maliyeti (milyon TL) ve (isteğe bağlı) akış matrisi."""
        Do = self.D[idx]                                  # k x m
        k = len(idx)
        sutun = np.arange(self.m)
        en_yakin = Do.argmin(axis=0)

        # Hızlı yol: kapasitesiz ya da kapasite zaten aşılmıyor
        hizli = True
        if self.kapasite is not None:
            yuk = np.bincount(en_yakin, weights=self.nufus, minlength=k)
            hizli = yuk.max() <= self.kapasite + 1e-9
        if hizli:
            ulasim = float((self.nufus * Do[en_yakin, sutun]).sum()) * self.c
            flow = None
            if flow_istenir:
                flow = np.zeros((k, self.m))
                flow[en_yakin, sutun] = self.nufus
            return ulasim, flow

        # Kapasiteli açgözlü atama
        kalan = [self.kapasite] * k
        sira = np.argsort(Do, axis=0).T.tolist()          # her talep için hastane sırası
        flow = np.zeros((k, self.m))
        for i in self.talep_sirasi:
            r = self.nufus_liste[i]
            for f in sira[i]:
                kf = kalan[f]
                if kf <= 1e-9:
                    continue
                t = r if r < kf else kf
                kalan[f] = kf - t
                flow[f, i] += t
                r -= t
                if r <= 1e-9:
                    break
        ulasim = float((flow * Do).sum()) * self.c
        return ulasim, flow

    # ------------------------------------------------------------------
    def maliyet(self, maske):
        """Amaç fonksiyonu: toplam maliyet (düşük = iyi)."""
        self.n_degerlendirme += 1
        idx = np.flatnonzero(maske)
        if idx.size == 0:
            return self.ceza * 3
        if self.kapasite is not None:
            eksik = self.toplam_nufus - idx.size * self.kapasite
            if eksik > 1e-9:
                return self.ceza * (1.0 + eksik / self.toplam_nufus)
        ulasim, _ = self._ulasim(idx)
        return float(self.yapim[idx].sum()) + ulasim

    def detay(self, maske):
        """Bir çözümün ayrıntılı dökümü (tablo ve harita için)."""
        idx = np.flatnonzero(maske)
        ulasim, flow = self._ulasim(idx, flow_istenir=True)
        Do = self.D[idx]
        hizmet = flow.sum(axis=1)
        ulasim_kalem = (flow * Do).sum(axis=1) * self.c
        with np.errstate(invalid="ignore", divide="ignore"):
            ort_uzaklik = np.where(hizmet > 0, (flow * Do).sum(axis=1) / hizmet, 0.0)
        yapim = float(self.yapim[idx].sum())
        return {
            "idx": idx,
            "flow": flow,
            "hizmet_nufus": hizmet,
            "ort_uzaklik_km": ort_uzaklik,
            "ulasim_kalem": ulasim_kalem,
            "yapim_kalem": self.yapim[idx],
            "doluluk": (hizmet / self.kapasite) if self.kapasite else None,
            "yapim": yapim,
            "ulasim": ulasim,
            "toplam": yapim + ulasim,
        }


# ======================================================================
# Başlangıç popülasyonları (seyrek çözümlerle başlamak yakınsamayı hızlandırır)
# ======================================================================
def _ikili_pop(rng, pop, n):
    q = rng.uniform(0.02, 0.30, size=(pop, 1))
    return rng.random((pop, n)) < q


def _surekli_pop(rng, pop, n):
    q = rng.uniform(0.02, 0.30, size=(pop, 1))
    acik = rng.random((pop, n)) < q
    return np.where(acik, 0.5 + 0.5 * rng.random((pop, n)), 0.5 * rng.random((pop, n)))


def _degerlendir(model, P, esik=None):
    if esik is None:
        return np.array([model.maliyet(x) for x in P])
    return np.array([model.maliyet(x > esik) for x in P])


# ======================================================================
# 1) Genetik Algoritma (ikili kodlama)
# ======================================================================
def genetik_algoritma(model, pop=40, iters=100, rng=None, caprazlama=0.9,
                      mutasyon=0.01, turnuva=3, elit=2):
    n = model.n
    P = _ikili_pop(rng, pop, n)
    f = _degerlendir(model, P)
    i0 = int(f.argmin())
    en_iyi, en_iyi_f = P[i0].copy(), float(f[i0])
    gecmis = [en_iyi_f]

    def sec():
        aday = rng.integers(0, pop, size=turnuva)
        return P[aday[np.argmin(f[aday])]]

    for _ in range(iters):
        sira = np.argsort(f)
        yeni = [P[i].copy() for i in sira[:elit]]
        while len(yeni) < pop:
            a, b = sec(), sec()
            if rng.random() < caprazlama:
                m = rng.random(n) < 0.5
                c1, c2 = np.where(m, a, b), np.where(m, b, a)
            else:
                c1, c2 = a.copy(), b.copy()
            for c in (c1, c2):
                cevir = rng.random(n) < mutasyon
                c[cevir] = ~c[cevir]
                yeni.append(c)
        P = np.array(yeni[:pop])
        f = _degerlendir(model, P)
        i0 = int(f.argmin())
        if f[i0] < en_iyi_f:
            en_iyi, en_iyi_f = P[i0].copy(), float(f[i0])
        gecmis.append(en_iyi_f)
    return en_iyi, en_iyi_f, gecmis


# ======================================================================
# 2) Diferansiyel Evrim (DE/rand/1/bin, sürekli kodlama + eşikleme)
# ======================================================================
def diferansiyel_evrim(model, pop=40, iters=100, rng=None, F=0.6, CR=0.9):
    n = model.n
    X = _surekli_pop(rng, pop, n)
    f = _degerlendir(model, X, 0.5)
    i0 = int(f.argmin())
    en_iyi, en_iyi_f = X[i0].copy(), float(f[i0])
    gecmis = [en_iyi_f]

    for _ in range(iters):
        for i in range(pop):
            diger = [j for j in range(pop) if j != i]
            a, b, c = rng.choice(diger, 3, replace=False)
            v = np.clip(X[a] + F * (X[b] - X[c]), 0.0, 1.0)
            cr = rng.random(n) < CR
            cr[rng.integers(n)] = True
            u = np.where(cr, v, X[i])
            fu = model.maliyet(u > 0.5)
            if fu <= f[i]:
                X[i], f[i] = u, fu
                if fu < en_iyi_f:
                    en_iyi, en_iyi_f = u.copy(), float(fu)
        gecmis.append(en_iyi_f)
    return en_iyi > 0.5, en_iyi_f, gecmis


# ======================================================================
# 3) Parçacık Sürü Optimizasyonu (sürekli kodlama + eşikleme)
# ======================================================================
def parcacik_suru(model, pop=40, iters=100, rng=None, w_basla=0.9, w_bitis=0.4,
                  c1=1.5, c2=1.5, vmax=0.4):
    n = model.n
    X = _surekli_pop(rng, pop, n)
    V = rng.uniform(-vmax / 4, vmax / 4, size=(pop, n))
    f = _degerlendir(model, X, 0.5)
    pbest, pbest_f = X.copy(), f.copy()
    g = int(f.argmin())
    gbest, gbest_f = X[g].copy(), float(f[g])
    gecmis = [gbest_f]

    for t in range(iters):
        w = w_basla - (w_basla - w_bitis) * t / max(1, iters - 1)
        r1, r2 = rng.random((pop, n)), rng.random((pop, n))
        V = w * V + c1 * r1 * (pbest - X) + c2 * r2 * (gbest - X)
        V = np.clip(V, -vmax, vmax)
        X = np.clip(X + V, 0.0, 1.0)
        f = _degerlendir(model, X, 0.5)
        iyi = f < pbest_f
        pbest[iyi], pbest_f[iyi] = X[iyi], f[iyi]
        g = int(pbest_f.argmin())
        if pbest_f[g] < gbest_f:
            gbest, gbest_f = pbest[g].copy(), float(pbest_f[g])
        gecmis.append(gbest_f)
    return gbest > 0.5, gbest_f, gecmis


# ======================================================================
# Referans: açgözlü ekleme (iyileşme olduğu sürece en iyi adayı aç)
# ======================================================================
def acgozlu_ekleme(model):
    n = model.n
    maske = np.zeros(n, bool)
    en_iyi = float("inf")
    while True:
        aday_f, aday_j = en_iyi, None
        for j in np.flatnonzero(~maske):
            maske[j] = True
            c = model.maliyet(maske)
            maske[j] = False
            if c < aday_f - 1e-9:
                aday_f, aday_j = c, j
        if aday_j is None:
            break
        maske[aday_j] = True
        en_iyi = aday_f
    return maske, en_iyi


# ======================================================================
ALGORITMALAR = {
    "Genetik Algoritma (GA)": genetik_algoritma,
    "Diferansiyel Evrim (DE)": diferansiyel_evrim,
    "Parçacık Sürü (PSO)": parcacik_suru,
}


def calistir(fonksiyon, model, seed, **parametreler):
    """Bir algoritmayı tek sefer çalıştırır, sonuç sözlüğü döndürür."""
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    maske, maliyet, gecmis = fonksiyon(model, rng=rng, **parametreler)
    sure = time.perf_counter() - t0
    return {"maske": np.asarray(maske, bool), "maliyet": float(maliyet),
            "gecmis": gecmis, "sure": sure}
