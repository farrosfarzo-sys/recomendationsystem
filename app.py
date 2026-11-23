"""
Feature Selection & Feature Extraction pada Dataset Book-Crossing
Aplikasi Streamlit - dikonversi dari notebook
"Farros_Raditya_Ahmad_Sifana_5003241017_Tugas_DMV (K) Dataset BookCrossing.ipynb"

Jalankan:  streamlit run app.py
Data     :  BX-Books.csv, BX-Users.csv, BX-Book-Ratings.csv
            (letakkan di folder yang sama dengan app.py / folder ./data, atau upload lewat sidebar)
"""
import itertools
import re
import tempfile
import time
import warnings
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import streamlit as st
from scipy.optimize import minimize
from sklearn.datasets import load_digits, load_wine, make_circles
from sklearn.decomposition import PCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import (RFE, RFECV, SelectFromModel, SelectKBest,
                                       SequentialFeatureSelector, VarianceThreshold,
                                       chi2, f_classif, f_regression,
                                       mutual_info_classif, mutual_info_regression)
from sklearn.linear_model import (ElasticNet, ElasticNetCV, LassoCV, LogisticRegression,
                                  RidgeCV, lasso_path)
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, roc_auc_score, silhouette_score
from sklearn.model_selection import (StratifiedKFold, cross_val_score, cross_validate,
                                     train_test_split)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Feature Selection - Book-Crossing", page_icon="📚", layout="wide")

plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": .25,
                     "axes.spines.top": False, "axes.spines.right": False})

BLUE, ORANGE, CYAN, RED = "#0a2a66", "#f5a623", "#1aa5c8", "#d62728"
CMAP3 = {0: BLUE, 1: ORANGE, 2: CYAN}
REQUIRED = ["BX-Books.csv", "BX-Users.csv", "BX-Book-Ratings.csv"]


# =====================================================================================
# Utilitas umum
# =====================================================================================
def show(fig):
    st.pyplot(fig)
    plt.close(fig)


def top_names(s, k):
    return list(s.sort_values(ascending=False).head(k).index)


def plot_top(s, title, noise, k=15, color=BLUE):
    s = s.sort_values(ascending=False).head(k)[::-1]
    cols = [RED if i in noise else color for i in s.index]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.barh(s.index, s.values, color=cols)
    ax.set_title(title + "   (merah = fitur noise/konstan buatan)", fontsize=10)
    fig.tight_layout()
    return fig


def gate(label, flag, help_text=None):
    """Tombol 'jalankan' yang statusnya diingat (untuk komputasi berat)."""
    if st.button(label, key="btn_" + flag, help=help_text):
        st.session_state[flag] = True
    return st.session_state.get(flag, False)


def make_logit():
    return Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(max_iter=2000))])


def l1_logreg(C):
    try:
        return LogisticRegression(penalty="l1", solver="liblinear", C=C, max_iter=500)
    except TypeError:
        return LogisticRegression(l1_ratio=1.0, solver="saga", C=C, max_iter=500)


def clean_name(s):
    return re.sub(r"[^0-9a-zA-Z]+", "_", str(s)).strip("_")[:18]


# =====================================================================================
# Sumber data
# =====================================================================================
def find_default_dir():
    for p in [Path("."), Path("data"), Path("/mnt/user-data/uploads")]:
        if all((p / f).exists() for f in REQUIRED):
            return str(p.resolve())
    return ""


def dir_signature(d):
    p = Path(d)
    return tuple((f, (p / f).stat().st_size, (p / f).stat().st_mtime_ns) for f in REQUIRED)


def get_data_dir():
    mode = st.sidebar.radio("Sumber data", ["Folder lokal", "Upload CSV"])
    if mode == "Folder lokal":
        d = st.sidebar.text_input("Path folder data", value=find_default_dir())
        if d and all((Path(d) / f).exists() for f in REQUIRED):
            return d
        st.sidebar.warning("Folder harus berisi: " + ", ".join(REQUIRED))
        return None
    ups = {f: st.sidebar.file_uploader(f, type="csv", key="up_" + f) for f in REQUIRED}
    if all(ups.values()):
        sig = tuple((f, u.size) for f, u in ups.items())
        if st.session_state.get("_upl_sig") != sig:
            tmp = tempfile.mkdtemp(prefix="bx_")
            for f, u in ups.items():
                (Path(tmp) / f).write_bytes(u.getvalue())
            st.session_state["_upl_sig"] = sig
            st.session_state["_upl_dir"] = tmp
        return st.session_state["_upl_dir"]
    return None


# =====================================================================================
# Memuat, membersihkan data, rekayasa fitur, pembagian data  (satu bundle, di-cache)
# =====================================================================================
@st.cache_resource(show_spinner="Memuat dan memproses data (hanya sekali per pengaturan) ...")
def build_bundle(data_dir, sig, seed, min_expl):
    p = Path(data_dir)
    rng = np.random.default_rng(seed)

    books = pd.read_csv(p / "BX-Books.csv", sep=";", quotechar='"', escapechar="\\",
                        encoding="latin-1", low_memory=False)
    users = pd.read_csv(p / "BX-Users.csv", sep=";", encoding="latin-1", na_values=["NULL"])
    ratings = pd.read_csv(p / "BX-Book-Ratings.csv", sep=";", encoding="latin-1")

    books.columns = ["isbn", "title", "author", "year", "publisher", "img_s", "img_m", "img_l"]
    users.columns = ["user_id", "location", "age"]
    ratings.columns = ["user_id", "isbn", "rating"]
    raw_shapes = {"Books": books.shape, "Users": users.shape, "Ratings": ratings.shape}

    # --- pembersihan ---
    users.loc[~users["age"].between(5, 100), "age"] = np.nan
    users["country"] = (users["location"].fillna("").str.split(",").str[-1]
                        .str.strip().replace("", "unknown"))
    books["year"] = pd.to_numeric(books["year"], errors="coerce")
    books.loc[~books["year"].between(1900, 2005), "year"] = np.nan
    for col in ["author", "publisher", "title"]:
        books[col] = books[col].fillna("unknown").astype(str).str.lower().str.strip()
    books["isbn"] = books["isbn"].astype(str).str.upper().str.strip()
    ratings["isbn"] = ratings["isbn"].astype(str).str.upper().str.strip()
    books = books.drop(columns=["img_s", "img_m", "img_l"])

    # --- rekayasa fitur: satu baris = satu pengguna ---
    inter = ratings.merge(books, on="isbn", how="inner")
    expl = inter[inter.rating > 0]
    stats = expl.groupby("user_id").rating.agg(["mean", "count"])
    target_users = stats.index[stats["count"] >= min_expl]
    d = inter[inter.user_id.isin(target_users)].copy()

    d["log_book_pop"] = np.log1p(d.isbn.map(inter.groupby("isbn").size()))
    d["log_author_pop"] = np.log1p(d.author.map(inter.groupby("author").size()))
    d["title_len"] = d.title.str.len()

    g = d.groupby("user_id")
    F = pd.DataFrame(index=sorted(target_users))
    F["n_interactions"] = g.size()
    F["log_n_interactions"] = np.log1p(F.n_interactions)
    F["pct_implicit"] = (d.rating == 0).groupby(d.user_id).mean()
    F["year_mean"] = g.year.mean()
    F["year_std"] = g.year.std()
    F["log_book_pop_mean"] = g.log_book_pop.mean()
    F["log_author_pop_mean"] = g.log_author_pop.mean()
    F["title_len_mean"] = g.title_len.mean()
    F["author_diversity"] = g.author.nunique() / F.n_interactions
    F["publisher_diversity"] = g.publisher.nunique() / F.n_interactions

    def share_features(col, top_n, prefix):
        top = d[col].value_counts().head(top_n).index
        ct = (d[d[col].isin(top)].groupby(["user_id", col]).size().unstack(fill_value=0)
              .reindex(index=F.index, columns=top, fill_value=0))
        ct = ct.div(F.n_interactions, axis=0)
        ct.columns = [f"{prefix}_{i:02d}_{clean_name(c)}" for i, c in enumerate(top)]
        return ct

    dec = pd.cut(d.year, [1900, 1960, 1980, 1990, 1995, 2000, 2006], right=False,
                 labels=["1900_59", "1960_79", "1980_89", "1990_94", "1995_99", "2000_05"])
    dec_ct = (d.assign(dec=dec).groupby(["user_id", "dec"]).size()
              .unstack(fill_value=0).reindex(F.index, fill_value=0))
    dec_ct = dec_ct.div(F.n_interactions, axis=0)
    dec_ct.columns = [f"yr_share_{c}" for c in dec_ct.columns]

    u = users.drop_duplicates("user_id").set_index("user_id").loc[F.index]
    age_missing = u.age.isna().astype(float).rename("age_missing")
    age = u.age.fillna(users.age.median()).rename("age")
    top_c = u.country.value_counts().head(15).index
    ctry = pd.get_dummies(u.country.where(u.country.isin(top_c))).astype(float).add_prefix("ctry_")
    ctry.columns = [clean_name(c) for c in ctry.columns]

    X = pd.concat([F, dec_ct, share_features("publisher", 60, "pub"),
                   share_features("author", 60, "auth"), age, age_missing, ctry], axis=1)
    X = X.fillna(X.median())

    # fitur buatan untuk pembelajaran
    X["const_feature"] = 1.0
    X["near_const_flag"] = (rng.random(len(X)) < 0.0005).astype(float)
    X["age_months"] = X.age * 12 + rng.normal(0, 2, len(X))
    for i in range(1, 21):
        X[f"noise_{i:02d}"] = rng.normal(size=len(X))
    injected = ["const_feature", "near_const_flag", "age_months"] + [f"noise_{i:02d}" for i in range(1, 21)]
    noise = [c for c in injected if c.startswith("noise") or c in ("const_feature", "near_const_flag")]

    y_reg = stats.loc[X.index, "mean"]
    y_bin = (y_reg > y_reg.median()).astype(int)
    y_3 = pd.Series(pd.qcut(y_reg, 3, labels=False), index=X.index)

    # --- pembagian data & standardisasi ---
    tr_idx, te_idx = train_test_split(X.index, test_size=0.2, stratify=y_bin, random_state=seed)
    X_train, X_test = X.loc[tr_idx], X.loc[te_idx]
    sc = StandardScaler().fit(X_train)
    mm = MinMaxScaler().fit(X_train)
    Xtr_s = pd.DataFrame(sc.transform(X_train), columns=X.columns, index=X_train.index)
    Xtr_mm = pd.DataFrame(mm.transform(X_train), columns=X.columns, index=X_train.index)

    sub_idx = X_train.sample(min(3000, len(X_train)), random_state=seed).index
    cmp_idx = X_train.sample(min(5000, len(X_train)), random_state=seed).index

    return dict(
        key=f"{data_dir}|{sig}|{seed}|{min_expl}", seed=seed,
        raw_shapes=raw_shapes, books=books, users=users, ratings=ratings,
        n_inter=len(inter), n_target=len(target_users), n_used=len(d),
        X=X, y_reg=y_reg, y_bin=y_bin, y_3=y_3, injected=injected, noise=noise,
        X_train=X_train, X_test=X_test,
        yb_tr=y_bin.loc[tr_idx], yb_te=y_bin.loc[te_idx],
        yr_tr=y_reg.loc[tr_idx], yr_te=y_reg.loc[te_idx],
        y3_tr=y_3.loc[tr_idx], y3_te=y_3.loc[te_idx],
        sc=sc, mm=mm, Xtr_s=Xtr_s, Xtr_mm=Xtr_mm,
        sub_idx=sub_idx, X_sub=X_train.loc[sub_idx], yb_sub=y_bin.loc[tr_idx].loc[sub_idx],
        cmp_idx=cmp_idx, X_cmp=X_train.loc[cmp_idx], yb_cmp=y_bin.loc[tr_idx].loc[cmp_idx],
        cv5=StratifiedKFold(5, shuffle=True, random_state=seed),
    )


# =====================================================================================
# Fungsi komputasi (di-cache)
# =====================================================================================
@st.cache_data(show_spinner="Menghitung skor filter (korelasi, ANOVA, χ², Mutual Information) ...")
def f_filters(_D, key):
    D = _D
    seed = D["seed"]
    X, Xtr_s, Xtr_mm = D["X"], D["Xtr_s"], D["Xtr_mm"]
    corr_y = D["X_train"].corrwith(D["yr_tr"]).abs().fillna(0)
    Xf = Xtr_s.fillna(0)
    F_clf, p_clf = f_classif(Xf, D["yb_tr"])
    F_reg, p_reg = f_regression(Xf, D["yr_tr"])
    anova = pd.DataFrame({
        "F_classif": np.nan_to_num(F_clf), "p_classif": np.nan_to_num(p_clf, nan=1),
        "F_regression": np.nan_to_num(F_reg), "p_regression": np.nan_to_num(p_reg, nan=1)},
        index=X.columns)
    chi_s, _ = chi2(Xtr_mm, D["yb_tr"])
    chi_sc = pd.Series(np.nan_to_num(chi_s), index=X.columns)
    t0 = time.time()
    mi_clf = pd.Series(mutual_info_classif(Xtr_s, D["yb_tr"], random_state=seed), index=X.columns)
    mi_reg = pd.Series(mutual_info_regression(Xtr_s, D["yr_tr"], random_state=seed), index=X.columns)
    return dict(corr_y=corr_y, anova=anova, chi=chi_sc, mi_clf=mi_clf, mi_reg=mi_reg,
                mi_time=time.time() - t0)


@st.cache_data(show_spinner="Menghitung relative contrast ...")
def f_contrast(seed):
    rng = np.random.default_rng(seed)

    def rel_contrast(p, n=200, reps=40):
        vals = []
        for _ in range(reps):
            P, q = rng.random((n, p)), rng.random(p)
            dist = np.linalg.norm(P - q, axis=1)
            vals.append((dist.max() - dist.min()) / dist.min())
        return np.median(vals)

    dims = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    return dims, [rel_contrast(p) for p in dims]


@st.cache_data(show_spinner="Menjalankan eksperimen k-NN dengan fitur noise ...")
def f_knn_noise(_D, key):
    D = _D
    rng = np.random.default_rng(D["seed"])
    Fs = f_filters(D, key)["anova"].F_classif
    real30 = [c for c in Fs.sort_values(ascending=False).index if c not in D["noise"]][:30]
    Xr, yr_ = D["X_train"].loc[D["sub_idx"], real30].values, D["yb_sub"].values
    knn = Pipeline([("sc", StandardScaler()), ("knn", KNeighborsClassifier(5))])
    counts = [0, 10, 30, 100, 300, 1000]
    accs = []
    for m in counts:
        Xn = np.hstack([Xr, rng.normal(size=(len(Xr), m))]) if m else Xr
        accs.append(cross_val_score(knn, Xn, yr_, cv=D["cv5"]).mean())
    return counts, accs


@st.cache_data(show_spinner="Menjalankan Exhaustive Feature Selection ...")
def f_efs(_D, key):
    D = _D
    pool8 = top_names(f_filters(D, key)["anova"].F_classif, 8)
    logit = make_logit()
    res, t0 = [], time.time()
    for r in range(1, len(pool8) + 1):
        for comb in itertools.combinations(pool8, r):
            res.append((comb, r, cross_val_score(logit, D["X_sub"][list(comb)], D["yb_sub"], cv=D["cv5"]).mean()))
    efs = pd.DataFrame(res, columns=["subset", "size", "cv_acc"]).sort_values("cv_acc", ascending=False)
    return efs, time.time() - t0


@st.cache_data(show_spinner="Menjalankan SFS dan SBS (bisa beberapa menit) ...")
def f_seq(_D, key):
    D = _D
    pool30 = top_names(f_filters(D, key)["anova"].F_classif, 30)
    out = {}
    for name, direction in [("SFS (forward)", "forward"), ("SBS (backward)", "backward")]:
        t0 = time.time()
        sfs = SequentialFeatureSelector(make_logit(), n_features_to_select=8, direction=direction,
                                        scoring="accuracy", cv=D["cv5"], n_jobs=-1)
        sfs.fit(D["X_sub"][pool30], D["yb_sub"])
        out[name] = {"features": [str(f) for f in np.array(pool30)[sfs.get_support()]],
                     "time_s": time.time() - t0}
    return out


@st.cache_data(show_spinner="Menjalankan stepwise (p-value) ...")
def f_step(_D, key, p_enter=0.05, p_remove=0.10, max_iter=100):
    D = _D
    pool30 = top_names(f_filters(D, key)["anova"].F_classif, 30)
    Xd, y = D["Xtr_s"][pool30], D["yr_tr"]
    inc = []
    for _ in range(max_iter):
        changed = False
        exc = [c for c in Xd.columns if c not in inc]
        if not exc:
            break
        newp = pd.Series({c: sm.OLS(y, sm.add_constant(Xd[inc + [c]])).fit().pvalues[c] for c in exc})
        if len(newp) and newp.min() < p_enter:
            inc.append(newp.idxmin())
            changed = True
        if inc:
            pv = sm.OLS(y, sm.add_constant(Xd[inc])).fit().pvalues.drop("const")
            if pv.max() > p_remove:
                inc.remove(pv.idxmax())
                changed = True
        if not changed:
            break
    r2 = sm.OLS(y, sm.add_constant(Xd[inc])).fit().rsquared if inc else 0.0
    return inc, r2


@st.cache_data(show_spinner="Menjalankan RFE dan RFECV ...")
def f_rfe(_D, key, topk):
    D = _D
    X = D["X"]
    t0 = time.time()
    rfe = RFE(LogisticRegression(max_iter=2000), n_features_to_select=topk, step=10).fit(
        D["Xtr_s"].fillna(0), D["yb_tr"])
    feats = list(X.columns[rfe.support_])
    t_rfe = time.time() - t0
    rfecv = RFECV(LogisticRegression(max_iter=2000), step=5, cv=3, scoring="accuracy",
                  min_features_to_select=1, n_jobs=-1)
    rfecv.fit(StandardScaler().fit_transform(D["X_sub"]), D["yb_sub"])
    scores = rfecv.cv_results_["mean_test_score"]
    nf = rfecv.cv_results_.get("n_features", np.arange(1, len(scores) + 1))
    return feats, t_rfe, np.asarray(nf), np.asarray(scores), int(rfecv.n_features_)


@st.cache_data(show_spinner="Menghitung Lasso / Ridge / Elastic Net ...")
def f_lasso(_D, key):
    D = _D
    seed = D["seed"]
    Xtr_s, yr_tr, X = D["Xtr_s"], D["yr_tr"], D["X"]
    alphas, coefs, _ = lasso_path(Xtr_s.values, yr_tr.values, alphas=np.logspace(-3.5, 0, 60))
    lcv = LassoCV(cv=5, random_state=seed).fit(Xtr_s, yr_tr)
    rcv = RidgeCV(alphas=np.logspace(-2, 4, 40)).fit(Xtr_s, yr_tr)
    ecv = ElasticNetCV(l1_ratio=[.2, .5, .8, .95], cv=5, random_state=seed).fit(Xtr_s, yr_tr)
    coef = pd.DataFrame({"Lasso": lcv.coef_, "Ridge": rcv.coef_, "ElasticNet": ecv.coef_}, index=X.columns)
    Xte_s = D["sc"].transform(D["X_test"])
    summ = pd.DataFrame({
        "alpha": [lcv.alpha_, rcv.alpha_, ecv.alpha_],
        "koef. tepat nol": (coef == 0).sum().values,
        "koef. non-nol": (coef != 0).sum().values,
        "R² uji": [m.score(Xte_s, D["yr_te"]) for m in (lcv, rcv, ecv)]},
        index=["Lasso", "Ridge", "ElasticNet"])
    en2 = ElasticNet(alpha=lcv.alpha_ / 0.2, l1_ratio=0.2, random_state=seed).fit(Xtr_s, yr_tr)
    en2_coef = pd.Series(en2.coef_, index=X.columns)
    return dict(alphas=alphas, coefs=coefs, lcv_alpha=lcv.alpha_, coef=coef, summ=summ,
                l1_ratio=ecv.l1_ratio_, en2_coef=en2_coef, en2_nz=int((en2.coef_ != 0).sum()))


@st.cache_data(show_spinner="Menjalankan L1-logistic dan L1-SVM ...")
def f_l1(_D, key):
    D = _D
    X = D["X"]
    rows = []
    for Cval in [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1]:
        pipe = Pipeline([("sc", StandardScaler()), ("clf", l1_logreg(Cval))])
        acc = cross_val_score(pipe, D["X_train"], D["yb_tr"], cv=D["cv5"], n_jobs=-1).mean()
        nz = int((pipe.fit(D["X_train"], D["yb_tr"]).named_steps["clf"].coef_ != 0).sum())
        rows.append((Cval, nz, acc))
    tab = pd.DataFrame(rows, columns=["C", "fitur non-nol", "CV acc"])
    C_star = float(tab[tab["CV acc"] >= tab["CV acc"].max() - 0.002].C.min())
    m = Pipeline([("sc", StandardScaler()), ("clf", l1_logreg(C_star))]).fit(D["X_train"], D["yb_tr"])
    w = pd.Series(m.named_steps["clf"].coef_.ravel(), index=X.columns)
    try:
        svm = Pipeline([("sc", StandardScaler()),
                        ("clf", LinearSVC(penalty="l1", dual=False, C=0.01, max_iter=5000))]
                       ).fit(D["X_train"], D["yb_tr"])
        svm_nz = int((svm.named_steps["clf"].coef_ != 0).sum())
    except Exception:
        svm_nz = None
    return tab, C_star, w, svm_nz


@st.cache_data(show_spinner="Melatih Random Forest ...")
def f_rf(_D, key):
    D = _D
    X, noise = D["X"], D["noise"]
    t0 = time.time()
    rf = RandomForestClassifier(300, min_samples_leaf=5, n_jobs=-1, random_state=D["seed"]
                                ).fit(D["X_train"], D["yb_tr"])
    t_fit = time.time() - t0
    imp = pd.Series(rf.feature_importances_, index=X.columns)
    sfm_n = int(SelectFromModel(rf, threshold="mean", prefit=True).get_support().sum())
    dt = DecisionTreeClassifier(max_depth=5, random_state=D["seed"]).fit(D["X_train"], D["yb_tr"])
    dt_imp = pd.Series(dt.feature_importances_, index=X.columns).sort_values(ascending=False).head(5)
    return dict(imp=imp, acc=rf.score(D["X_test"], D["yb_te"]), t_fit=t_fit, sfm_n=sfm_n, dt_imp=dt_imp)


def make_methods(topk, seed):
    mi_fn = partial(mutual_info_classif, random_state=seed)

    def pipe_with(sel):
        return Pipeline([("sc", StandardScaler()), ("sel", sel), ("clf", LogisticRegression(max_iter=2000))])

    return {
        "Semua fitur (baseline)": make_logit(),
        "Filter: Variance Threshold": pipe_with(VarianceThreshold(0.5)),
        "Filter: ANOVA F (k=15)": pipe_with(SelectKBest(f_classif, k=topk)),
        "Filter: Mutual Info (k=15)": pipe_with(SelectKBest(mi_fn, k=topk)),
        "Wrapper: RFE (k=15)": pipe_with(RFE(LogisticRegression(max_iter=1000), n_features_to_select=topk, step=10)),
        "Wrapper: Filter30 + SFS (k=10)": pipe_with(Pipeline([
            ("f", SelectKBest(f_classif, k=30)),
            ("s", SequentialFeatureSelector(LogisticRegression(max_iter=1000), n_features_to_select=10, cv=3, n_jobs=-1))])),
        "Embedded: L1-logistic (k=15)": pipe_with(SelectFromModel(l1_logreg(0.05), max_features=topk, threshold=-np.inf)),
        "Embedded: RF importance (k=15)": pipe_with(SelectFromModel(
            RandomForestClassifier(100, min_samples_leaf=5, n_jobs=-1, random_state=seed),
            max_features=topk, threshold=-np.inf)),
        "Extraction: PCA (k=15)": Pipeline([("sc", StandardScaler()), ("pca", PCA(topk)),
                                           ("clf", LogisticRegression(max_iter=2000))]),
        "Extraction: LDA (1 komponen)": Pipeline([("sc", StandardScaler()), ("lda", LDA(n_components=1)),
                                                 ("clf", LogisticRegression(max_iter=2000))]),
    }


@st.cache_data(show_spinner="Membandingkan metode (bisa beberapa menit) ...")
def f_compare(_D, key, topk, names):
    D = _D
    methods = {k: v for k, v in make_methods(topk, D["seed"]).items() if k in names}
    X_cmp, yb_cmp = D["X_cmp"], D["yb_cmp"]

    def n_feats(model):
        if "sel" in model.named_steps:
            s = model.named_steps["sel"]
            if isinstance(s, Pipeline):
                return 10
            return int(s.get_support().sum())
        if "pca" in model.named_steps:
            return model.named_steps["pca"].n_components_
        return 1 if "lda" in model.named_steps else D["X"].shape[1]

    rows = []
    for name, mdl in methods.items():
        cvr = cross_validate(mdl, X_cmp, yb_cmp, cv=D["cv5"], scoring="accuracy", n_jobs=1)
        t0 = time.time()
        mdl.fit(X_cmp, yb_cmp)
        ft = time.time() - t0
        proba = mdl.predict_proba(D["X_test"])[:, 1]
        rows.append((name, n_feats(mdl), cvr["test_score"].mean(),
                     accuracy_score(D["yb_te"], mdl.predict(D["X_test"])),
                     roc_auc_score(D["yb_te"], proba), ft))
    return pd.DataFrame(rows, columns=["metode", "n_fitur/komponen", "CV acc", "akurasi uji",
                                       "AUC uji", "waktu fit (dtk)"]).set_index("metode")


@st.cache_data(show_spinner="Menghitung PCA ...")
def f_pca(_D, key):
    D = _D
    Xtr_s = D["Xtr_s"]
    p = Xtr_s.shape[1]
    pca = PCA().fit(Xtr_s)
    ev, evr = pca.explained_variance_, pca.explained_variance_ratio_
    cum = np.cumsum(evr)
    ks_r = [k for k in [1, 2, 5, 10, 20, 40, 80, 120, 160] if k <= p]
    Xc = Xtr_s.values
    mu = Xc.mean(0)
    W = pca.components_
    mse = [np.mean((Xc - (mu + (Xc - mu) @ W[:k].T @ W[:k])) ** 2) for k in ks_r]
    load = pd.DataFrame(W[:2].T, index=D["X"].columns, columns=["PC1", "PC2"])
    return dict(evr=evr, cum=cum, kaiser=int((ev > 1).sum()),
                k80=int(np.searchsorted(cum, .80) + 1), k90=int(np.searchsorted(cum, .90) + 1),
                k95=int(np.searchsorted(cum, .95) + 1), ks_r=ks_r, mse=mse, load=load, p=p)


@st.cache_data(show_spinner="Memilih k lewat cross-validation ...")
def f_pca_cv(_D, key):
    D = _D
    p = D["X"].shape[1]
    ks = [k for k in [2, 5, 10, 15, 20, 30, 50, 80, 120] if k < min(p, len(D["X_cmp"]) * 0.7)]
    cv = [cross_val_score(Pipeline([("sc", StandardScaler()), ("pca", PCA(k)),
                                    ("clf", LogisticRegression(max_iter=2000))]),
                          D["X_cmp"], D["yb_cmp"], cv=D["cv5"]).mean() for k in ks]
    base = cross_val_score(make_logit(), D["X_cmp"], D["yb_cmp"], cv=D["cv5"]).mean()
    return ks, cv, base


@st.cache_data(show_spinner="Menghitung LDA ...")
def f_lda(_D, key):
    D = _D
    Xte_s = D["sc"].transform(D["X_test"])
    Z_pca = PCA(2).fit(D["Xtr_s"]).transform(Xte_s)
    lda = LDA(n_components=2).fit(D["Xtr_s"], D["y3_tr"])
    Z_lda = lda.transform(Xte_s)
    y3_te = D["y3_te"].values
    y3c = D["y_3"].loc[D["cmp_idx"]]
    mk = lambda step: Pipeline([("sc", StandardScaler()), step, ("clf", LogisticRegression(max_iter=2000))])
    cv_lda = cross_val_score(mk(("lda", LDA(n_components=2))), D["X_cmp"], y3c, cv=D["cv5"]).mean()
    cv_pca2 = cross_val_score(mk(("pca", PCA(2))), D["X_cmp"], y3c, cv=D["cv5"]).mean()
    cv_all = cross_val_score(make_logit(), D["X_cmp"], y3c, cv=D["cv5"]).mean()
    return dict(Z_pca=Z_pca, Z_lda=Z_lda, y3_te=y3_te,
                sil_pca=silhouette_score(Z_pca, y3_te), sil_lda=silhouette_score(Z_lda, y3_te),
                cv_all=cv_all, cv_pca2=cv_pca2, cv_lda=cv_lda)


@st.cache_data(show_spinner="Menjalankan Kernel PCA pada data pengguna ...")
def f_kpca_users(_D, key):
    D = _D
    p = D["X"].shape[1]
    out = {}
    for nm, pipe in {
        "PCA linear (10)": Pipeline([("sc", StandardScaler()), ("pca", PCA(10)),
                                     ("clf", LogisticRegression(max_iter=2000))]),
        "Kernel PCA RBF (10)": Pipeline([("sc", StandardScaler()),
                                         ("kp", KernelPCA(10, kernel="rbf", gamma=1.0 / p)),
                                         ("clf", LogisticRegression(max_iter=2000))]),
    }.items():
        out[nm] = cross_val_score(pipe, D["X_sub"], D["yb_sub"], cv=D["cv5"]).mean()
    return out


@st.cache_data(show_spinner="Menjalankan t-SNE ...")
def f_tsne(_D, key):
    D = _D
    sub = D["sub_idx"][:1500]
    Zs = PCA(30).fit_transform(D["sc"].transform(D["X_train"].loc[sub]))
    emb = TSNE(2, perplexity=30, init="pca", random_state=D["seed"]).fit_transform(Zs)
    return emb, D["y3_tr"].loc[sub].values


def build_selected(D, key, topk):
    """Kumpulkan subset fitur dari semua metode seleksi (dipakai di halaman Sintesis)."""
    flt = f_filters(D, key)
    sel = {
        "Korelasi |r|": top_names(flt["corr_y"], topk),
        "ANOVA F (f_classif)": top_names(flt["anova"].F_classif, topk),
        "ANOVA F (f_regression)": top_names(flt["anova"].F_regression, topk),
        "Chi-square": top_names(flt["chi"], topk),
        "Mutual Info (classif)": top_names(flt["mi_clf"], topk),
        "Mutual Info (regression)": top_names(flt["mi_reg"], topk),
    }
    efs, _ = f_efs(D, key)
    sel["EFS (8 kandidat)"] = list(efs.iloc[0].subset)
    seq = f_seq(D, key)
    sel["SFS (forward)"] = seq["SFS (forward)"]["features"]
    sel["SBS (backward)"] = seq["SBS (backward)"]["features"]
    sel["Stepwise (p-value)"] = f_step(D, key)[0]
    sel[f"RFE (k={topk})"] = f_rfe(D, key, topk)[0]
    lasso = f_lasso(D, key)
    for nm, col in [("Lasso (|coef| top-k)", "Lasso"), ("Elastic Net (|coef| top-k)", "ElasticNet")]:
        c = lasso["coef"][col].abs()
        sel[nm] = top_names(c[lasso["coef"][col] != 0], topk)
    w = f_l1(D, key)[2]
    sel["L1-logistic (top-k)"] = top_names(w.abs()[w != 0], topk)
    sel["Random Forest importance"] = top_names(f_rf(D, key)["imp"], topk)
    return sel


# =====================================================================================
# Halaman-halaman
# =====================================================================================
def page_data(D):
    st.header("1. Memuat dan membersihkan data")
    st.markdown(
        "Dataset **Book-Crossing** terdiri dari tiga tabel: buku, pengguna, dan rating. "
        "Pembersihan yang dilakukan: usia di luar 5-100 menjadi `NaN`, negara diambil dari kolom "
        "`location`, tahun terbit di luar 1900-2005 menjadi `NaN`, teks dinormalisasi (huruf kecil), "
        "dan ISBN diseragamkan.")
    c = st.columns(3)
    for col, (nm, sh) in zip(c, D["raw_shapes"].items()):
        col.metric(f"{nm} (mentah)", f"{sh[0]:,} baris", f"{sh[1]} kolom", delta_color="off")

    t1, t2, t3 = st.tabs(["Books", "Users", "Ratings"])
    with t1:
        st.write("Missing value:"); st.dataframe(D["books"].isna().sum().rename("n_missing").to_frame().T)
        st.dataframe(D["books"].head(10))
    with t2:
        st.write("Missing value:"); st.dataframe(D["users"].isna().sum().rename("n_missing").to_frame().T)
        st.dataframe(D["users"].head(10))
    with t3:
        st.write("Missing value:"); st.dataframe(D["ratings"].isna().sum().rename("n_missing").to_frame().T)
        st.dataframe(D["ratings"].head(10))

    st.subheader("Distribusi data")
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.4))
    D["ratings"].rating.value_counts().sort_index().plot.bar(ax=ax[0], color=BLUE)
    ax[0].set_title("Distribusi rating (0 = implisit)")
    D["users"].age.plot.hist(bins=50, ax=ax[1], color=ORANGE); ax[1].set_title("Distribusi usia (setelah dibersihkan)")
    D["books"].year.plot.hist(bins=50, ax=ax[2], color=CYAN); ax[2].set_title("Tahun terbit buku")
    fig.tight_layout(); show(fig)
    st.info(f"Rating implisit (0): **{(D['ratings'].rating == 0).mean():.1%}** dari semua interaksi.")


def page_features(D):
    X = D["X"]
    st.header("2. Rekayasa fitur")
    st.markdown("Satu baris = satu **pengguna**. Target: rata-rata rating eksplisit (`y_reg`), "
                "biner di atas/bawah median (`y_bin`), dan 3 kelas kuantil (`y_3`).")
    c = st.columns(4)
    c[0].metric("Interaksi dgn buku di katalog", f"{D['n_inter']:,}")
    c[1].metric("Pengguna terpilih", f"{D['n_target']:,}")
    c[2].metric("Interaksi dipakai", f"{D['n_used']:,}")
    c[3].metric("Dimensi X", f"{X.shape[0]:,} × {X.shape[1]}")
    st.write("Proporsi kelas `y_bin`:", D["y_bin"].value_counts(normalize=True).round(3).to_dict(),
             "| median rata-rata rating:", round(float(D["y_reg"].median()), 2))
    st.markdown(
        "**Kelompok fitur:** statistik interaksi (jumlah, % implisit), tahun terbit, popularitas buku/penulis, "
        "keragaman penulis/penerbit, proporsi per dekade, proporsi top-60 penerbit & penulis, usia, dan "
        "negara (top-15).  \n**Fitur buatan untuk pembelajaran:** `const_feature`, `near_const_flag`, "
        "`age_months` (redundan dengan `age`), dan `noise_01`-`noise_20`.")
    st.dataframe(X.head(20))
    st.write("Fitur buatan:", ", ".join(D["injected"]))
    st.subheader("Pembagian data")
    st.write(f"Train: **{len(D['X_train']):,}** | Test: **{len(D['X_test']):,}** | "
             f"subsampel wrapper: **{len(D['X_sub']):,}** | subsampel perbandingan: **{len(D['X_cmp']):,}**")


def page_curse(D):
    st.header("3. The Curse of Dimensionality")
    dims, rc = f_contrast(D["seed"])
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.plot(range(len(dims)), rc, "o-", color=BLUE)
    ax.set_xticks(range(len(dims))); ax.set_xticklabels(dims); ax.set_yscale("log")
    ax.set_xlabel("jumlah dimensi p"); ax.set_ylabel("relative contrast (log)")
    ax.set_title("Jarak kehilangan kontras saat dimensi naik")
    c1, c2 = st.columns([3, 2])
    with c1: show(fig)
    with c2: st.dataframe(pd.Series(rc, index=dims, name="relative contrast").round(3).to_frame())

    st.subheader("Fitur noise merusak k-NN")
    st.caption("30 fitur nyata terbaik (menurut ANOVA) + sejumlah fitur noise, dievaluasi dengan CV 5-fold.")
    if gate("Jalankan eksperimen k-NN", "run_knn"):
        counts, accs = f_knn_noise(D, D["key"])
        fig, ax = plt.subplots(figsize=(6.5, 3.8))
        ax.plot(range(len(counts)), accs, "o-", color=ORANGE)
        for i, a in enumerate(accs):
            ax.annotate(f"{a:.3f}", (i, a), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
        ax.set_xticks(range(len(counts))); ax.set_xticklabels(counts)
        ax.set_xlabel("fitur noise yang ditambahkan ke 30 fitur nyata"); ax.set_ylabel("akurasi CV 5-fold")
        ax.set_title("Fitur noise menurunkan akurasi k-NN")
        show(fig)


def page_selext(D):
    st.header("4. Selection versus Extraction")
    cols8 = ["age", "log_n_interactions", "pct_implicit", "year_mean", "log_book_pop_mean",
             "log_author_pop_mean", "title_len_mean", "author_diversity"]
    D8 = D["X_train"][cols8]
    st.subheader("SELECTION → kolom asli tidak berubah")
    st.dataframe(D8[["age", "year_mean", "log_book_pop_mean"]].head(3).round(2))
    st.subheader("EXTRACTION → z = Wᵀx (semua fitur asli terlibat)")
    p3 = PCA(3).fit(StandardScaler().fit_transform(D8))
    st.dataframe(pd.DataFrame(p3.components_, index=["z1", "z2", "z3"], columns=cols8).round(2))
    st.caption("Baris = komponen, kolom = bobot tiap fitur asli.")


def page_filter(D):
    key, X, noise = D["key"], D["X"], D["noise"]
    topk = st.session_state["topk"]
    flt = f_filters(D, key)
    corr_y, anova, chi_sc = flt["corr_y"], flt["anova"], flt["chi"]
    st.header("5. Filter Methods")
    tabs = st.tabs(["Variance Threshold", "Korelasi", "ANOVA F", "Chi-square", "Mutual Information",
                    "Kesepakatan metode", "Keterbatasan filter"])

    with tabs[0]:
        thr = st.select_slider("Threshold variansi (MinMax)", [1e-5, 1e-4, 1e-3, 1e-2, 5e-2], value=1e-3,
                               format_func=lambda v: f"{v:g}")
        vt = VarianceThreshold(threshold=thr).fit(D["Xtr_mm"])
        var_s = pd.Series(D["Xtr_mm"].var().values, index=X.columns).sort_values()
        removed = list(X.columns[~vt.get_support()])
        st.write(f"Fitur dibuang (variansi MinMax < {thr:g}): **{len(removed)}** dari {X.shape[1]}")
        st.write(removed)
        st.write("Variansi terkecil:"); st.dataframe(var_s.head(8).rename("variansi").to_frame())

    with tabs[1]:
        show(plot_top(corr_y, "|Korelasi| dengan rata-rata rating (y_reg)", noise, topk))
        thr = st.slider("Ambang |r| antar-fitur", 0.5, 0.99, 0.9, 0.01)
        C = D["Xtr_s"].drop(columns=["const_feature"]).corr().abs()
        pairs = C.where(np.triu(np.ones(C.shape), 1).astype(bool)).stack().sort_values(ascending=False)
        st.write(f"Pasangan fitur sangat berkorelasi (|r| > {thr}):")
        st.dataframe(pairs[pairs > thr].round(4).rename("|r|").to_frame())

    with tabs[2]:
        st.dataframe(anova.sort_values("F_classif", ascending=False).head(10).round(4))
        st.write(f"Fitur signifikan (p<0.05): classif = **{int((anova.p_classif < .05).sum())}**, "
                 f"regression = **{int((anova.p_regression < .05).sum())}**")
        st.write("Fitur noise yang lolos p<0.05 (false positive):",
                 [c for c in noise if anova.loc[c, "p_classif"] < .05])
        show(plot_top(anova.F_classif, "ANOVA F (f_classif) – target y_bin", noise, topk))

    with tabs[3]:
        show(plot_top(chi_sc, "Skor χ² (MinMax) – target y_bin", noise, topk))

    with tabs[4]:
        st.write(f"Waktu hitung MI: {flt['mi_time']:.1f} dtk")
        show(plot_top(flt["mi_clf"], "Mutual information (classif)", noise, topk))

    with tabs[5]:
        S = pd.DataFrame({"|corr|": corr_y, "F_clf": anova.F_classif, "F_reg": anova.F_regression,
                          "chi2": chi_sc, "MI_clf": flt["mi_clf"], "MI_reg": flt["mi_reg"]})
        R = S.corr(method="spearman")
        fig, ax = plt.subplots(figsize=(5.2, 4.4))
        im = ax.imshow(R, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(R))); ax.set_xticklabels(R.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(R))); ax.set_yticklabels(R.columns)
        for i in range(len(R)):
            for j in range(len(R)):
                ax.text(j, i, f"{R.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
        ax.grid(False); fig.colorbar(im); ax.set_title("Korelasi peringkat antar skor filter"); fig.tight_layout()
        show(fig)

    with tabs[6]:
        st.markdown("Contoh XOR: `y = 1` jika `x1·x2 > 0`. Tiap fitur sendirian tidak informatif, "
                    "tetapi **pasangannya** sangat informatif.")
        rng = np.random.default_rng(0)
        x = rng.normal(size=(3000, 2)); yx = (x[:, 0] * x[:, 1] > 0).astype(int)
        st.write("F per fitur:", np.round(f_classif(x, yx)[0], 3).tolist())
        st.write("MI per fitur:", np.round(mutual_info_classif(x, yx, random_state=0), 4).tolist())
        tree = DecisionTreeClassifier(max_depth=4, random_state=0)
        res = {n: cross_val_score(tree, x[:, c], yx, cv=5).mean()
               for n, c in {"hanya x1": [0], "hanya x2": [1], "x1 + x2 (subset)": [0, 1]}.items()}
        st.dataframe(pd.Series(res, name="akurasi pohon").round(3).to_frame())


def page_wrapper(D):
    key, X = D["key"], D["X"]
    st.header("6. Wrapper Methods")
    st.caption("Wrapper mengevaluasi subset fitur dengan model sungguhan, sehingga jauh lebih lambat. "
               "Hasil di-cache setelah dijalankan.")
    tabs = st.tabs(["EFS", "SFS & SBS", "Stepwise", "RFE / RFECV", "Waspada overfitting"])

    with tabs[0]:
        if gate("Jalankan EFS (8 kandidat, 255 subset)", "run_efs"):
            efs, t = f_efs(D, key)
            best = efs.iloc[0]
            st.write(f"{len(efs)} subset dievaluasi dalam {t:.1f} dtk. Untuk p={X.shape[1]} ada "
                     f"2^{X.shape[1]} ≈ {2.0 ** X.shape[1]:.1e} subset.")
            st.success(f"Subset terbaik EFS: {list(best.subset)} → CV acc {best.cv_acc:.4f}")
            fig, ax = plt.subplots(figsize=(7, 3.8))
            rng = np.random.default_rng(D["seed"])
            ax.scatter(efs["size"] + rng.normal(0, .05, len(efs)), efs.cv_acc, s=12, alpha=.5, color=BLUE)
            bs = efs.groupby("size").cv_acc.max()
            ax.plot(bs.index, bs.values, "o-", color=ORANGE, label="terbaik per ukuran")
            ax.set_xlabel("ukuran subset"); ax.set_ylabel("akurasi CV")
            ax.set_title("EFS: semua subset dari 8 kandidat"); ax.legend()
            show(fig)

    with tabs[1]:
        if gate("Jalankan SFS & SBS (30 kandidat → 8 fitur)", "run_seq"):
            seq = f_seq(D, key)
            for n, r in seq.items():
                st.write(f"**{n}** ({r['time_s']:.1f} dtk): {r['features']}")
            a, b = set(seq["SFS (forward)"]["features"]), set(seq["SBS (backward)"]["features"])
            st.info(f"Kemiripan (Jaccard) SFS vs SBS: {len(a & b) / len(a | b):.2f} → dua arah pencarian "
                    "greedy bisa menghasilkan subset berbeda.")

    with tabs[2]:
        if gate("Jalankan stepwise", "run_step"):
            feats, r2 = f_step(D, key)
            st.write(f"Stepwise memilih **{len(feats)}** fitur:", feats)
            st.write(f"R² model akhir = {r2:.4f}")

    with tabs[3]:
        topk = st.session_state["topk"]
        if gate("Jalankan RFE & RFECV", "run_rfe"):
            feats, t, nf, scores, nopt = f_rfe(D, key, topk)
            st.write(f"RFE memilih {topk} fitur dalam {t:.1f} dtk:", feats)
            fig, ax = plt.subplots(figsize=(7, 3.6))
            ax.plot(nf, scores, "o-", color=BLUE); ax.axvline(nopt, color=RED, ls="--")
            ax.set_xlabel("jumlah fitur"); ax.set_ylabel("akurasi CV")
            ax.set_title(f"RFECV: jumlah fitur optimal = {nopt}")
            show(fig)

    with tabs[4]:
        if gate("Hitung selisih CV vs akurasi uji (butuh EFS, SFS & SBS)", "run_over"):
            efs, _ = f_efs(D, key)
            seq = f_seq(D, key)
            sets = [("EFS", list(efs.iloc[0].subset)), ("SFS", seq["SFS (forward)"]["features"]),
                    ("SBS", seq["SBS (backward)"]["features"])]
            rows = []
            for name, feats in sets:
                cv_in = cross_val_score(make_logit(), D["X_sub"][feats], D["yb_sub"], cv=D["cv5"]).mean()
                m = make_logit().fit(D["X_sub"][feats], D["yb_sub"])
                te = accuracy_score(D["yb_te"], m.predict(D["X_test"][feats]))
                rows.append((name, len(feats), cv_in, te, cv_in - te))
            st.dataframe(pd.DataFrame(rows, columns=["metode", "n_fitur", "CV(saat seleksi)",
                                                     "akurasi uji", "selisih"]).round(4))
            st.caption("Selisih positif = skor CV saat seleksi terlalu optimistis (overfitting pada proses seleksi).")


def page_embedded(D):
    key, X, noise = D["key"], D["X"], D["noise"]
    topk = st.session_state["topk"]
    st.header("7. Embedded Methods")
    tabs = st.tabs(["Geometri L1 vs L2", "Lasso / Ridge / Elastic Net", "L1-logistic & L1-SVM", "Random Forest"])

    with tabs[0]:
        b_ols = np.array([2.0, 0.4]); A = np.array([[1.0, 0.2], [0.2, 1.0]])
        loss = lambda b: (b - b_ols) @ A @ (b - b_ols)
        t = 1.0
        lasso_sol = minimize(loss, [0.5, 0.0], constraints=[{"type": "ineq", "fun": lambda b: t - abs(b[0]) - abs(b[1])}], method="SLSQP").x
        ridge_sol = minimize(loss, [0.5, 0.3], constraints=[{"type": "ineq", "fun": lambda b: t ** 2 - b @ b}], method="SLSQP").x
        g1, g2 = np.meshgrid(np.linspace(-1.5, 2.6, 300), np.linspace(-1.3, 1.3, 300))
        Z = np.array([[loss(np.array([a, b])) for a in g1[0]] for b in g2[:, 0]])
        fig, ax = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
        for a, ttl, sol in [(ax[0], "Lasso: |β1|+|β2| ≤ t", lasso_sol), (ax[1], "Ridge: β1²+β2² ≤ t²", ridge_sol)]:
            a.contour(g1, g2, Z, levels=[0.1, 0.4, 1.0, 2.0, 3.5], colors=BLUE, linewidths=.8)
            if "Lasso" in ttl: a.fill([t, 0, -t, 0], [0, t, 0, -t], color=ORANGE, alpha=.35)
            else: a.add_patch(plt.Circle((0, 0), t, color=CYAN, alpha=.35))
            a.scatter(*b_ols, color=BLUE); a.annotate("OLS", b_ols, xytext=(6, 4), textcoords="offset points")
            a.scatter(*sol, color=RED, zorder=5)
            a.annotate(f"({sol[0]:.2f}, {sol[1]:.2f})", sol, xytext=(6, -14), textcoords="offset points", color=RED)
            a.axhline(0, color="gray", lw=.6); a.axvline(0, color="gray", lw=.6)
            a.set_title(ttl); a.set_xlabel("β1"); a.set_aspect("equal")
        ax[0].set_ylabel("β2"); fig.tight_layout(); show(fig)
        st.info(f"Anggaran t=1 yang sama → Lasso: {np.round(lasso_sol, 2)} (β2 = {lasso_sol[1]:.4f}) | "
                f"Ridge: {np.round(ridge_sol, 2)} (keduanya non-nol)")

    with tabs[1]:
        if gate("Jalankan Lasso / Ridge / Elastic Net", "run_lasso"):
            r = f_lasso(D, key)
            fig, ax = plt.subplots(figsize=(7.5, 4))
            ax.plot(r["alphas"], r["coefs"].T, lw=.8); ax.set_xscale("log"); ax.axvline(r["lcv_alpha"], color="k", ls="--")
            ax.set_xlabel("alpha (penalti)"); ax.set_ylabel("koefisien")
            ax.set_title("Lasso path (garis putus = alpha terbaik dari CV)")
            show(fig)
            st.dataframe(r["summ"].round(4))
            st.write(f"ElasticNet l1_ratio terbaik = {r['l1_ratio']}")
            st.markdown("**Fitur berkorelasi: `age` vs `age_months` (redundan buatan)**")
            st.dataframe(r["coef"].loc[["age", "age_months"]].round(4))
            st.caption("Lasso mempertahankan hanya salah satu dari pasangan berkorelasi; Ridge membagi bobot "
                       "sama rata. Efek 'menjaga kelompok' Elastic Net muncul saat porsi L2 lebih besar:")
            st.dataframe(r["en2_coef"][["age", "age_months"]].round(4).rename("l1_ratio=0.2").to_frame())
            st.caption(f"(l1_ratio=0.2: {r['en2_nz']} koefisien non-nol)")

    with tabs[2]:
        if gate("Jalankan L1-logistic & L1-SVM", "run_l1"):
            tab, C_star, w, svm_nz = f_l1(D, key)
            st.dataframe(tab.round(4))
            st.write(f"C terpilih = **{C_star}** → {int((w != 0).sum())} fitur non-nol; noise yang ikut "
                     f"terpilih: {[c for c in noise if w[c] != 0]}")
            show(plot_top(w.abs(), f"|Koefisien| L1-logistic (C={C_star})", noise, topk))
            st.write("L1-SVM (C=0.01):", f"{svm_nz} fitur non-nol" if svm_nz is not None else "dilewati")

    with tabs[3]:
        if gate("Latih Random Forest", "run_rf"):
            r = f_rf(D, key)
            imp = r["imp"]
            nz = imp[[c for c in noise if c.startswith("noise")]]
            st.write(f"RF dilatih dalam {r['t_fit']:.1f} dtk | akurasi uji = {r['acc']:.4f}")
            st.write(f"Importance noise terbesar = {nz.max():.5f}; rata-rata noise = {nz.mean():.5f}")
            st.write(f"Fitur di atas lantai noise: **{int((imp > nz.max()).sum())}** dari {len(imp)}")
            show(plot_top(imp, "Importance Random Forest", noise, topk))
            st.write(f"SelectFromModel(threshold='mean') mempertahankan **{r['sfm_n']}** fitur")
            st.write("Decision tree (depth 5) – 5 fitur terpenting:")
            st.dataframe(r["dt_imp"].round(4).rename("importance").to_frame())


def page_compare(D):
    topk = st.session_state["topk"]
    st.header("8. Membandingkan metode")
    st.caption("Semua model: logistic regression pada subsampel 5.000 pengguna, CV 5-fold, lalu diuji di data uji.")
    all_names = list(make_methods(topk, D["seed"]).keys())
    default = [n for n in all_names if "SFS" not in n]
    names = st.multiselect("Metode yang dibandingkan (SFS paling lambat)", all_names, default=default)
    if names and gate("Jalankan perbandingan", "run_cmp"):
        df = f_compare(D, D["key"], topk, tuple(names))
        st.dataframe(df.round(4))
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
        order = df.index[::-1]
        ax[0].barh(order, df.loc[order, "akurasi uji"], color=BLUE)
        ax[0].set_xlim(df["akurasi uji"].min() - .02, df["akurasi uji"].max() + .02)
        ax[0].set_title("Akurasi data uji")
        ax[1].barh(order, df.loc[order, "waktu fit (dtk)"], color=ORANGE)
        ax[1].set_title("Waktu fit (dtk)"); ax[1].set_xscale("log")
        fig.tight_layout(); show(fig)


def page_extraction(D):
    key, X = D["key"], D["X"]
    st.header("9. Feature Extraction")
    tabs = st.tabs(["PCA & scree plot", "PCA sebagai kompresi", "LDA", "Kernel PCA", "t-SNE"])
    pc = f_pca(D, key)

    with tabs[0]:
        st.write(f"PC1 saja: **{pc['evr'][0]:.1%}** | komponen untuk 80%/90%/95%: "
                 f"**{pc['k80']}/{pc['k90']}/{pc['k95']}** | Kaiser (eigenvalue>1): **{pc['kaiser']}** dari p={pc['p']}")
        K = min(40, len(pc["evr"])); xs = np.arange(1, K + 1)
        fig, ax = plt.subplots(figsize=(7.5, 4))
        ax.plot(xs, pc["evr"][:K], "o-", color=BLUE, label="variansi per komponen")
        ax.plot(xs, pc["cum"][:K], "o-", color=ORANGE, label="kumulatif")
        for q in (.9, .95): ax.axhline(q, color="gray", ls=":", lw=.8)
        ax.set_xlabel("komponen k"); ax.set_ylabel("share of total variance"); ax.set_title("Scree plot"); ax.legend()
        show(fig)
        if gate("Pilih k lewat cross-validation", "run_pcacv"):
            ks, cv, base = f_pca_cv(D, key)
            fig, ax = plt.subplots(figsize=(7, 3.8))
            ax.plot(range(len(ks)), cv, "o-", color=BLUE, label="PCA(k) + logistic")
            ax.axhline(base, color=RED, ls="--", label=f"semua fitur ({base:.3f})")
            ax.set_xticks(range(len(ks))); ax.set_xticklabels(ks)
            ax.set_xlabel("jumlah komponen k"); ax.set_ylabel("akurasi CV")
            ax.set_title("Memilih k lewat cross-validation"); ax.legend()
            show(fig)
            st.write("k terbaik (CV):", ks[int(np.argmax(cv))])
        load = pc["load"]
        st.markdown("**Bobot terbesar PC1 dan PC2** (interpretabilitas lebih rendah: tiap PC campuran banyak fitur)")
        st.dataframe(pd.DataFrame({
            "PC1": load.PC1.abs().sort_values(ascending=False).head(5).index.to_list(),
            "PC2": load.PC2.abs().sort_values(ascending=False).head(5).index.to_list()}))

    with tabs[1]:
        fig, ax = plt.subplots(figsize=(6.5, 3.6))
        ax.plot(pc["ks_r"], pc["mse"], "o-", color=BLUE); ax.set_xscale("log")
        ax.set_xlabel("k komponen"); ax.set_ylabel("MSE rekonstruksi (data standar)")
        ax.set_title("Galat rekonstruksi PCA pada data pengguna")
        show(fig)
        st.subheader("Replikasi “PCA as Compression” pada dataset Digits")
        Xd = load_digits().data; mu_d = Xd.mean(0); pd_ = PCA().fit(Xd); cd = np.cumsum(pd_.explained_variance_ratio_)
        ks_d = [2, 5, 10, 20, 40]; sample = [0, 1, 2, 3]
        fig, ax = plt.subplots(4, 6, figsize=(9, 6.2))
        for r, i in enumerate(sample):
            for c, k in enumerate(ks_d):
                rec = mu_d + (Xd[i] - mu_d) @ pd_.components_[:k].T @ pd_.components_[:k]
                ax[r, c].imshow(rec.reshape(8, 8), cmap="gray_r"); ax[r, c].axis("off")
                if r == 0: ax[r, c].set_title(f"k={k}\n{cd[k - 1]:.0%} var", fontsize=8)
            ax[r, 5].imshow(Xd[i].reshape(8, 8), cmap="gray_r"); ax[r, 5].axis("off")
            if r == 0: ax[r, 5].set_title("asli\n64 piksel", fontsize=8)
        fig.tight_layout(); show(fig)

    with tabs[2]:
        if gate("Jalankan LDA pada data pengguna", "run_lda"):
            r = f_lda(D, key)
            fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
            for a, Z, ttl in [(ax[0], r["Z_pca"], "PCA (unsupervised): max variance"),
                              (ax[1], r["Z_lda"], "LDA (supervised): max class separation")]:
                for k_, col in CMAP3.items():
                    a.scatter(Z[r["y3_te"] == k_, 0], Z[r["y3_te"] == k_, 1], s=5, alpha=.5, color=col, label=f"kelas {k_}")
                a.set_title(ttl)
            ax[1].legend(markerscale=3); fig.tight_layout(); show(fig)
            st.write(f"Silhouette (data uji): PCA = **{r['sil_pca']:.3f}** | LDA = **{r['sil_lda']:.3f}**")
            st.write(f"Akurasi CV 3-kelas (acak = 0.333): semua fitur = **{r['cv_all']:.3f}** | "
                     f"PCA(2) = **{r['cv_pca2']:.3f}** | LDA(2) = **{r['cv_lda']:.3f}**")
            st.caption("LDA harus dievaluasi dengan cross-validation karena fiturnya dilatih memakai label "
                       "(kebocoran bila dinilai di data latih).")
        st.markdown("Pada data Book-Crossing, rating seseorang sulit diprediksi dari profil bacaannya sehingga "
                    "pemisahan kelasnya lemah. Replikasi **LDA: Supervised Extraction** pada dataset **Wine**:")
        wine = load_wine(); Xw = StandardScaler().fit_transform(wine.data); yw = wine.target
        Zp, Zl = PCA(2).fit_transform(Xw), LDA(n_components=2).fit_transform(Xw, yw)
        fig, ax = plt.subplots(1, 2, figsize=(10.5, 4))
        for a, Z, ttl in [(ax[0], Zp, f"PCA: silhouette {silhouette_score(Zp, yw):.2f}"),
                          (ax[1], Zl, f"LDA: silhouette {silhouette_score(Zl, yw):.2f}")]:
            for k_, col in CMAP3.items():
                a.scatter(Z[yw == k_, 0], Z[yw == k_, 1], s=14, color=col, label=f"cultivar {k_ + 1}")
            a.set_title(ttl)
        ax[1].legend(); fig.tight_layout(); show(fig)

    with tabs[3]:
        gamma = st.slider("γ (RBF) untuk data lingkaran", 0.5, 10.0, 4.0, 0.5)
        Xc_, yc_ = make_circles(500, factor=0.3, noise=0.05, random_state=D["seed"])
        lin = PCA(2).fit_transform(Xc_); ker = KernelPCA(2, kernel="rbf", gamma=gamma).fit_transform(Xc_)
        fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
        for a, Z, ttl in [(ax[0], Xc_, "Data asli (2-D)"), (ax[1], lin, "PCA linear"),
                          (ax[2], ker, f"Kernel PCA (RBF, γ={gamma:g})")]:
            a.scatter(Z[:, 0], Z[:, 1], c=np.where(yc_ == 0, BLUE, ORANGE), s=8); a.set_title(ttl)
        fig.tight_layout(); show(fig)
        res = {nm: cross_val_score(LogisticRegression(), Z[:, :1], yc_, cv=5).mean()
               for nm, Z in [("PCA linear (1 komp.)", lin), ("Kernel PCA (1 komp.)", ker)]}
        st.dataframe(pd.Series(res, name="akurasi logistic").round(3).to_frame())
        if gate("Bandingkan PCA vs Kernel PCA pada data pengguna", "run_kpca"):
            st.dataframe(pd.Series(f_kpca_users(D, key), name="akurasi CV").round(4).to_frame())

    with tabs[4]:
        st.caption("t-SNE hanya untuk visualisasi (1.500 pengguna, PCA 30 komponen sebagai pra-proses).")
        if gate("Jalankan t-SNE", "run_tsne"):
            emb, ylab = f_tsne(D, key)
            fig, ax = plt.subplots(figsize=(5.8, 4.6))
            for k_, col in CMAP3.items():
                ax.scatter(emb[ylab == k_, 0], emb[ylab == k_, 1], s=5, color=col, alpha=.6, label=f"kelas {k_}")
            ax.legend(markerscale=3); ax.set_title("t-SNE (1.500 pengguna) – visualisasi saja")
            show(fig)


def page_synthesis(D):
    key, topk, noise, X = D["key"], st.session_state["topk"], D["noise"], D["X"]
    st.header("10. Sintesis: fitur mana yang paling konsisten dipilih?")
    st.caption("Menjalankan semua metode seleksi (yang belum pernah dijalankan akan dihitung sekarang; "
               "EFS, SFS/SBS, dan RFECV memerlukan waktu).")
    if not gate("Hitung konsensus semua metode", "run_syn"):
        return
    selected = build_selected(D, key, topk)
    sel_df = pd.DataFrame({m: pd.Series(1, index=f) for m, f in selected.items()}).reindex(X.columns).fillna(0).astype(int)
    votes = sel_df.sum(axis=1).sort_values(ascending=False)
    st.write(f"**{len(selected)}** metode diikutkan.")
    c1, c2 = st.columns([3, 2])
    with c1:
        top = votes.head(20)[::-1]
        fig, ax = plt.subplots(figsize=(7, 5.2))
        ax.barh(top.index, top.values, color=[RED if i in noise else BLUE for i in top.index])
        ax.set_xlabel("jumlah metode yang memilih"); ax.set_title("Konsensus seleksi fitur (merah = noise)")
        fig.tight_layout(); show(fig)
    with c2:
        st.write("Top 20 fitur menurut jumlah suara:")
        st.dataframe(votes.head(20).rename("suara").to_frame())
    st.subheader("False positive per metode")
    st.dataframe(pd.DataFrame({"n_terpilih": sel_df.sum(), "noise/konstan terpilih": sel_df.loc[noise].sum()}))
    with st.expander("Subset fitur tiap metode"):
        for m, f in selected.items():
            st.markdown(f"**{m}** ({len(f)}): {', '.join(f)}")


def page_conclusion(D):
    st.header("Kesimpulan")
    st.markdown(
        "Berdasarkan hasil analisis dan penerapan metode Feature Selection, dapat disimpulkan bahwa pemilihan "
        "metode seleksi fitur perlu mempertimbangkan ukuran dataset, jenis model, tingkat interpretabilitas, "
        "dan sumber daya komputasi.\n\n"
        "- **Filter**: lebih cepat dan efisien untuk dataset besar, tetapi kurang mampu mendeteksi interaksi antarfitur.\n"
        "- **Wrapper**: menangkap interaksi fitur dengan lebih baik, namun biaya komputasi besar dan berisiko overfitting.\n"
        "- **Embedded** (Lasso, Elastic Net, Random Forest): seleksi fitur terjadi selama proses pelatihan model.\n"
        "- **Feature Extraction** (PCA, LDA, Kernel PCA): mereduksi dimensi, tetapi mengurangi kemudahan interpretasi fitur asli.\n\n"
        "Penambahan fitur noise dapat menurunkan performa model, sementara seleksi fitur hanya memberikan "
        "peningkatan akurasi yang relatif kecil (sekitar 0,56-0,59 dengan R² pengujian sekitar 0,06). Manfaat "
        "utama Feature Selection di sini bukan hanya meningkatkan akurasi, tetapi juga mengurangi jumlah fitur, "
        "meningkatkan efisiensi komputasi, dan menghasilkan model yang lebih sederhana serta mudah diinterpretasikan.")
    st.caption("Angka pada kesimpulan berasal dari hasil notebook asli; hasil di aplikasi dapat sedikit berbeda "
               "tergantung pengaturan sidebar.")


PAGES = {
    "1. Data & pembersihan": page_data,
    "2. Rekayasa fitur": page_features,
    "3. Curse of dimensionality": page_curse,
    "4. Selection vs Extraction": page_selext,
    "5. Filter methods": page_filter,
    "6. Wrapper methods": page_wrapper,
    "7. Embedded methods": page_embedded,
    "8. Perbandingan metode": page_compare,
    "9. Feature extraction": page_extraction,
    "10. Sintesis": page_synthesis,
    "Kesimpulan": page_conclusion,
}


def main():
    st.title("📚 Feature Selection & Feature Extraction")
    st.caption("Dataset Book-Crossing · Tugas DMV · Farros Raditya Ahmad Sifana (5003241017)")

    st.sidebar.header("Pengaturan")
    data_dir = get_data_dir()
    seed = int(st.sidebar.number_input("Random state", 0, 9999, 42))
    min_expl = int(st.sidebar.slider("Min. rating eksplisit per pengguna", 1, 20, 5))
    st.session_state["topk"] = int(st.sidebar.slider("Top-K fitur", 5, 30, 15))
    st.sidebar.divider()
    page = st.sidebar.radio("Halaman", list(PAGES.keys()))

    if page == "Kesimpulan":
        PAGES[page](None)
        return
    if data_dir is None:
        st.info("Letakkan **BX-Books.csv**, **BX-Users.csv**, dan **BX-Book-Ratings.csv** di folder yang sama "
                "dengan `app.py` (atau folder `data/`), atau upload ketiganya lewat sidebar.")
        st.stop()

    D = build_bundle(data_dir, dir_signature(data_dir), seed, min_expl)
    PAGES[page](D)


if __name__ == "__main__":
    main()
