import streamlit as st
import joblib
import pandas as pd
import numpy as np

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(
    page_title="Anime Recommender",
    layout="wide"
)

# =========================================================
# LOAD SYSTEM
# =========================================================
@st.cache_resource
def load_system():
    return joblib.load("anime_recommender_final.joblib")

sys = load_system()

X_train = sys["X_train"]
X_test = sys["X_test"]
train_meta = sys["train_meta"]
test_meta = sys["test_meta"]
anime_train = sys["anime_train"]
anime_test = sys["anime_test"]
knn = sys["knn_model"]
genre_col = sys["genre_col"]

# =========================================================
# HELPERS
# =========================================================
def is_same_anime(name1, name2):
    name1 = name1.lower().strip()
    name2 = name2.lower().strip()

    if name1 == name2:
        return True

    suffixes = [
        ' movie', ' ova', ' special', ' tv', ' ona',
        ': season', ' season', ' 2nd season', ' 3rd season',
        ' part', ' ii', ' iii', ' iv'
    ]

    for s in suffixes:
        name1 = name1.split(s)[0]
        name2 = name2.split(s)[0]

    if name1 == name2:
        return True

    if len(name1) > 5 and name1 in name2:
        return True
    if len(name2) > 5 and name2 in name1:
        return True

    return False


def find_anime(anime_name):
    test_matches = test_meta[test_meta["name"].str.contains(anime_name, case=False, na=False)]
    train_matches = train_meta[train_meta["name"].str.contains(anime_name, case=False, na=False)]

    # Exact match priority
    for src, matches in [("TEST", test_matches), ("TRAIN", train_matches)]:
        for anime_id, row in matches.iterrows():
            if row["name"].lower().strip() == anime_name.lower().strip():
                return anime_id, src

    # Fallback to first partial
    if not test_matches.empty:
        return test_matches.index[0], "TEST"
    if not train_matches.empty:
        return train_matches.index[0], "TRAIN"

    return None, None


def get_recommendations(anime_id, source, k=5):
    if source == "TEST":
        idx = list(anime_test.index).index(anime_id)
        query_vec = X_test[idx]
        query_data = anime_test.loc[anime_id]
    else:
        idx = list(anime_train.index).index(anime_id)
        query_vec = X_train[idx]
        query_data = anime_train.loc[anime_id]

    query_name = query_data["name"]

    distances, indices = knn.kneighbors(
        query_vec.reshape(1, -1),
        n_neighbors=k * 3
    )

    recs = []
    for i, d in zip(indices[0], distances[0]):
        rec_id = anime_train.index[i]
        rec_data = anime_train.loc[rec_id]

        if is_same_anime(query_name, rec_data["name"]):
            continue

        recs.append((rec_data, 1 - d))

        if len(recs) == k:
            break

    return query_data, recs


def shorten(text, max_len=150):
    if pd.isna(text):
        return "N/A"
    text = str(text)
    return text if len(text) <= max_len else text[:max_len] + "..."


# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.title("🔎 Anime Search")

all_anime_names = pd.concat([
    train_meta["name"],
    test_meta["name"]
]).dropna().unique()

all_anime_names = np.sort(all_anime_names)

anime_name = st.sidebar.selectbox(
    "Type or select an anime",
    options=all_anime_names,
    index=None,
    placeholder="Start typing..."
)

k = st.sidebar.slider(
    "Number of recommendations",
    min_value=3,
    max_value=10,
    value=5
)

run = st.sidebar.button("Recommend")

# =========================================================
# MAIN UI
# =========================================================
st.title("🎌 Anime Recommender System")
st.caption("Content-based • Cosine similarity • k-NN")

if run:
    if not anime_name:
        st.warning("Please select an anime.")
    else:
        anime_id, source = find_anime(anime_name)

        if anime_id is None:
            st.error("Anime not found.")
        else:
            query_data, recs = get_recommendations(anime_id, source, k)

            st.subheader(f"🎯 {query_data['name']}")
            st.caption(f"Source: {source}")

            st.markdown(f"**Type:** {query_data.get('type','N/A')}")
            st.markdown(f"**{genre_col.capitalize()}:** {shorten(query_data.get(genre_col))}")
            st.markdown(f"**Rating:** {query_data.get('rating','N/A')}")
            st.markdown("---")

            if not recs:
                st.warning("No distinct recommendations found.")
            else:
                st.subheader("📋 Similar Anime")

                for i, (rec, sim) in enumerate(recs, 1):
                    st.markdown(f"### {i}. {rec['name']}")
                    st.markdown(f"**Similarity:** `{sim:.4f}`")
                    st.markdown(f"**Type:** {rec.get('type','N/A')}")
                    st.markdown(f"**{genre_col.capitalize()}:** {shorten(rec.get(genre_col))}")
                    st.markdown(f"**Rating:** {rec.get('rating','N/A')}")
                    st.markdown("---")
