import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import ast 
import ssl

# --- 1. SỬA LỖI SSL (Bắt buộc cho Mac) ---
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# --- 2. CẤU HÌNH TRANG ---
st.set_page_config(page_title="Cinema AI Pro", page_icon="🍿", layout="wide")
st.title("🍿 Cinema AI - Gợi ý phim (Phiên bản Ổn định)")
st.markdown("Hệ thống gợi ý phim sử dụng Machine Learning (Content-Based Filtering).")

# --- 3. TẢI DỮ LIỆU (CƠ CHẾ 3 LỚP) ---
@st.cache_data
def load_data():
    df = pd.DataFrame()
    
    # Danh sách các link dự phòng
    urls = [
        "https://raw.githubusercontent.com/kavyappan/Movie-Recommendation-System/main/tmdb_5000_movies.csv", # Link mới 1
        "https://raw.githubusercontent.com/campusx-official/Movie-Recommender-System-Project/main/tmdb_5000_movies.csv", # Link mới 2
    ]
    
    # Thử tải từ mạng
    for url in urls:
        try:
            df = pd.read_csv(url)
            if not df.empty:
                st.toast(f"✅ Đã tải dữ liệu thành công từ Server!", icon="☁️")
                break
        except:
            continue
            
    # --- CHẾ ĐỘ KHẨN CẤP (OFFLINE MODE) ---
    # Nếu tất cả link trên đều lỗi 404, ta dùng dữ liệu tự tạo để App không bị sập
    if df.empty:
        st.warning("⚠️ Không tải được dữ liệu Big Data. Đang chạy chế độ Demo (Offline)...")
        data_offline = {
            'title': ['Avatar', 'The Avengers', 'Titanic', 'Frozen', 'Iron Man', 'The Dark Knight', 'Interstellar', 'Parasite', 'Spirited Away', 'Your Name'],
            'vote_average': [7.2, 7.4, 7.5, 7.3, 7.4, 8.5, 8.6, 8.5, 8.5, 8.6],
            'release_date': ['2009-12-10', '2012-04-25', '1997-11-18', '2013-11-27', '2008-04-30', '2008-07-16', '2014-11-05', '2019-05-30', '2001-07-20', '2016-08-26'],
            # Giả lập định dạng JSON string giống hệt file thật
            'genres': [
                '[{"name": "Action"}, {"name": "Adventure"}, {"name": "Fantasy"}]',
                '[{"name": "Action"}, {"name": "Sci-Fi"}]',
                '[{"name": "Drama"}, {"name": "Romance"}]',
                '[{"name": "Animation"}, {"name": "Family"}]',
                '[{"name": "Action"}, {"name": "Sci-Fi"}]',
                '[{"name": "Action"}, {"name": "Crime"}, {"name": "Drama"}]',
                '[{"name": "Adventure"}, {"name": "Drama"}, {"name": "Sci-Fi"}]',
                '[{"name": "Comedy"}, {"name": "Thriller"}, {"name": "Drama"}]',
                '[{"name": "Animation"}, {"name": "Family"}, {"name": "Fantasy"}]',
                '[{"name": "Animation"}, {"name": "Romance"}, {"name": "Drama"}]'
            ],
            'keywords': [
                '[{"name": "culture clash"}, {"name": "future"}]',
                '[{"name": "superhero"}, {"name": "marvel"}]',
                '[{"name": "shipwreck"}, {"name": "iceberg"}]',
                '[{"name": "snow"}, {"name": "queen"}]',
                '[{"name": "technology"}, {"name": "billionaire"}]',
                '[{"name": "batman"}, {"name": "joker"}]',
                '[{"name": "space"}, {"name": "black hole"}]',
                '[{"name": "class"}, {"name": "poor family"}]',
                '[{"name": "spirit"}, {"name": "magic"}]',
                '[{"name": "body swap"}, {"name": "time travel"}]'
            ],
            'overview': [
                'In the 22nd century, a paraplegic Marine is dispatched to the moon Pandora.',
                'Earth mightiest heroes must come together to fight an alien invasion.',
                'A seventeen-year-old aristocrat falls in love with a kind but poor artist.',
                'Young princess Anna sets off on a journey to find her estranged sister Elsa.',
                'A billionaire engineer builds a high-tech suit of armor to fight crime.',
                'Batman sets out to dismantle the remaining criminal organizations that plague the city.',
                'A team of explorers travel through a wormhole in space in an attempt to ensure humanity survival.',
                'Greed and class discrimination threaten the newly formed symbiotic relationship between the wealthy Park family and the destitute Kim clan.',
                'During her family move to the suburbs, a sullen 10-year-old girl wanders into a world ruled by gods, witches, and spirits.',
                'Two strangers find themselves linked in a bizarre way.'
            ]
        }
        df = pd.DataFrame(data_offline)

    # --- DATA CLEANING (Xử lý dữ liệu) ---
    def convert(text):
        L = []
        try:
            if isinstance(text, str) and '[' in text:
                for i in ast.literal_eval(text):
                    L.append(i['name'])
        except:
            return [] 
        return L 

    # Kiểm tra cột tồn tại trước khi xử lý
    if 'genres' in df.columns:
        df['genres'] = df['genres'].apply(convert)
    if 'keywords' in df.columns:
        df['keywords'] = df['keywords'].apply(convert)
    
    df['overview'] = df['overview'].fillna('')
    
    def join_features(x):
        return " ".join(x) if isinstance(x, list) else ""

    df['soup'] = df['genres'].apply(join_features) + " " + \
                 df['keywords'].apply(join_features) + " " + \
                 df['overview']
                 
    return df

# Hiển thị spinner
with st.spinner('Đang khởi tạo hệ thống...'):
    df = load_data()

# --- 4. HUẤN LUYỆN MÔ HÌNH ---
@st.cache_resource
def train_model(data):
    if data.empty: return None
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = tfidf.fit_transform(data['soup'])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    return cosine_sim

cosine_sim = train_model(df)
indices = pd.Series(df.index, index=df['title']).drop_duplicates()

def get_recommendations(title, cosine_sim=cosine_sim):
    if cosine_sim is None: return pd.DataFrame()
    try:
        idx = indices[title]
        if isinstance(idx, pd.Series): idx = idx.iloc[0]
    except KeyError:
        return pd.DataFrame()

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11] # Lấy top 10
    movie_indices = [i[0] for i in sim_scores]
    return df.iloc[movie_indices]

# --- 5. GIAO DIỆN NGƯỜI DÙNG ---
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("🔍 Tìm phim")
    search_term = st.text_input("Nhập tên phim (VD: Avatar, Batman...):", "")
    
    # Lọc danh sách
    if search_term:
        filtered_movies = df[df['title'].str.contains(search_term, case=False, na=False)]['title'].values
    else:
        filtered_movies = df['title'].values[:20] # Mặc định hiện 20 phim đầu

    if len(filtered_movies) > 0:
        selected_movie = st.selectbox("Chọn phim:", filtered_movies)
        
        movie_data = df[df['title'] == selected_movie].iloc[0]
        st.metric("Điểm đánh giá", f"{movie_data['vote_average']}/10")
        st.write(f"**Thể loại:** {', '.join(movie_data['genres'])}")
        st.caption(movie_data['overview'])
    else:
        st.warning("Không tìm thấy phim này.")
        selected_movie = None

with col2:
    st.subheader("🎯 Kết quả Gợi ý")
    if selected_movie and st.button("Phân tích & Gợi ý", type="primary"):
        results = get_recommendations(selected_movie)
        if results.empty:
            st.warning("Chưa có đủ dữ liệu để gợi ý cho phim này.")
        else:
            c1, c2 = st.columns(2)
            for i, (index, row) in enumerate(results.iterrows()):
                with (c1 if i % 2 == 0 else c2):
                    with st.container(border=True):
                        st.markdown(f"#### {row['title']}")
                        release = row['release_date'] if pd.notna(row['release_date']) else "N/A"
                        st.markdown(f"*⭐ {row['vote_average']} | 📅 {release}*")
                        st.progress(int(min(row['vote_average'] * 10, 100)))