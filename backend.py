import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


file_path = 'https://raw.githubusercontent.com/SrmxKub/Linear-Project/refs/heads/main/perfumes_data_final.csv'
df = pd.read_csv(file_path)
df['num_seller_ratings'] = df['num_seller_ratings'].apply(
    lambda x: float(x.replace('K', '')) * 1000 if isinstance(x, str) and 'K' in x else float(x)
)

df['num_seller_ratings'] = df['num_seller_ratings'].fillna(0).astype(int)

# Cosine Similarity Calculation
def cosine_similarity_cal(feature, filtered_data, input_value):

    input_df = pd.DataFrame({feature: [input_value]})
    combined_df = pd.concat([filtered_data[[feature]], input_df])
    tfidf_matrix = TfidfVectorizer().fit_transform(combined_df[feature])
    return cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()

# Find most similar perfume
def most_similar_perfumes(scents, base_notes, middle_notes, typ, min_price, max_price, top_n = 10):
    
    filtered_data = df[(min_price <= df['price_THB']) & (df['price_THB'] <= max_price)]

    if typ != 'All':
        filtered_data = filtered_data[filtered_data['department'] == typ] 

    if filtered_data.empty:
        return None

    scent_sim = cosine_similarity_cal('scents', filtered_data, scents)
    base_note_sim = cosine_similarity_cal('base_note', filtered_data, base_notes)
    middle_note_sim = cosine_similarity_cal('middle_note', filtered_data, middle_notes)
    
    similarity = (scent_sim + base_note_sim + middle_note_sim) / 3

    data = filtered_data.copy()
    data['cosine_scent'] = scent_sim
    data['cosine_base_note'] = base_note_sim
    data['cosine_middle_note'] = middle_note_sim
    data['cosine_similarity'] = similarity

    top_similar_perfumes = data.sort_values(by = 'cosine_similarity', ascending = False).head(top_n)
    top_similar_perfumes['price_THB'] = top_similar_perfumes['price_THB'].astype(int)

    return top_similar_perfumes[['name', 'brand', 'department', 'price_THB', 'ml', 'item_rating', 'cosine_similarity', 'img_url']]


# Create heatmap with perfume type
def get_heatmap():
    departments = ['Men', 'Women', 'Unisex', 'All']
    
    for department in departments:
    
        filtered_data = df[df['department'] == department]
        if filtered_data.empty:
            filtered_data = df
        
        correlation_matrix = filtered_data[['old_price', 'new_price', 'ml', 'item_rating', 'seller_rating', 'num_seller_ratings']].corr()

        plt.figure(figsize = (16, 7),)
        sns.heatmap(correlation_matrix, annot = True, cmap = 'coolwarm', fmt = ".2f", annot_kws = {"size": 14})
        plt.xticks(fontsize = 12)
        plt.yticks(fontsize = 12)
        plt.title(f'Correlations Martrix of {department} Perfumes', fontsize=20, fontweight='bold')
        plt.subplots_adjust(left=0.21)
        plt.savefig(f'img/{department}_heatmap.png')


if __name__ == "__main__":
    get_heatmap()