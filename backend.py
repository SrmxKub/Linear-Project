import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


file_path = 'https://raw.githubusercontent.com/SrmxKub/Linear-Project/refs/heads/main/perfumes_data.csv'
df = pd.read_csv(file_path)
df['num_seller_ratings'] = df['num_seller_ratings'].apply(lambda x: float(x.replace('K', '')) * 1000 if 'K' in x else x).astype(int)

def cosine_similarity_cal(feature, filtered_data, input_value):

    input_df = pd.DataFrame({feature: [input_value]})
    combined_df = pd.concat([filtered_data[[feature]], input_df])
    tfidf_matrix = TfidfVectorizer().fit_transform(combined_df[feature])

    return cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()


# scents = "Woody"
# base_notes = "Oakmoss, Patchouli, Vetiver"
# middle_notes = "Jasmine, Hazelnut, Cashmirwood, Honey"
# department = "Men"  # Filter


def most_similar_perfumes(scents, base_notes, middle_notes, department, top_n = 10):
    
    filtered_data = df[df['department'] == department]

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
    return top_similar_perfumes[['name', 'brand', 'department', 'cosine_similarity', 'cosine_scent', 'cosine_base_note', 'cosine_middle_note']]

def get_heatmap():
    departments = ['Men', 'Women', 'Unisex', None]
    
    for department in departments:
    
        filtered_data = df[df['department'] == department]
        if filtered_data.empty:
            filtered_data = df
        
        correlation_matrix = filtered_data[['old_price', 'new_price', 'ml', 'item_rating', 'seller_rating', 'num_seller_ratings']].corr()

        plt.figure(figsize = (14, 8))
        sns.heatmap(correlation_matrix, annot = True, cmap = 'coolwarm', fmt = ".2f", annot_kws = {"size": 14})
        plt.tight_layout(rect = [0, 0, 1, 1], pad = 5)
        plt.xticks(fontsize = 11)
        plt.yticks(fontsize = 11)
        
        if department != None:
            plt.title(f'Corretions Martrix of {department} Perfumes', fontsize=20)
            plt.savefig(f'img/{department}_heatmap.png')
        else:
            plt.title(f'Corretions Martrix of All Perfumes', fontsize=20)
            plt.savefig(f'img/All_heatmap.png')


if __name__ == "__main__":
    get_heatmap()