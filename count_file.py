import pandas as pd

file_path = 'https://raw.githubusercontent.com/SrmxKub/Linear-Project/refs/heads/main/perfumes_data_final.csv'
perfume_data = pd.read_csv(file_path)


# Function to split and count unique values across rows for a given column
def count_unique_values(column_data):

    split_values = column_data.dropna().str.split(',').apply(lambda x: [i.strip() for i in x])
    all_values = [item for sublist in split_values for item in sublist]

    return pd.Series(all_values).value_counts()

# Count occurrences for 'scents', 'base_note', and 'middle_note'
scents_count = count_unique_values(perfume_data['scents'])
base_note_count = count_unique_values(perfume_data['base_note'])
middle_note_count = count_unique_values(perfume_data['middle_note'])

# Display the results for review
scents_count.head(), base_note_count.head(), middle_note_count.head()
output_file_path = 'perfume_scent_base_middle_counts.xlsx'

with pd.ExcelWriter(output_file_path) as writer:
    scents_count.to_frame('Count').to_excel(writer, sheet_name='Scents')
    base_note_count.to_frame('Count').to_excel(writer, sheet_name='Base Notes')
    middle_note_count.to_frame('Count').to_excel(writer, sheet_name='Middle Notes')

import os
# This will open the file after saving (works on Windows/Mac/Linux depending on system)

# os.startfile(output_file_path)  # For Windows
os.system(f"open {output_file_path}")  # For macOS
# os.system(f"xdg-open {output_file_path}")  # For Linux