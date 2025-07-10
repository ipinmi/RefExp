import re 
import csv
import pandas as pd
# i can see you working lol
# pc came on i was going to the baath room lmao wtf
def clean_text(text):
    """
    Cleans the input expression by removing special characters, extra spaces, and converting to lowercase.
    """
    # Remove special characters except for alphanumeric characters, spaces, and underscores
    text = re.sub(r'[^\w\s]', '', text)
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Convert to lowercase
    text = text.lower()
    
    return text


if __name__ == "__main__":

    # Load the CSV file containing referring expressions
    annotation_df = pd.read_csv('category_annotations.csv')

    expressions = [row['name'] for _, row in annotation_df.iterrows()]

    # Clean the expressions
    cleaned_expressions = [clean_text(expr) for expr in expressions]

    # Append cleaned expressions to the DataFrame
    annotation_df['cleaned_expression'] = cleaned_expressions

    # Reorder columns
    annotation_df = annotation_df[['Unnamed: 0', 'id', 'name', 'cleaned_expression', 'annot_type']]

    # Save the updated DataFrame to a new CSV file
    annotation_df.to_csv('category_annotations_cleaned.csv', index=False)

