import pandas as pd

def prepare_data():
    df = pd.read_csv('data/ruwikibooks_rules.csv')

    expanded_data = []

    for _, row in df.iterrows():
        expanded_data.append({
            'question': row['question'],
            'answer': row['answer']
        })

        if pd.notna(row['alternative_questions']):
            alt_questions = row['alternative_questions'].split(';')
            for alt_q in alt_questions:
                if alt_q.strip():
                    expanded_data.append({
                        'question': alt_q.strip(),
                        'answer': row['answer']
                    })

    faq_df = pd.DataFrame(expanded_data)
    faq_df.to_csv('data/expanded_rules.csv', index=False)

if __name__ == "__main__":
    prepare_data()