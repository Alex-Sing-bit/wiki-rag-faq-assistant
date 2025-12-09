import os

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class Embedder:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def embed(self, questions_list):
        question_embeddings = self.model.encode(
            questions_list,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        return question_embeddings

    def embed_one(self, question):
        return self.embed([question])

    def find_similar(self, user_question, question_embeddings):
        similarities = self.model.similarity(question_embeddings, user_question)
        similarities = similarities.tolist()
        return [n[0] for n in similarities]

    def find_similar_cosine_similarity(self, user_question, question_embeddings):
        return cosine_similarity(user_question, question_embeddings)[0]


def load_data(filepath='expanded_rules.csv'):
    df = pd.read_csv(filepath)
    print("Данные загружены:")
    print(df.head())
    print(df.tail())
    return df


def load_embeddings(filepath='question_embeddings.npy'):
    return np.load(filepath)


def find_top_similar_questions(user_question, df, question_embeddings, embedder, builtin_method=True, top_n=3):
    user_embedding = embedder.embed_one(user_question)

    if builtin_method:
        similarities = embedder.find_similar(user_embedding, question_embeddings)
    else:
        similarities = embedder.find_similar_cosine_similarity(user_embedding, question_embeddings)

    similarity_df = pd.DataFrame({
        'question': df['question'],
        'similarity': similarities,
        'answer': df['answer']
    })

    return similarity_df.groupby('answer').first().reset_index().sort_values('similarity', ascending=False).head(top_n)


def display_results(user_question, top_results, df):
    print(f"\nВопрос пользователя: {user_question}")

    for idx, row in top_results.iterrows():
        print(f"\n- {row['question']} (схожесть: {row['similarity']:.4f})")
        answer = df[df['question'] == row['question']]['answer'].values[0]
        print(f"  Ответ: {answer}")


def get_results(user_question = "Как удалить учебник новичка?"):
    df = load_data('expanded_rules.csv')
    questions = np.array(df['question'])

    embedder = Embedder()

    path = 'question_embeddings.npy'
    if os.path.exists(path):
        question_embeddings = load_embeddings(path)
    else:
        question_embeddings = embedder.embed(questions)
        np.save('question_embeddings.npy', question_embeddings)

    top_results = find_top_similar_questions(
        user_question,
        df,
        question_embeddings,
        embedder
    )

    display_results(user_question, top_results, df)

    return top_results


if __name__ == "__main__":
    get_results('Как удалить учебник новичка?')