import requests
import json
import pandas as pd
from typing import Dict, Any
import os

import embeding


def basic_model_settings(context, user_question):
    system_prompt = """Ты — помощник-эксперт по правилам Викиучебника. 
        Твоя задача — на основе предоставленного контекста из официальной базы знаний дать точный, структурированный и полезный ответ.

        ИНСТРУКЦИИ:
        1. Используй ТОЛЬКО информацию из предоставленного контекста.
        2. Не выдумывай и не добавляй факты, которых нет в контексте.
        3. Если в контексте нет полного ответа на вопрос, честно скажи: "В предоставленных материалах нет полной информации по этому вопросу."
        4. Структурируй ответ: краткий вывод, затем детали по пунктам.
        5. Сохраняй официальный, но понятный стиль."""

    user_prompt = f"""КОНТЕКСТ ИЗ БАЗЫ ЗНАНИЙ ВИКИУЧЕБНИКА: {context}

        ВОПРОС ПОЛЬЗОВАТЕЛЯ: {user_question}

        Сформируй ответ, строго следуя инструкциям выше. Отвечай на русском языке."""

    payload = {
        "model": "google/gemma-3-27b-it:free",
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        "temperature": 0.3,
        "max_tokens": 700,
        "top_p": 0.9,
        "frequency_penalty": 0.1,
        "presence_penalty": 0.1
    }

    return payload

def creative_model_settings(context, user_question):
    system_prompt = """Ты — помощник-эксперт по правилам Викиучебника. 
        Твоя задача — на основе предоставленного контекста из официальной базы знаний дать точный, структурированный и полезный ответ.

        ИНСТРУКЦИИ:
        1. Используй ТОЛЬКО информацию из предоставленного контекста, если контекст дает ответ на предоставленный вопрос.
        2. Если в контексте нет полного ответа на вопрос, честно скажи: "В предоставленных материалах нет полной информации по этому вопросу."
        4. Если в контексте нет никакой подходящей для ответа информации, но вопрос связан с Викиучебником, дай пользователю общий совет для такого случая. 
        Обязательно укажи, что этот ответ создан на основе твоих знаний, а не заданных правил Викиучебника и может быть использован только как направление для дальнейших действий, а не официальная рекомендация.
        5. Структурируй ответ: краткий вывод, затем детали по пунктам.
        6. Сохраняй официальный, но понятный стиль."""

    user_prompt = f"""КОНТЕКСТ ИЗ БАЗЫ ЗНАНИЙ ВИКИУЧЕБНИКА: {context}

        ВОПРОС ПОЛЬЗОВАТЕЛЯ: {user_question}

        Сформируй ответ, строго следуя инструкциям выше. Отвечай на русском языке."""

    payload = {
        "model": "google/gemma-3-27b-it:free",
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        "temperature": 0.8,
        "max_tokens": 900,
        "top_p": 0.7,
        "frequency_penalty": 0.1,
        "presence_penalty": 0.1
    }

    return payload

def enhance_answer_with_gemma(user_question: str, search_results: pd.DataFrame, creative_answer=False) -> str:

    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

    if search_results.empty or len(search_results) == 0:
        return "К сожалению, в базе знаний нет информации для ответа на ваш вопрос."

    context_parts = []
    for i, (_, row) in enumerate(search_results.head(3).iterrows()):
        context_parts.append(
            f"[Источник {i + 1}]:\n"
            f"Вопрос: {row['question']}\n"
            f"Полный ответ: {row['answer'][:400]}..."
        )

    context = "\n\n".join(context_parts)

    if creative_answer:
        payload = creative_model_settings(context, user_question)
    else:
        payload = basic_model_settings(context, user_question)

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            data=json.dumps(payload),
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            llm_answer = result['choices'][0]['message']['content'].strip()

            usage = result.get('usage', {})
            print(f"[DEBUG] Использовано токенов: {usage.get('total_tokens', 'N/A')}")

            return llm_answer
        else:
            error_msg = f"Ошибка API: {response.status_code}"
            try:
                error_detail = response.json()
                error_msg += f" - {error_detail.get('error', {}).get('message', '')}"
            except:
                error_msg += f" - {response.text[:100]}"

            print(f"[ERROR] {error_msg}")
            return generate_fallback_answer(search_results)

    except requests.exceptions.Timeout:
        print("[ERROR] Таймаут запроса к OpenRouter API")
        return generate_fallback_answer(search_results)
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Ошибка соединения: {e}")
        return generate_fallback_answer(search_results)


def generate_fallback_answer(search_results: pd.DataFrame) -> str:
    if not search_results.empty:
        best_match = search_results.iloc[0]

        fallback_response = (
            f"**На основе наиболее релевантного вопроса:**\n\n"
            f"**Вопрос:** {best_match['question']}\n\n"
            f"**Подробнее:** {best_match['answer'][:500]}..."
        )
        return fallback_response

    return "Не удалось получить ответ от системы. Пожалуйста, попробуйте переформулировать вопрос."


def generate_rag_response(user_question: str, semantic_results: pd.DataFrame, use_llm: bool = True, creative_answer=False) -> Dict[str, Any]:
    if len(semantic_results) == 0:
        return {
            'status': 'no_results',
            'message': 'Извините, не нашёл подходящего ответа в базе знаний.',
            'suggestions': ['Попробуйте переформулировать вопрос', 'Упростите запрос']
        }

    best_result = semantic_results.iloc[0]
    simple_response = {
        'status': 'success',
        'confidence': best_result['similarity'],
        'full_answer': best_result['answer'][:500] + '...' if len(best_result['answer']) > 500 else best_result[
            'answer']
    }

    if use_llm:
        try:
            llm_enhanced = enhance_answer_with_gemma(user_question, semantic_results, creative_answer=creative_answer)
            simple_response['llm_enhanced'] = llm_enhanced

            simple_response['context_used'] = len(semantic_results.head(3))

        except Exception as e:
            print(f"[ERROR] Не удалось улучшить ответ через LLM: {e}")
            simple_response['llm_enhanced'] = generate_fallback_answer(semantic_results)
            simple_response['llm_failed'] = True

    return simple_response


# Пример вызова
if __name__ == "__main__":
    print("Тестирование LLM-улучшения ответов...")

    test_question = "Как пригласить участника в Викиучебник?"
    test_results = embeding.get_results(test_question)

    response = generate_rag_response(test_question, test_results, creative_answer=True, use_llm=True)

    print("\nВопрос пользователя:", test_question, '\n')
    print(response['full_answer'])

    if 'llm_enhanced' in response:
        print("\nУлучшенный совет (Gemma 3):")
        print(response['llm_enhanced'])