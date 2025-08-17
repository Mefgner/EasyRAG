import requests
import query

def ask_question(question: str, context: str):
    response = requests.post(
        "http://localhost:8080/v1/chat/completions", 
        json={
            "messages": [
                {
                    "role": "system", 
                    "content": "You are a helpful assistant.\n"
                    "Use only given context to answer the question.\n"
                    "Also tell at the end files you have used.\n"
                    "If context is empty, say: `I don't know`"
                },
                {
                    "role": "user", 
                    "content": f"Question: {question}\nContext: {context}"
                },
            ], 
            "model": "Qwen3-4B-it",
            "temperature": 0.7,
            "max_tokens": 256,
            }
        )
    return response.json()["choices"][0]["message"]["content"]


while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break
    context, _ = query.query(user_input)
    response = ask_question(user_input, str(context.values()))
    print("AI:", response)