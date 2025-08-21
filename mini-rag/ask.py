import requests
import query
import json

def ask_question(question: str, context: str):
    url = "http://localhost:8080/v1/chat/completions"
    payload = {
            "messages": [
                {
                    "role": "system", 
                    "content": "<system>\n"
                    "You are a helpful assistant created by Mefgner inside of the project: `Mini-RAG`\n"
                    "Use wisely given context to answer the question.\n"
                    "Always end with: `Files used: <comma-separated filenames or None>`.\n"
                    "If context is empty, say: `Sorry. I don't understand the question.`\n"
                    "</system>\n"
                },
                {
                    "role": "user", 
                    "content": 
                        f"<question>{question}</question>\n"
                        f"<context>{context}</context>"
                },
            ], 
            "model": "Qwen3-4B-it",
            "temperature": 0.7,
            "max_tokens": 512,
            "stream": True
            }
    stream = requests.post(url, json= payload, stream=True)
    stream.encoding = 'utf-8'
    return stream


def render_stream(stream: requests.Response):
    with stream:
        # buffer = ""
        for line in stream.iter_lines(decode_unicode=True):
            # print(line)
            if not line or not line.startswith("data:"):
                continue
            
            data = line[len("data:"):].strip()
            
            if data == "[DONE]":
                break
            
            chunk = json.loads(data)
            
            # try:
            #     chunk = json.loads(data)
            # except json.JSONDecodeError:
            #     buffer += data
            #     try:
            #         chunk = json.loads(buffer)
            #         buffer = ""
            #     except json.JSONDecodeError:
            #         continue

            delta = chunk["choices"][0]["delta"].get("content")
            if delta:
                print(delta, end="", flush=True)
    print()

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break
    context, _ = query.query(user_input)
    stream = ask_question(user_input, str(context.values()))
    render_stream(stream)