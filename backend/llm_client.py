import ollama

OLLAMA_HOST = "http://192.168.1.16:11434"
OLLAMA_MODEL = "gemma4:31b"

client = ollama.Client(host=OLLAMA_HOST)


def generate(prompt: str) -> str:
    response = client.generate(model=OLLAMA_MODEL, prompt=prompt)
    return response["response"]


def chat(messages: list) -> str:
    """messages: [{"role": "user"/"assistant", "content": "..."}]"""
    response = client.chat(model=OLLAMA_MODEL, messages=messages)
    return response["message"]["content"]


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    print(generate("안녕하세요! 연결 테스트입니다."))
