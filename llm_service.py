import os

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

ELEVENLABS_API_KEY = '431f452112cab175b80762e50e525c8f'
VOICE_ID = '21m00Tcm4TlvDq8ikWAM'

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)


def chat_completion(query: str):
    """Retrieve text from OpenAI and pass it to the text-to-speech function."""
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": query},
        ],
        model="mixtral-8x7b-32768",
        stream=True
    )

    for chunk in response:
        delta = chunk.choices[0].delta.content
        print(delta, end="")
        # yield delta
