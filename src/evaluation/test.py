GOOGLE_GENAI_USE_VERTEXAI=True
GOOGLE_CLOUD_LOCATION="global"

import os
import google.generativeai as genai
# from google import genai

from dotenv import load_dotenv

from google import genai as vortex_genai
from google.genai import types
import base64


load_dotenv()

def generate(prompt, model="gemini-2.5-pro", temperature=0.7):
    client = vortex_genai.Client(
        vertexai=True,
        project="gen-lang-client-0685512248",
        location="global",
    )

    contents = [
    types.Content(
        role="user",
        parts=[
            types.Part.from_text(text=prompt),
        ]
    )
    ]

    generate_content_config = types.GenerateContentConfig(
    temperature = temperature,
    top_p = 0.95,
    seed = 0,
    max_output_tokens = 65535,
    safety_settings = [types.SafetySetting(
        category="HARM_CATEGORY_HATE_SPEECH",
        threshold="OFF"
    ),types.SafetySetting(
        category="HARM_CATEGORY_DANGEROUS_CONTENT",
        threshold="OFF"
    ),types.SafetySetting(
        category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
        threshold="OFF"
    ),types.SafetySetting(
        category="HARM_CATEGORY_HARASSMENT",
        threshold="OFF"
    )],
    thinking_config=types.ThinkingConfig(
        thinking_budget=-1,
    ),
    )
    raw_response = ""
    for chunk in client.models.generate_content_stream(
    model = model,
    contents = contents,
    config = generate_content_config,
    ):
        raw_response += chunk.text

    return raw_response

# Initialize OpenAI client
api_key = os.getenv("GEMINI_API_KEY")

temperature = 0.2
model = "gemini-2.5-pro"
prompt = "Hi, how are you?"
client = None

if client is None:
    genai.configure(api_key=api_key)
    model_obj = genai.GenerativeModel(model)
    response = model_obj.generate_content(prompt, generation_config={"temperature": temperature})
    raw_response = response.text
else:
    response = generate(
        model=model,
        prompt=prompt,
        temperature=temperature
    )
    raw_response = response

print(raw_response)

# %%







                    