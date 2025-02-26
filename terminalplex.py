import os
import requests
from openai import OpenAI
from dotenv import load_dotenv
from googlesearch import search
from bs4 import BeautifulSoup
from rich import print

system_prompt_cited_answer = """You are a helpful assistant who is expert at answering user's queries based on the cited context (search results with /citation number/ /website link/ and brief descriptions)"""
cited_answer_prompt = """
Provide a relevant, informative response to the user's query using the given context.

- Answer directly without referring the user to any external links.
- Use an unbiased, journalistic tone and avoid repeating text.
- Cite all information using [citation number of the website link] notation, matching each part of your answer to its source from the context block given below.

Context Block:
{context_block}

User Query:
{query}
"""
# Prompt Credits - https://github.com/Yusuke710

user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36"

def search_google(query):
    search_results = search(query, num_results=3)
    search_results = [result for result in search_results if result is not None and result != '']
    print("*"*40)
    print("Sources : \n")
    for num, result in enumerate(search_results):
        print(f"{num+1}. {result}")
    print("*"*40)
    return search_results
    
def extract_text_from_url(url):
    response = requests.get(url, headers={"User-Agent": user_agent})
    soup = BeautifulSoup(response.text, "html.parser")
    text = soup.get_text()
    return text

def get_all_text_from_urls(query):
    text = ""
    urls = search_google(query)
    for num, url in enumerate(urls):
        text += "\n".join(f"/{num+1}/. /{url}/ /{extract_text_from_url(url)}/")
    return text

def main():
    load_dotenv()

    client = OpenAI(
        api_key=os.getenv("API_KEY"),
        base_url=os.getenv("BASE_URL")
    )

    prompt = "What is Perplexity AI?"

    context_block = get_all_text_from_urls(prompt)
    prompt = cited_answer_prompt.format(context_block=context_block, query=prompt)

    response = client.chat.completions.create(
        model=os.getenv("MODEL_NAME"),
        messages=[
            {"role": "system", "content": system_prompt_cited_answer},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1000,
        stream=True
    )

    #Streamnig the response
    print("-"*40)
    for chunk in response:
        print(chunk.choices[0].delta.content, end="")
    print("\n")
    print("-"*40)


if __name__ == "__main__":
    main()