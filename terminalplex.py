import os
import requests
from openai import OpenAI
from dotenv import load_dotenv
from googlesearch import search
from bs4 import BeautifulSoup
from rich import print
import asyncio

system_prompt_search = """You are a helpful assistant whose primary goal is to reformulate user query for Google search to get valuable knowledge for further answer."""
search_prompt = """
You should use Google search for most queries to find the most accurate and updated information. Follow these conditions:
- Reformulated user query for Google search.
- User query may sometimes refer to previous messages. Make sure your Google search considers the entire message history.
- Return only the query as answer.

User Query:
{query}
"""

system_prompt_cited_answer = """You are a helpful assistant who is expert at answering user's queries based on the cited context (search results with /citation number/ /website link/ and brief descriptions)"""
cited_answer_prompt = """
Provide a relevant, informative response to the user's query using the given context.

- Answer directly without referring the user to any external links.
- Use an unbiased, journalistic tone and avoid repeating text.
- Cite all information using [citation number of the website link] notation without the word citation and website link, matching each part of your answer to its source from the context block given below.

Context Block:
{context_block}

User Query:
{query}
"""
# Prompt Credits - https://github.com/Yusuke710

user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36"

async def search_google(query):
    loop = asyncio.get_event_loop()
    search_results = await loop.run_in_executor(None, lambda: list(search(query, num_results=3)))
    search_results = [result for result in search_results if result is not None and result != '']
    print("*"*40)
    print("Sources : \n")
    for num, result in enumerate(search_results):
        print(f"{num+1}. {result}")
    print("*"*40)
    return search_results
    
async def extract_text_from_url(url):
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(None, lambda: requests.get(url, headers={"User-Agent": user_agent}))
    soup = await loop.run_in_executor(None, lambda: BeautifulSoup(response.text, "html.parser"))
    text = soup.get_text()
    return text[:10000]

async def get_all_text_from_urls(query):
    text = ""
    urls = await search_google(query)
    for num, url in enumerate(urls):
        text += "\n".join(f"/{num+1}/. /{url}/ /{await extract_text_from_url(url)}/")
    return text

def get_search_query(query, prev_messages, client):
    query = search_prompt.format(query=query)
    prev_messages = prev_messages or []
    new_messages = prev_messages + [{"role": "user", "content": query}]
    response = client.chat.completions.create(
        model=os.getenv("MODEL_NAME"),
        messages=[{"role": "system", "content": system_prompt_search}, *new_messages],
        max_tokens=50,
    )
    return response.choices[0].message.content

def get_cited_answer(query, context_block, prev_messages, client):
    query = cited_answer_prompt.format(query=query, context_block=context_block)
    prev_messages = prev_messages or []
    new_messages = prev_messages + [{"role": "user", "content": query}]
    full_answer = ""
    response = client.chat.completions.create(
        model=os.getenv("MODEL_NAME"),
        messages=[{"role": "system", "content": system_prompt_cited_answer}, *new_messages],
        max_tokens=os.getenv("MAX_TOKENS"),
        stream=True
    )

    #Streamnig the response
    print("-"*40)
    print("Answer: \n")
    for chunk in response:
        full_answer += chunk.choices[0].delta.content
        print(chunk.choices[0].delta.content, end="")
    print("\n")
    print("-"*40)

    new_prev_messages = prev_messages + [{"role": "assistant", "content": full_answer}]
    return new_prev_messages
    

async def main():
    load_dotenv()
    client = OpenAI(
            api_key=os.getenv("API_KEY"),
            base_url=os.getenv("BASE_URL")
        )
    
    prev_messages = None

    while True:
        prompt = input("Enter your prompt: ")
        print("\n")
        if prompt.lower() == "exit" or prompt.lower() == "quit":
            break
        search_query = get_search_query(prompt, prev_messages, client)

        context_block = await get_all_text_from_urls(search_query)
        prev_messages = get_cited_answer(prompt, context_block, prev_messages, client)
        print("\n")



if __name__ == "__main__":
    asyncio.run(main())