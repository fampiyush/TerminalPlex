import os
import requests
from openai import OpenAI
from dotenv import load_dotenv
from ddgs import DDGS
from bs4 import BeautifulSoup
from rich import print
import asyncio

ddgs = DDGS()

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
- Cite all information using [citation number] notation of the website link (dont't write the word citation only the number like [1]), matching each part of your answer to its source from the context block given below.

Context Block:
{context_block}

User Query:
{query}
"""
# Prompt Credits - https://github.com/Yusuke710

user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36"

async def search_ddg(query):
    loop = asyncio.get_event_loop()
    try:
        search_results = await loop.run_in_executor(None, lambda: list(ddgs.text(query, max_results=4)))
        urls = [result['href'] for result in search_results if result and 'href' in result]
        return urls
    except Exception as e:
        print(f"Error during DuckDuckGo search: {e}")
        return []

def print_sources(urls):
    print("*"*40)
    print("Sources : \n")
    for num, url in enumerate(urls):
        print(f"{num+1}. {url}")
    print("*"*40)
    
async def extract_text_from_url(url):
    loop = asyncio.get_event_loop()
    try:
        response = await loop.run_in_executor(None, lambda: requests.get(url, headers={"User-Agent": user_agent}))
        response.raise_for_status()
    except requests.RequestException as e:
        return None
    soup = await loop.run_in_executor(None, lambda: BeautifulSoup(response.text, "html.parser"))
    text = soup.get_text()
    return text[:10000]

async def get_all_text_from_urls(query):
    text = ""
    urls = await search_ddg(query)
    urls_c = urls.copy()
    num = 1
    for url in urls:
        fetched_text = await extract_text_from_url(url)
        if fetched_text is not None:
            text += "\n".join(f"/{num}/. /{url}/ /{fetched_text}/")
            num += 1
        else:
            urls_c.remove(url)
    print_sources(urls_c)
            
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

    print(f"Search Query: {response.choices[0].message.content}\n")
    return response.choices[0].message.content

def get_cited_answer(query, context_block, prev_messages, client):
    query = cited_answer_prompt.format(query=query, context_block=context_block)
    prev_messages = prev_messages or []
    new_messages = prev_messages + [{"role": "user", "content": query}]
    full_answer = ""
    response = client.chat.completions.create(
        model=os.getenv("MODEL_NAME"),
        messages=[{"role": "system", "content": system_prompt_cited_answer}, *new_messages],
        max_tokens=int(os.getenv("MAX_TOKENS")),
        stream=True
    )

    #Streamnig the response
    print("-"*40)
    print("Answer: \n")
    for chunk in response:
        full_answer += chunk.choices[0].delta.content if chunk.choices[0].delta.content else ""
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