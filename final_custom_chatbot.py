import requests
import re
import json
import concurrent.futures
from typing import Callable, Iterable
from langchain.docstore.document import Document
from pathlib import Path
from bs4 import BeautifulSoup
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from openai import OpenAI


class Assistant:

    def __init__(self):
        self.OPENAI_API_KEY = "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"
        self.client = OpenAI(api_key=self.OPENAI_API_KEY)
        self.vector_store = self.load_data()

    def run_parallel_exec(self, exec_func: Callable, iterable: Iterable, *func_args, **kwargs):
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=kwargs.pop("max_workers", 100)
        ) as executor:
            # Start the load operations and mark each future with each element
            future_element_map = {
                executor.submit(exec_func, element, *func_args): element
                for element in iterable
            }
            result: list[tuple] = []
            for future in concurrent.futures.as_completed(future_element_map):
                element = future_element_map[future]
                try:
                    data = future.result()
                except Exception as exc:
                    print(
                        f"Got error while running parallel_exec: {element}: \n{exc}")
                    result.append((element, exc))
                else:
                    result.append((element, data))
            try:
                return dict(result)
            except Exception as e:
                return result

    def get_site_text(self, url):
        with requests.get(url) as response:
            return response.text

    def parse_site_text(self, site_text):
        soup = BeautifulSoup(site_text, 'html.parser')
        all_text = soup.get_text(separator='\n', strip=True)
        return all_text

    def parse_url_text(self, url):
        site_text = self.get_site_text(url)
        print("Parsing {}".format(url))
        return self.parse_site_text(site_text)

    def load_data(self):
        with open("sitemap.xml", "r") as file:
            xml_data = file.read()

        urls = re.findall(r'<loc>(.*?)</loc>', xml_data)

        if Path("url_texts.json").exists():
            with open("url_texts.json", "r") as file:
                url_texts = json.load(file)
        else:
            url_texts = ""

        if not url_texts:
            url_texts = self.run_parallel_exec(self.parse_url_text, urls)

        with open("url_texts.json", "w") as file:
            json.dump(url_texts, file)

        documents = [Document(page_content=text, metadata={
            "source": url}) for url, text in url_texts.items()]

        text_splitter = CharacterTextSplitter(separator='\n',
                                              chunk_size=1024,
                                              chunk_overlap=50)

        docs = text_splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        vectorStore_openAI = FAISS.from_documents(docs, embeddings)

        return vectorStore_openAI

    def main(self, user_question: str):
        query = user_question.strip().lower()
        # query = input("Question: ").strip().lower()
        if query in ['exit']:
            print("Exiting chat. Goodbye!")
            return "Bye"

        prompt = """You are a helpful assistant for the users of FiftyFive Technologies Ltd. You will be given a context and a question, 
        and you will answer the question based on the context by formulating a brief and relevant answer. If the user sends "Hello," 
        don't go through the context; only respond with "Hi there.""

        Here is the context:
        {context}

        Question: {question}
        Response Length: Please ensure your response is within 100-150 words.
        """

        docs = self.vector_store.similarity_search(query, k=3)
        context = "\n".join([doc.page_content for doc in docs])

        output = self.client.chat.completions.create(model="gpt-3.5-turbo", temperature=0, messages=[
            {"role": "user", "content": prompt.format(context=context, question=query)}])

        sources = ', '.join(x.metadata.get("source") for x in docs)
        print(output.choices[0].message.content +
              f"\n\n Here are the sources: {sources}" + "\n")

        # output_statement = output.choices[0].message.content + \
        #     f"\n\n Here are the sources: {sources}" + "\n"

        output_statement = output.choices[0].message.content + "\n"

        return output_statement


if __name__ == "__main__":
    assistant = Assistant()
    assistant.main()
