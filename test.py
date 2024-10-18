import os
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import Pinecone
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
import pinecone

class Document:
    def __init__(self, page_content: str, metadata: dict = None):
        self.page_content = page_content
        self.metadata = metadata if metadata is not None else {}

class SimpleTextLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self):
        with open(self.file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        return [Document(page_content=text, metadata={})]

class ChatBot:
    def __init__(self):
        load_dotenv()
        self.loader = SimpleTextLoader('./horoscope.txt') 
        self.documents = self.loader.load()

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

     
        pinecone.init(api_key=os.getenv('PINECONE_API_KEY'), environment='gcp-starter')  
        index_name = "langchain-demo"

       
        if index_name not in pinecone.list_indexes():
            pinecone.create_index(name=index_name, metric="cosine", dimension=768)
        
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=4)
        docs = text_splitter.split_documents(self.documents)
        
        self.docsearch = Pinecone.from_documents(docs, embeddings, index_name=index_name)

        self.vectorstore = self.docsearch

      
        repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1" 
        self.llm = HuggingFaceEndpoint(
            repo_id=repo_id,
            temperature=0.7,  
            top_k=30,
            max_new_tokens=128,  
            huggingfacehub_api_token=os.getenv('HUGGINGFACE_API_KEY'),
        )

        template = """
        You are an astrologer and fortune teller. These humans will ask you questions about their future, relationships, career, and other aspects of their life. 
        Use the following horoscope or astrological context to answer the question. If you don't know the answer, just say you don't know. 
        Keep the answer mystical but concise, within 6 sentences.

        Context: {context}
        Question: {question}
        Answer: 
        """

        self.prompt = PromptTemplate(
            template=template, 
            input_variables=["context", "question"]
        )

        self.rag_chain = (
            {"context": self.vectorstore.as_retriever(), "question": RunnablePassthrough()}
            | self.prompt 
            | self.llm
            | StrOutputParser()
        )

    def ask(self, user_input):
            try:
                result = self.rag_chain.invoke(user_input) 
                return result
            except Exception as e:
                print(f"Error during invocation: {e}")
                return "Sorry, I couldn't process your request."
if __name__ == "__main__":
    bot = ChatBot()
    user_input = input("Ask me anything: ")
    result = bot.ask(user_input)
    print(result)