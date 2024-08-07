from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain.vectorstores.utils import filter_complex_metadata
from langchain.schema.runnable import Runnable

from ai71 import AI71

class ChatPDF:
    vector_store = None
    retriever = None
    chain = None
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100)
        self.messages=[{"role": "system", "content": "You are a helpful server that helps users order cuisines."}]
        self.prompt = "Context: {context}\n Question: {question}"
        AI71_API_KEY = "api71-api-0340d9f7-ce0d-4155-aa10-c42e41b6ada2"
        self.client = AI71(AI71_API_KEY)

    def ingest(self, pdf_file_path: str):
        docs = PyPDFLoader(file_path=pdf_file_path).load()
        print("length of docs: "+str(len(docs)))
        for doc in docs:
            print(doc.page_content)
            print("===============================")
        chunks = self.text_splitter.split_documents(docs)
        chunks = filter_complex_metadata(chunks)
        print("length of chunks: "+str(len(chunks)))
        vector_store = Chroma.from_documents(documents=chunks, embedding=FastEmbedEmbeddings())
        self.retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 3,
                "score_threshold": 0.5,
            },
        )

    def ask(self, query: str):
        if not self.retriever:
            return "Please, add a PDF document first."
        
        content=""
        retrieved_docs = self.retriever.get_relevant_documents(query)
        
        #print("Retrieved Documents:")
        context=""
        for doc in retrieved_docs:
            #print(f"Document ID: {doc.metadata.get('doc_id', 'Unknown')}")
            #print(doc.page_content)
            #print("\n---\n")
            context+=doc.page_content
        #print(context)
        self.messages.append({"role": "user", "content":  self.prompt.format(question=query,context=context)})
        
        for chunk in self.client.chat.completions.create(
            messages=self.messages,
            model="tiiuae/falcon-180B-chat",
            stream=True,
        ):
            delta_content = chunk.choices[0].delta.content
            if delta_content:
                #print(delta_content, sep="", end="", flush=True)
                content += delta_content
        print(self.messages)
        print("\n\n")
        self.messages.pop()
        self.messages.append({"role": "user", "content": query})
        self.messages.append({"role": "assistant", "content": content})
        
        
        self.messages.append({"role": "system", "content":  """Has the user explicitly said that he has finished ordering? 
        If yes, what has the customer ordered?

        """})
        
        status=""
        for chunk in self.client.chat.completions.create(
            messages=self.messages,
            model="tiiuae/falcon-180B-chat",
            stream=True,
        ):
            delta_content = chunk.choices[0].delta.content
            if delta_content:
                #print(delta_content, sep="", end="", flush=True)
                status += delta_content
        print(status)
        self.messages.pop()
        
        
        return content

    def clear(self):
        self.vector_store = None
        self.retriever = None
        self.chain = None
        
