import getpass
import os
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.vectorstores import Pinecone as PC
from langchain_openai import OpenAIEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from openai import OpenAI
import openai
import pinecone
from pinecone import Pinecone, ServerlessSpec
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import OpenAI

load_dotenv()


OPENAI_API_KEY= '<--ENTER YOUR OPENAI KEY HERE-->'
pc= Pinecone(api_key='<--ENTER YOUR PINECONE API KEY HERE',
                          environment=os.getenv("<-ENTER YOUR PINE CONE ENVIRONMENT NAME HERE->")
                          )
index = pc.Index("<-NAME OF INDEX AGAIN->")                     


loader = TextLoader("story.txt")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, 
                                      chunk_overlap=200, 
                                      length_function=len)
docs = text_splitter.split_documents(documents)
 


print (f'Now you have {len(docs)} documents')


embeddings_model = OpenAIEmbeddings(
                                    openai_api_key=OPENAI_API_KEY
                                    )

#MY INDEX NAME IS samii 
vector_store = PC.from_documents(docs, embeddings_model, index_name='samii')


from langchain_openai import ChatOpenAI
llm = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
chain = load_qa_chain(llm, chain_type="stuff")



def get_similiar_docs(query, k=2, score=False):
  if score:
    similar_docs = vector_store.similarity_search_with_score(query, k=k)
  else:
    similar_docs = vector_store.similarity_search(query, k=k)
  return similar_docs


llm = OpenAI()
chain = load_qa_chain(llm, chain_type="stuff")
def get_answer(query):
  similar_docs = get_similiar_docs(query)
  answer = chain.run(input_documents=similar_docs, question=query)
  return answer

query = "Does Luna have any friends?"
answer = get_answer(query)
print(answer)

print("abcdef")