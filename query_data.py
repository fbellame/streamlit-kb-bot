from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import ChatVectorDBChain

from langchain.schema import (
    SystemMessage
)

system_template="""Vous êtes un assistant IA pour répondre aux questions sur les voeux présidentiels le plus récent. 
On vous donne les parties extraites suivantes d\'un long document et une question. Fournissez une réponse conversationnelle.
Si vous ne connaissez pas la réponse, dites simplement "Hmm, je ne suis pas sûr". N'essayez pas d'inventer une réponse.
Si la question ne porte pas sur les voeux présidentiels les plus récent, informez-les poliment que vous êtes réglé pour ne répondre qu\'aux questions sur les voeux présidentiels les plus récents.
----------------
{context}"""
messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}")
]
QA_PROMPT = ChatPromptTemplate.from_messages(messages)

def get_chain(vectorstore):
    llm = ChatOpenAI(model="gpt-4", temperature=0)

    qa_chain = ChatVectorDBChain.from_llm(
        llm,
        vectorstore,
        qa_prompt=QA_PROMPT
    )
    return qa_chain