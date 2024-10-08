from operator import itemgetter

from langchain_openai import ChatOpenAI
from langchain.chains.base import Chain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
from langchain_core.vectorstores import VectorStoreRetriever



def get_qa_chain(
        retriever: VectorStoreRetriever,
        streaming: bool = False
    ) -> Chain:
    template = """
    Answer any questions based solely on the context below. If the context
    doesn't provide the answer, still do your best to answer the question 
    factually, but indicate there isn't a clear answer in the context 
    and that you're giving a best-effort response.

    Question:
    {question}

    Context:
    {context}
    """
    primary_qa_llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, streaming=streaming)

    prompt = ChatPromptTemplate.from_template(template)

    retrieval_augmented_qa_chain = (
        # INVOKE CHAIN WITH: {"question" : "<<SOME USER QUESTION>>"}
        # "question" : populated by getting the value of the "question" key
        # "context"  : populated by getting the value of the "question" key and chaining it into the base_retriever
        {"context": itemgetter("question") | retriever, "question": itemgetter("question")}
        # "context"  : is assigned to a RunnablePassthrough object (will not be called or considered in the next step)
        #              by getting the value of the "context" key from the previous step
        | RunnablePassthrough.assign(context=itemgetter("context"))
        # "answer" : the "context" and "question" values are used to format our prompt object and then piped
        #              into the LLM and stored in a key called "answer": NOTE: Key MUST be "answer" for LangSmith.
        # "contexts"  : populated by getting the value of the "context" key from the previous step.
        #               NOTE: Key must be "contexts" for LangSmith  
        | {"answer": prompt | primary_qa_llm, "contexts": itemgetter("context")}
    )

    return retrieval_augmented_qa_chain