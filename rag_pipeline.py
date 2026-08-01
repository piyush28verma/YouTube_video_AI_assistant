from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI


class RAGPipeline:
    def __init__(self, api_key, model_name="gemini-1.5-flash", chunk_size=1000, chunk_overlap=150, top_k=4):
        if not api_key:
            raise ValueError("GOOGLE_API_KEY is required.")

        self.api_key = api_key
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.vector_store = None

        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=self.api_key,
        )
        self.llm = ChatGoogleGenerativeAI(
            model=self.model_name,
            google_api_key=self.api_key,
            temperature=0.2,
        )

    def build_vector_store(self, raw_documents):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )
        chunked_docs = splitter.split_documents(raw_documents)
        if not chunked_docs:
            raise ValueError("No valid chunks generated from transcript.")

        self.vector_store = FAISS.from_documents(chunked_docs, self.embeddings)
        return self.vector_store

    def ask(self, question):
        if not self.vector_store:
            raise ValueError("Vector store is not initialized.")

        retriever = self.vector_store.as_retriever(search_kwargs={"k": self.top_k})
        context_docs = retriever.invoke(question)

        context_text = "\n\n".join(
            [f"[{doc.metadata.get('timestamp', 'N/A')}] {doc.page_content}" for doc in context_docs]
        )

        prompt = f"""You are an intelligent assistant answering questions based ONLY on the YouTube video transcript context provided below.

Context:
{context_text}

Question: {question}

Instructions:
- Base your answer strictly on the transcript context.
- If the context does not contain enough information to answer the question, state: "The transcript does not contain information to answer this question."
- Keep your answer clear, accurate, and concise.

Answer:"""

        response = self.llm.invoke(prompt)
        answer = response.content if hasattr(response, "content") else str(response)

        sources = [
            {
                "timestamp": doc.metadata.get("timestamp", "N/A"),
                "text": doc.page_content,
            }
            for doc in context_docs
        ]

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
        }
