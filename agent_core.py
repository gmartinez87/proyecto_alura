import os
from langchain_cohere import CohereEmbeddings, ChatCohere
from langchain_community.vectorstores import FAISS
# Importaciones corregidas para LangChain moderno
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

def create_vectorstore(text_chunks, cohere_api_key):
    """
    Toma los fragmentos de texto, genera embeddings usando Cohere 
    y los almacena en un índice vectorial local (FAISS).
    """
    print("Generando embeddings con Cohere y creando Vectorstore FAISS...")
    
    if not text_chunks:
        raise ValueError("El documento no contiene texto legible o está vacío.")
        
    try:
        # En el futuro cercano, también podríamos necesitar actualizar el modelo de embeddings
        # a embed-multilingual-v4.0, pero v3.0 aún es ampliamente soportado.
        embeddings = CohereEmbeddings(
            cohere_api_key=cohere_api_key, 
            model="embed-multilingual-v3.0"
        )
        
        vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        print("Vectorstore creado exitosamente.")
        return vectorstore
    except Exception as e:
        print(f"Error al crear el vectorstore: {e}")
        raise e 

def setup_rag_chain(vectorstore, cohere_api_key):
    """
    Configura el motor de respuesta: un modelo de lenguaje conectado al buscador.
    """
    print("Configurando el LLM de Cohere y la cadena RAG...")
    
    # 1. Inicializamos el LLM de Cohere con el modelo de NUEVA GENERACIÓN (2026)
    llm = ChatCohere(
        cohere_api_key=cohere_api_key,
        model="command-a-03-2025", # <- ¡EL CAMBIO CRÍTICO ESTÁ AQUÍ!
        temperature=0.3 
    )
    
    # 2. Creamos el Prompt Template
    system_prompt = (
        "Eres 'Alura Agente', un asistente corporativo útil y profesional. "
        "Usa los siguientes fragmentos de contexto recuperado para responder a la pregunta. "
        "Si no sabes la respuesta basándote en el contexto, di simplemente que no tienes "
        "esa información en los documentos proporcionados. No inventes información.\n\n"
        "Contexto:\n{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    # 3. Creamos la cadena que combina los documentos con el LLM
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    
    # 4. Configuramos el recuperador
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 4} 
    )
    
    # 5. Unimos el recuperador y la cadena de respuesta
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    print("Cadena RAG configurada exitosamente.")
    return rag_chain

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    API_KEY = os.environ.get("COHERE_API_KEY", "")
    
    if not API_KEY or API_KEY == "tu_api_key_de_cohere_aqui":
        print("⚠️ ADVERTENCIA: Por favor, configura tu API Key en .env")
    else:
        dummy_chunks = [
            "La academia Alura ofrece cursos de programación en Python.",
            "Alura Agente fue desarrollado con LangChain."
        ]
        v_store = create_vectorstore(dummy_chunks, API_KEY)
        if v_store:
            chain = setup_rag_chain(v_store, API_KEY)
            respuesta = chain.invoke({"input": "¿Con qué se desarrolló Alura Agente?"})
            print(f"\nRespuesta del Agente: {respuesta['answer']}")