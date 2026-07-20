import streamlit as st
import os
import ssl
import httpx
import warnings
from dotenv import load_dotenv

# --- WORKAROUND DEFINITIVO PARA REDES CORPORATIVAS / ERROR SSL ---
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except AttributeError:
    pass

original_init = httpx.Client.__init__
def patched_init(self, *args, **kwargs):
    kwargs['verify'] = False
    original_init(self, *args, **kwargs)
httpx.Client.__init__ = patched_init

original_async_init = httpx.AsyncClient.__init__
def patched_async_init(self, *args, **kwargs):
    kwargs['verify'] = False
    original_async_init(self, *args, **kwargs)
httpx.AsyncClient.__init__ = patched_async_init

warnings.filterwarnings("ignore")
# -----------------------------------------------------------------

load_dotenv()

from document_processor import load_pdf_document, split_text_into_chunks
from agent_core import create_vectorstore, setup_rag_chain

st.set_page_config(
    page_title="Alura Agente IA",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed" # Added to ensure sidebar starts hidden if any remnants exist
)

# Custom CSS to completely hide the sidebar toggle button and the sidebar container
hide_sidebar_style = """
    <style>
        [data-testid="collapsedControl"] {
            display: none;
        }
        [data-testid="stSidebar"] {
            display: none;
        }
    </style>
"""
st.markdown(hide_sidebar_style, unsafe_allow_html=True)


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "document_processed" not in st.session_state:
    st.session_state.document_processed = False

cohere_api_key = os.getenv("COHERE_API_KEY", "")
document_path = os.getenv("DOCUMENT_PATH", "documento.pdf")

has_valid_config = cohere_api_key and cohere_api_key != "tu_api_key_de_cohere_aqui"

# Removed the entire with st.sidebar block

if has_valid_config and not st.session_state.document_processed:
    if not os.path.exists(document_path):
        st.error(f"⚠️ No se encontró el archivo '{document_path}'. Asegúrate de que esté en la carpeta del proyecto.")
    else:
        with st.spinner("Procesando documento inicial y conectando con la IA..."):
            try:
                raw_text = load_pdf_document(document_path)
                if not raw_text:
                    st.error("No se pudo extraer texto del PDF.")
                    st.stop()
                    
                chunks = split_text_into_chunks(raw_text)
                vectorstore = create_vectorstore(chunks, cohere_api_key)
                
                if not vectorstore:
                    st.error("Error al crear la base de datos vectorial.")
                    st.stop()
                    
                rag_chain = setup_rag_chain(vectorstore, cohere_api_key)
                
                st.session_state.rag_chain = rag_chain
                st.session_state.document_processed = True
                st.session_state.chat_history = [] 
                
            except Exception as e:
                st.error(f"Ocurrió un error en la configuración inicial: {e}")

st.title("🤖 Alura Agente Corporativo")

if not st.session_state.document_processed:
    if not has_valid_config:
         st.warning("⚠️ Falta la configuración en el archivo .env. Asegúrate de configurar COHERE_API_KEY.")
    else:
         st.markdown("Hubo un error al procesar el documento. Revisa la consola o los mensajes de error arriba.")
else:
    st.markdown("¡Hola! Soy Alura Agente. Hazme preguntas sobre el documento cargado.")

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Haz una pregunta sobre el documento...")

if user_input:
    if not st.session_state.document_processed or st.session_state.rag_chain is None:
        st.warning("⚠️ El agente aún no está listo. Verifica la configuración en .env.")
    else:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
            
        with st.chat_message("assistant"):
            with st.spinner("Buscando en los documentos..."):
                try:
                    response = st.session_state.rag_chain.invoke({"input": user_input})
                    answer = response["answer"]
                    
                    st.markdown(answer)
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"Error al generar la respuesta: {e}")