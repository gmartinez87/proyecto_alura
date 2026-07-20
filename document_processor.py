import os
from pypdf import PdfReader
# Actualizamos la importación a la nueva estructura de LangChain
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_pdf_document(file_path):
    """
    Carga un archivo PDF y extrae todo su texto.
    
    Args:
        file_path (str): La ruta al archivo PDF.
        
    Returns:
        str: El texto completo extraído del PDF.
    """
    print(f"Leyendo el archivo: {file_path}")
    try:
        reader = PdfReader(file_path)
        text = ""
        for page_num, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        print(f"Extracción completada. {len(reader.pages)} páginas procesadas.")
        return text
    except Exception as e:
        print(f"Error al leer el PDF: {e}")
        return None

def split_text_into_chunks(text, chunk_size=1000, chunk_overlap=200):
    """
    Divide un texto largo en fragmentos más pequeños (chunks).
    
    Utiliza RecursiveCharacterTextSplitter de LangChain, que es ideal para
    texto general, ya que intenta mantener párrafos y oraciones juntos.
    
    Args:
        text (str): El texto a dividir.
        chunk_size (int): El tamaño máximo (en caracteres) de cada fragmento.
        chunk_overlap (int): La cantidad de caracteres superpuestos entre fragmentos adyacentes.
                             Esto ayuda a no perder contexto en los bordes de los chunks.
                             
    Returns:
        list: Una lista de fragmentos de texto (chunks).
    """
    print(f"Dividiendo el texto en chunks (Tamaño: {chunk_size}, Superposición: {chunk_overlap})...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    
    chunks = text_splitter.split_text(text)
    print(f"Texto dividido en {len(chunks)} chunks.")
    return chunks

if __name__ == "__main__":
    # --- PRUEBA LOCAL DEL PASO 1 ---
    # Para probar esto, necesitas un archivo llamado 'documento_prueba.pdf' en el mismo directorio.
    # O cambia 'test_file_path' a un archivo que tengas disponible.
    
    test_file_path = "documento_prueba.pdf"
    
    # Creamos un PDF falso rápido solo para que el script no falle si lo ejecutas directamente
    # En un entorno real, querrás usar un PDF real.
    if not os.path.exists(test_file_path):
        print(f"No se encontró '{test_file_path}'. Crea un archivo PDF con ese nombre para una prueba real.")
        print("Continuando simulación con texto de prueba en memoria...")
        # Texto simulado
        extracted_text = "La academia Alura ofrece cursos de tecnología. " * 50 + "Python es muy popular para la IA. " * 50
    else:
        # 1. Cargar el documento
        extracted_text = load_pdf_document(test_file_path)
    
    if extracted_text:
        # 2. Dividir el texto
        # Usamos chunks pequeños aquí para que sea fácil ver el resultado de la prueba
        document_chunks = split_text_into_chunks(extracted_text, chunk_size=200, chunk_overlap=50)
        
        # 3. Mostrar los primeros chunks como verificación
        print("\n--- Muestra de los primeros 3 chunks ---")
        for i, chunk in enumerate(document_chunks[:3]):
            print(f"Chunk {i+1} (Longitud: {len(chunk)}):")
            print(f"{chunk}\n---")
    else:
        print("No se pudo procesar el documento. Revisa la ruta o el archivo.")