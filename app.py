import os
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from git import Repo
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Configuration and constants
SUPPORTED_EXTENSIONS = {'.py', '.js', '.tsx', '.jsx', '.ipynb', '.java',
                        '.cpp', '.ts', '.go', '.rs', '.vue', '.swift', '.c', '.h'}

IGNORED_DIRS = {'node_modules', 'venv', 'env', 'dist', 'build', '.git',
                '__pycache__', '.next', '.vscode', 'vendor'}

# Utility Functions
def clone_repository(repo_url):
    """Clones a GitHub repository to a temporary directory."""
    try:
        repo_name = repo_url.split("/")[-1]
        repo_path = os.path.join("/tmp", repo_name)
        Repo.clone_from(repo_url, str(repo_path))
        return str(repo_path)
    except Exception as e:
        st.error(f"Error cloning repository: {e}")
        return None

def get_file_content(file_path, repo_path):
    """Get content of a single file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        rel_path = os.path.relpath(file_path, repo_path)
        return {
            "name": rel_path,
            "content": content
        }
    except Exception as e:
        st.warning(f"Error processing file {file_path}: {str(e)}")
        return None

def get_main_files_content(repo_path):
    """Get content of supported code files from the local repository."""
    files_content = []
    try:
        for root, _, files in os.walk(repo_path):
            if any(ignored_dir in root for ignored_dir in IGNORED_DIRS):
                continue

            for file in files:
                file_path = os.path.join(root, file)
                if os.path.splitext(file)[1] in SUPPORTED_EXTENSIONS:
                    file_content = get_file_content(file_path, repo_path)
                    if file_content:
                        files_content.append(file_content)
    except Exception as e:
        st.error(f"Error reading repository: {str(e)}")
    return files_content

def get_huggingface_embeddings(text, model_name="sentence-transformers/all-mpnet-base-v2"):
    """Generate embeddings using HuggingFace model."""
    model = SentenceTransformer(model_name)
    return model.encode(text)

def setup_vector_store(file_content, repo_url, pinecone_api_key):
    """Set up Pinecone vector store with repository documents."""
    try:
        # Initialize Pinecone
        pc = Pinecone(api_key=pinecone_api_key)
        
        # Ensure the index exists
        index_name = "codebase-rag"
        if index_name not in pc.list_indexes().names():
            pc.create_index(
                name=index_name,
                dimension=768,  # Matching the embedding model dimension
                metric="cosine"
            )
        
        # Create documents
        documents = []
        for file in file_content:
            doc = Document(
                page_content=f"{file['name']}\n{file['content']}",
                metadata={"source": file['name'], "text": file['content']}
            )
            documents.append(doc)

        # Create vector store
        vectorstore = PineconeVectorStore.from_documents(
            documents=documents,
            embedding=HuggingFaceEmbeddings(),
            index_name=index_name,
            namespace=repo_url
        )
        
        return pc.Index(index_name)
    except Exception as e:
        st.error(f"Error setting up vector store: {e}")
        return None

def perform_rag(query, pinecone_index, groq_api_key, repo_url):
    """Perform Retrieval Augmented Generation."""
    try:
        raw_query_embedding = get_huggingface_embeddings(query)

        top_matches = pinecone_index.query(
            vector=raw_query_embedding.tolist(), 
            top_k=5, 
            include_metadata=True, 
            namespace=repo_url
        )

        contexts = [item['metadata']['text'] for item in top_matches['matches']]

        augmented_query = "<CONTEXT>\n" + "\n\n-------\n\n".join(contexts[:10]) + "\n-------\n</CONTEXT>\n\n\n\nMY QUESTION:\n" + query

        client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=groq_api_key
        )

        system_prompt = """You are a Senior Software Engineer specializing in code analysis.
        Answer questions about the codebase precisely and technically.
        Always consider the provided context when formulating your response."""

        llm_response = client.chat.completions.create(
            model="llama-3.1-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": augmented_query}
            ]
        )

        return llm_response.choices[0].message.content
    except Exception as e:
        st.error(f"RAG error: {e}")
        return "Unable to generate response. Please try again."

# Streamlit App
def main():
    st.title("🤖 Codebase RAG Chatbot")
    
    # Get API keys from environment variables
    pinecone_api_key = os.getenv('PINECONE_API_KEY')
    groq_api_key = os.getenv('GROQ_API_KEY')

    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # GitHub Repo URL Input
        repo_url = st.text_input("GitHub Repository URL", 
                                 placeholder="https://github.com/username/repo")
        
        # Display API key status
        st.write("Pinecone API Key:", "✅ Loaded" if pinecone_api_key else "❌ Not Found")
        st.write("Groq API Key:", "✅ Loaded" if groq_api_key else "❌ Not Found")
        
        # Process and Index Button
        if st.button("Process Repository"):
            if repo_url and pinecone_api_key and groq_api_key:
                with st.spinner("Cloning Repository..."):
                    repo_path = clone_repository(repo_url)
                
                if repo_path:
                    with st.spinner("Extracting File Contents..."):
                        file_content = get_main_files_content(repo_path)
                    
                    with st.spinner("Setting up Vector Store..."):
                        st.session_state.pinecone_index = setup_vector_store(
                            file_content, repo_url, pinecone_api_key
                        )
                    
                    st.session_state.repo_url = repo_url
                    st.success("Repository processed successfully!")
            else:
                st.warning("Please provide repository URL and ensure API keys are set in .env")

    # Chat Interface
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question about the codebase"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Perform RAG if repository is processed
        if hasattr(st.session_state, 'pinecone_index') and hasattr(st.session_state, 'repo_url'):
            with st.spinner("Generating response..."):
                response = perform_rag(
                    query=prompt, 
                    pinecone_index=st.session_state.pinecone_index,
                    groq_api_key=groq_api_key,
                    repo_url=st.session_state.repo_url
                )
            
            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                st.markdown(response)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
        else:
            st.warning("Please process a repository first using the sidebar.")

if __name__ == "__main__":
    main()