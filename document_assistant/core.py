# core.py
import streamlit as st
import logging
import google.generativeai as genai
import torch
from typing import Dict, Any, List
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
MAX_FILE_SIZE = 25 * 1024 * 1024  # 25MB for academic papers
MAX_BATCH_SIZE = 5
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def setup_apis() -> Dict[str, Any]:
    """Setup and verify API configurations"""
    try:
        if 'GOOGLE_API_KEY' not in st.secrets:
            st.error("Please set GOOGLE_API_KEY in streamlit secrets")
            st.stop()
        if 'HF_API_KEY' not in st.secrets:
            st.error("Please set HF_API_KEY in streamlit secrets")
            st.stop()

        # Configure Gemini
        genai.configure(api_key=st.secrets['GOOGLE_API_KEY'])
        
        # Setup configurations
        api_config = {
            'model': genai.GenerativeModel('gemini-1.5-pro'),
            'hf_key': st.secrets["HF_API_KEY"],
            'summary_url': "https://api-inference.huggingface.co/models/facebook/bart-large-cnn",
            'image_url': "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-3.5-large",
            'device': DEVICE
        }
        
        logger.info(f"Using device: {DEVICE}")
        return api_config
        
    except Exception as e:
        logger.error(f"API setup error: {str(e)}")
        st.error(f"Failed to setup APIs: {str(e)}")
        raise

def init_session_state():
    """Initialize session state variables"""
    try:
        # Model states
        if 'initialized' not in st.session_state:
            st.session_state.initialized = False
            
        if 'model_manager' not in st.session_state:
            st.session_state.model_manager = None
            
        # Document management
        if 'documents' not in st.session_state:
            st.session_state.documents = {}
            
        if 'active_docs' not in st.session_state:
            st.session_state.active_docs = set()
            
        if 'previous_files' not in st.session_state:
            st.session_state.previous_files = set()
            
        if 'processing_status' not in st.session_state:
            st.session_state.processing_status = {}

        # Initialize API configurations
        if 'api_config' not in st.session_state:
            st.session_state.api_config = setup_apis()
        
        # Initialize LLM model and config
        if 'llm_model' not in st.session_state:
            genai.configure(api_key=st.secrets['GOOGLE_API_KEY'])
            st.session_state.llm_model = genai.GenerativeModel('gemini-1.5-pro')
            st.session_state.llm_config = genai.types.GenerationConfig(
                temperature=0.7,
                max_output_tokens=1024,
            )
        
        # Chat functionality
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
            
    except Exception as e:
        logger.error(f"Session state initialization error: {str(e)}")
        st.error("Failed to initialize application state")
        raise

# Initialize API configurations
try:
    API_CONFIG = setup_apis()
    HEADERS = {
        "Authorization": f"Bearer {API_CONFIG['hf_key']}",
        "Content-Type": "application/json"
    }
except Exception as e:
    logger.error(f"Failed to initialize API configuration: {str(e)}")
    st.error("Failed to initialize application. Please check your API keys and try again.")