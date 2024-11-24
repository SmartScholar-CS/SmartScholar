# streamlit_app.py
import streamlit as st
from document_assistant.core import init_session_state, logger
from document_assistant.document_processor import DocumentProcessor
from document_assistant.ui_components import DocumentContainer, ChatInterface
from document_assistant.models import ModelManager
from pathlib import Path
import time

class DocumentAssistant:
    """Main application class"""
    
    def __init__(self):
        """Initialize the application"""
        try:
            # Initialize session state and APIs
            init_session_state()
            
            # Setup page configuration
            self.setup_page()
            
            # Initialize models if not already done
            if not st.session_state.get('initialized', False):
                self.initialize_models()
            
            # Initialize document processor
            self.doc_processor = DocumentProcessor(
                model_manager=st.session_state.get('model_manager')
            )
            
        except Exception as e:
            logger.error(f"Initialization error: {str(e)}")
            st.error("Failed to initialize application")
            raise

    def setup_page(self):
        """Configure page settings and styles"""
        try:
            st.set_page_config(
                page_title="📚 Document Assistant",
                page_icon="📚",
                layout="wide",
                initial_sidebar_state="expanded"
            )

            # Add custom CSS
            st.markdown("""
                <style>
                /* Hide Streamlit elements */
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                
                /* Custom styling */
                .stApp {
                    max-width: 1200px;
                    margin: 0 auto;
                }
                
                .stMetric {
                    background-color: rgba(255, 255, 255, 0.05);
                    border-radius: 5px;
                    padding: 10px;
                }
                
                /* Status indicators */
                .status-indicator {
                    display: inline-block;
                    width: 10px;
                    height: 10px;
                    border-radius: 50%;
                    margin-right: 5px;
                }
                .status-ready { background-color: #28a745; }
                .status-pending { background-color: #ffc107; }
                .status-failed { background-color: #dc3545; }
                </style>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            logger.error(f"Page setup error: {str(e)}")
            st.error("Error setting up page")

    def initialize_models(self):
        """Initialize AI models"""
        try:
            status_placeholder = st.empty()
            status_placeholder.info("🚀 Initializing AI models...")
            start_time = time.time()

            # Initialize model manager
            model_manager = ModelManager()
            
            # Initialize models
            if model_manager.initialize_models():
                # Store in session state
                st.session_state.model_manager = model_manager
                st.session_state.initialized = True
                
                # Show success message
                elapsed_time = time.time() - start_time
                status_placeholder.success(
                    f"✅ System initialized successfully! ({elapsed_time:.1f}s)"
                )
                time.sleep(1)
                status_placeholder.empty()
            else:
                status_placeholder.error(
                    "❌ Failed to initialize models. Please refresh the page."
                )
                
        except Exception as e:
            logger.error(f"Model initialization error: {str(e)}")
            st.error("Failed to initialize AI models")

    def show_model_status(self):
        """Display model initialization status"""
        try:
            if st.session_state.get('model_manager'):
                st.sidebar.markdown("### 🔧 System Status")
                model_manager = st.session_state.model_manager
                
                status = {
                    'Classifier': model_manager.classifier is not None,
                    'Summarizer': model_manager.summary_model is not None,
                    'Similarity': model_manager.similarity_model is not None,
                    'Image Generator': model_manager.image_generator is not None
                }
                
                for model, ready in status.items():
                    status_class = "ready" if ready else "failed"
                    status_text = "Ready" if ready else "Failed"
                    st.sidebar.markdown(f"""
                        <div>
                            <span class="status-indicator status-{status_class}"></span>
                            {model}: {status_text}
                        </div>
                    """, unsafe_allow_html=True)
                
                st.sidebar.markdown("---")
                
        except Exception as e:
            logger.error(f"Status display error: {str(e)}")

    def render_sidebar(self):
        """Render sidebar with file upload and document management"""
        try:
            with st.sidebar:
                st.title("📎 Document Management")
                
                # Show model status
                self.show_model_status()
                
                # Only show file upload if models are initialized
                if st.session_state.get('initialized', False):
                    # File uploader
                    uploaded_files = st.file_uploader(
                        "Upload Documents",
                        type=['pdf', 'docx', 'doc', 'txt'],
                        accept_multiple_files=True,
                        help="Supported formats: PDF, Word (DOCX/DOC), and Text files"
                    )
                    
                    if uploaded_files:
                        self.process_uploads(uploaded_files)
                    
                    # Document selection
                    if st.session_state.documents:
                        st.markdown("### 📑 Selected Documents")
                        
                        # Deselect all button
                        if st.button("Deselect All", use_container_width=True):
                            st.session_state.active_docs.clear()
                            st.rerun()
                        
                        # Document checkboxes
                        for doc_name in st.session_state.documents:
                            selected = st.checkbox(
                                f"📄 {Path(doc_name).stem}",
                                value=doc_name in st.session_state.active_docs,
                                key=f"select_{doc_name}"
                            )
                            
                            if selected:
                                st.session_state.active_docs.add(doc_name)
                            else:
                                st.session_state.active_docs.discard(doc_name)
                                
        except Exception as e:
            logger.error(f"Sidebar rendering error: {str(e)}")
            st.error("Error in sidebar")

    def process_uploads(self, files):
        """Process uploaded files"""
        try:
            if not self.doc_processor:
                st.error("⚠️ System not fully initialized. Please wait...")
                return
                
            current_files = {f.name for f in files}
            
            # Handle removed files
            removed_files = st.session_state.previous_files - current_files
            for file_name in removed_files:
                if file_name in st.session_state.documents:
                    del st.session_state.documents[file_name]
                    st.session_state.active_docs.discard(file_name)
            
            # Process new files
            new_files = [f for f in files if f.name not in st.session_state.documents]
            if new_files:
                with st.spinner(f"Processing {len(new_files)} file(s)..."):
                    self.doc_processor.process_documents(new_files)
            
            # Update tracked files
            st.session_state.previous_files = current_files
            st.session_state.current_file_count = len(current_files)
            
        except Exception as e:
            logger.error(f"File processing error: {str(e)}")
            st.error("Error processing files")

    def render_main_content(self):
        """Render main content area"""
        try:
            st.title("📚 Document Assistant")
            
            if not st.session_state.get('initialized', False):
                st.info("🔄 System initialization in progress...")
                return
                
            if not st.session_state.documents:
                st.info("👈 Please upload documents to get started!")
                return
                
            # Create tabs
            tab_chat, tab_docs = st.tabs(["💭 Chat", "📑 Documents"])
            
            # Render chat tab
            with tab_chat:
                ChatInterface.render()
            
            # Render documents tab
            with tab_docs:
                if st.session_state.active_docs:
                    for doc_name in st.session_state.active_docs:
                        doc_info = st.session_state.documents.get(doc_name)
                        if doc_info:
                            DocumentContainer.render(doc_name, doc_info)
                else:
                    st.info("👈 Please select documents from the sidebar to view details")
                    
        except Exception as e:
            logger.error(f"Content rendering error: {str(e)}")
            st.error("Error displaying content")

    def run(self):
        """Run the application"""
        try:
            # Render sidebar
            self.render_sidebar()
            
            # Render main content
            self.render_main_content()
            
            # Footer
            st.markdown("---")
            st.markdown(
                "💡 Powered by AI | Made with Streamlit",
                help="Using state-of-the-art AI models for document analysis"
            )
            
        except Exception as e:
            logger.error(f"Application error: {str(e)}")
            st.error("An error occurred. Please refresh the page.")

def main():
    """Application entry point"""
    try:
        app = DocumentAssistant()
        app.run()
    except Exception as e:
        logger.error(f"Startup error: {str(e)}")
        st.error("Failed to start the application. Please try again.")

if __name__ == "__main__":
    main()