# streamlit_app.py
import streamlit as st
from document_assistant.core import init_session_state, logger
from document_assistant.document_processor import DocumentProcessor
from document_assistant.ui_components import DocumentContainer, ChatInterface
from document_assistant.models import ModelManager
from pathlib import Path
import time

class DocumentAssistant:
    def __init__(self):
        # Initialize session state
        init_session_state()
        self.setup_page()
        
        # Initialize system
        if not st.session_state.get('initialized', False):
            status = st.empty()
            status.info("🚀 Initializing AI models...")
            
            try:
                # Initialize model manager
                model_manager = ModelManager()
                if model_manager.initialize_models():
                    st.session_state.model_manager = model_manager
                    st.session_state.initialized = True
                    self.model_manager = model_manager
                    self.doc_processor = DocumentProcessor(model_manager=model_manager)
                    status.success("✅ System initialized successfully!")
                    time.sleep(1)
                    status.empty()
                else:
                    status.error("❌ Failed to initialize models")
            except Exception as e:
                logger.error(f"Initialization error: {str(e)}")
                status.error(f"Failed to initialize: {str(e)}")
        else:
            # Restore from session state
            self.model_manager = st.session_state.model_manager
            self.doc_processor = DocumentProcessor(model_manager=self.model_manager)

    def setup_page(self):
        """Configure page settings"""
        st.set_page_config(
            page_title="📚 Document Assistant",
            page_icon="📚",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        # Add custom CSS
        st.markdown("""
            <style>
                [data-testid="stSidebar"][aria-expanded="true"] {
                    min-width: 300px;
                    max-width: 300px;
                }
                .main .block-container {
                    max-width: 1200px;
                    padding: 1rem;
                }
                .stExpander {
                    border: 1px solid rgba(128, 128, 128, 0.2);
                    border-radius: 10px;
                    margin-bottom: 1rem;
                }
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                header {visibility: hidden;}
            </style>
        """, unsafe_allow_html=True)

    def initialize_system(self):
        """Initialize the entire system"""
        try:
            status = st.empty()
            
            # Show initialization message
            status.info("🚀 Initializing AI models...")

            # Initialize model manager if not already done
            if not st.session_state.get('initialized', False):
                self.model_manager = ModelManager()
                
                if self.model_manager.initialize_models():
                    # Store in session state
                    st.session_state.model_manager = self.model_manager
                    st.session_state.initialized = True
                    
                    # Initialize document processor
                    self.doc_processor = DocumentProcessor(model_manager=self.model_manager)
                    
                    # Show success message
                    status.success("✅ System initialized successfully!")
                    time.sleep(1)
                    status.empty()
                else:
                    status.error("❌ Failed to initialize models")
            else:
                # If already initialized, restore from session state
                self.model_manager = st.session_state.model_manager
                self.doc_processor = DocumentProcessor(model_manager=self.model_manager)
                
        except Exception as e:
            logger.error(f"System initialization error: {str(e)}")
            st.error(f"Failed to initialize system: {str(e)}")


    def initialize_models(self):
        """Initialize model manager"""
        if not st.session_state.get('initialized', False):
            status = st.empty()
            status.info("🚀 Initializing AI models...")
            
            try:
                # Initialize model manager
                model_manager = ModelManager()
                if model_manager.initialize_models():
                    st.session_state.model_manager = model_manager
                    st.session_state.initialized = True
                    self.doc_processor = DocumentProcessor(model_manager=model_manager)
                    status.success("✅ System initialized successfully!")
                    time.sleep(1)
                    status.empty()
                else:
                    status.error("❌ Failed to initialize models")
            except Exception as e:
                logger.error(f"Model initialization error: {str(e)}")
                status.error("Failed to initialize models")

    def render_sidebar(self):
        """Render sidebar"""
        with st.sidebar:
            st.title("📎 Document Management")
            
            # Only show file upload if initialized
            if st.session_state.get('initialized', False):
                uploaded_files = st.file_uploader(
                    "Upload Documents",
                    type=['pdf', 'docx', 'doc', 'txt'],
                    accept_multiple_files=True,
                    help="Supported formats: PDF, Word (DOCX/DOC), and Text files"
                )
                
                if uploaded_files:
                    self.process_files(uploaded_files)
                
                if st.session_state.documents:
                    st.markdown("### 📑 Selected Documents")
                    
                    if st.button("Deselect All", use_container_width=True):
                        st.session_state.active_docs.clear()
                        st.rerun()
                    
                    for doc_name in st.session_state.documents:
                        selected = st.checkbox(
                            f"📄 {Path(doc_name).stem}",
                            value=doc_name in st.session_state.active_docs
                        )
                        
                        if selected:
                            st.session_state.active_docs.add(doc_name)
                        else:
                            st.session_state.active_docs.discard(doc_name)

    def process_files(self, files):
        """Process uploaded files synchronously"""
        try:
            if not self.doc_processor or not self.model_manager:
                st.error("⚠️ System not initialized")
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
                    for file in new_files:
                        self.doc_processor.process_single_document(file)
            
            # Update tracked files
            st.session_state.previous_files = current_files
            
        except Exception as e:
            logger.error(f"File processing error: {str(e)}")
            st.error("Error processing files")

    def render_main_content(self):
        """Render main content"""
        st.title("📚 Document Assistant")
        
        if not st.session_state.documents:
            st.info("👈 Please upload documents to get started!")
            return
            
        # Create tabs
        tab_docs, tab_chat = st.tabs(["📑 Documents", "💭 Chat"])
        
        # Documents tab
        with tab_docs:
            if st.session_state.active_docs:
                for doc_name in st.session_state.active_docs:
                    doc_info = st.session_state.documents.get(doc_name)
                    if doc_info:
                        DocumentContainer.render(doc_name, doc_info)
            else:
                st.info("👈 Please select documents from the sidebar")
        
        # Chat tab
        with tab_chat:
            ChatInterface.render()

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