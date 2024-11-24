# document_processor.py
from document_assistant.core import logger, MAX_FILE_SIZE
from document_assistant.processors import PDFProcessor, DocxProcessor, DocProcessor, TxtProcessor
from document_assistant.models import ModelManager
from typing import Dict, List, Optional
from pathlib import Path
import streamlit as st

class DocumentProcessor:
    """Main document processing class"""
    
    def __init__(self, model_manager: Optional[ModelManager] = None):
        """Initialize with file processors and model manager"""
        self.processors = {
            'pdf': PDFProcessor.process,
            'docx': DocxProcessor.process,
            'doc': DocProcessor.process,
            'txt': TxtProcessor.process
        }
        self.model_manager = model_manager
        print(f"DocumentProcessor initialized with model manager: {self.model_manager is not None}")

    def check_models(self) -> bool:
        """Check if models are ready"""
        if not self.model_manager:
            st.error("⚠️ Models not properly initialized")
            return False
        if not self.model_manager.check_initialized():
            st.warning("⌛ Waiting for models to initialize...")
            return False
        return True

    def process_documents(self, files: List) -> None:
        """Process multiple documents"""
        try:
            if not self.check_models():
                st.error("Models not fully initialized. Please wait...")
                return

            for file in files:
                try:
                    if file.name not in st.session_state.documents:
                        self.process_single_document(file)
                except Exception as e:
                    logger.error(f"Error processing {file.name}: {str(e)}")
                    st.error(f"Failed to process {file.name}")
                    
        except Exception as e:
            logger.error(f"Document processing error: {str(e)}")
            st.error("An error occurred during document processing")

    def process_single_document(self, file) -> None:
        """Process a single document with all features"""
        try:
            progress = st.empty()
            progress.text(f"📄 Processing {file.name}...")
            
            # Check file size
            if file.size > MAX_FILE_SIZE:
                raise ValueError(f"File size exceeds limit ({MAX_FILE_SIZE/1024/1024}MB)")
            
            # 1. Extract text and metadata
            doc_info = self._process_text(file)
            if not doc_info:
                return

            # 2. Generate summary
            if hasattr(self.model_manager, 'summary_model'):
                progress.text("📝 Generating summary...")
                summary = self.model_manager.summary_model.generate_summary(doc_info['content'])
                if summary and summary != "Could not generate summary.":
                    doc_info['summary'] = summary

            # 3. Classify document
            progress.text("🏷️ Classifying content...")
            classification = self.model_manager.classify_document(doc_info['content'])
            if classification:
                doc_info['classification'] = classification

            # 4. Generate visualization
            if hasattr(self.model_manager, 'image_generator'):
                progress.text("🎨 Creating visualization...")
                image, error = self.model_manager.image_generator.generate_image(
                    doc_info.get('summary', doc_info['content'][:500]),
                    doc_info.get('title', Path(file.name).stem)
                )
                if image:
                    doc_info['image'] = image
                elif error:
                    logger.warning(f"Image generation warning: {error}")

            # Store processed document
            st.session_state.documents[file.name] = doc_info
            st.session_state.active_docs.add(file.name)

            # 5. Calculate similarities
            if len(st.session_state.documents) > 1 and hasattr(self.model_manager, 'similarity_model'):
                progress.text("🔄 Analyzing similarities...")
                self._update_similarities(file.name, doc_info)

            # Complete
            progress.empty()
            st.success(f"✅ Processed {file.name} successfully!")
            
        except Exception as e:
            logger.error(f"Error processing {file.name}: {str(e)}")
            st.error(f"Error processing {file.name}: {str(e)}")
            raise  # This will help with debugging

    def _process_text(self, file) -> Optional[Dict]:
        """Process document text"""
        try:
            file_ext = Path(file.name).suffix[1:].lower()
            if file_ext not in self.processors:
                raise ValueError(f"Unsupported file format: {file_ext}")
            
            result = self.processors[file_ext](file)
            if not result or not result.get('content'):
                raise ValueError("No content could be extracted")
            
            return result
            
        except Exception as e:
            logger.error(f"Text processing error: {str(e)}")
            st.error(f"Could not process {file.name}: {str(e)}")
            return None

    def _update_similarities(self, new_file: str, new_doc: Dict) -> None:
        """Update similarity scores for all documents"""
        try:
            docs = st.session_state.documents
            
            # Initialize similarities
            if 'similarities' not in new_doc:
                new_doc['similarities'] = {}
            
            # Calculate similarities with existing documents
            for doc_name, doc in docs.items():
                if doc_name != new_file:
                    if 'similarities' not in doc:
                        doc['similarities'] = {}
                        
                    # Calculate similarity
                    scores = self.model_manager.similarity_model.calculate_similarity(
                        new_doc['content'],
                        [doc['content']]
                    )
                    
                    if scores and len(scores) > 0:
                        score = scores[0]
                        # Update both documents
                        new_doc['similarities'][doc_name] = score
                        doc['similarities'][new_file] = score
                        
        except Exception as e:
            logger.error(f"Similarity update error: {str(e)}")
            logger.warning("Could not calculate document similarities")