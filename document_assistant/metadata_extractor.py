# metadata_extractor.py
import re
import streamlit as st
from typing import Any, Dict, Optional, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# metadata_extractor.py

class MetadataExtractor:
    def __init__(self):
        # Initialize LLM from session state
        self.llm_model = st.session_state.llm_model
        self.llm_config = st.session_state.llm_config

    def extract_metadata(self, text: str, filename: str) -> Dict[str, Any]:
        """Extract metadata using LLM"""
        try:
            # Get first 1000 characters for title extraction
            sample_text = text[:1000]
            
            # Create prompt for title extraction
            prompt = f"""You are analyzing an academic document. Extract ONLY the main title.

Document content:
{sample_text}

Return the title in exactly this format: "TITLE"

For example, if the document is about machine learning, you would return:
"Machine Learning in Neural Networks: A Comprehensive Study"

Only return the title in quotes, nothing else."""

            # Get response from LLM
            response = self.llm_model.generate_content(
                prompt,
                generation_config=self.llm_config
            )
            
            # Extract title from response (between quotes)
            title_match = re.search(r'"([^"]+)"', response.text)
            title = title_match.group(1) if title_match else filename
            
            metadata = {
                'title': title,
                'authors': self._extract_authors(sample_text),
                'year': self._extract_year(sample_text),
                'filename': filename
            }
            
            return metadata
            
        except Exception as e:
            logger.error(f"Metadata extraction error: {str(e)}")
            return {
                'title': filename,
                'authors': [],
                'year': None,
                'filename': filename
            }

    def _extract_authors(self, text: str) -> List[str]:
        """Extract authors using LLM"""
        authors = []
        try:
            prompt = f"""You are analyzing an academic document. Extract ONLY the author names.

Document content:
{text}

Return the authors in exactly this format: "Author1, Author2, Author3"

For example:
"John Smith, Jane Doe, Robert Johnson"

Only return the authors in quotes, nothing else. If no authors are found, return "None"."""

            response = self.llm_model.generate_content(
                prompt,
                generation_config=self.llm_config
            )
            
            # Extract authors from response
            authors_match = re.search(r'"([^"]+)"', response.text)
            if authors_match and authors_match.group(1) != "None":
                # Split authors and remove academic degrees
                raw_authors = authors_match.group(1).split(',')
                for author in raw_authors:
                    # Clean author name
                    author = author.strip()
                    # Remove common academic titles
                    author = re.sub(r'\b(PhD|Ph\.D\.|M\.Phil\.|M\.A\.|Ph\.D|M\.Phil|M\.A)\b', '', author, flags=re.IGNORECASE)
                    # Clean up extra spaces and punctuation
                    author = re.sub(r'\s+', ' ', author).strip(' .,')
                    if author:
                        authors.append(author)

            return authors
            
        except Exception as e:
            logger.error(f"Author extraction error: {str(e)}")
            return []

    def _extract_year(self, text: str) -> Optional[int]:
        """Extract publication year using LLM"""
        try:
            prompt = f"""You are analyzing an academic document. Extract ONLY the publication year.

Document content:
{text}

Return the year in exactly this format: "YYYY"

For example: "2023"

Only return the 4-digit year in quotes. If no year is found, return "None"."""

            response = self.llm_model.generate_content(
                prompt,
                generation_config=self.llm_config
            )
            
            # Extract year from response
            year_match = re.search(r'"(\d{4})"', response.text)
            if year_match:
                year = int(year_match.group(1))
                current_year = datetime.now().year
                if 1900 <= year <= current_year:
                    return year
            return None
            
        except Exception as e:
            logger.error(f"Year extraction error: {str(e)}")
            return None