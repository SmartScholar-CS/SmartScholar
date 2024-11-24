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
                authors = [author.strip() for author in authors_match.group(1).split(',')]
                return authors
            return []
            
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

# class MetadataExtractor:
#     """Handles extraction of document metadata"""
    
#     def __init__(self):
#         self.year_pattern = r'\b(19|20)\d{2}\b'
#         self.author_patterns = [
#             # Pattern for comma-separated authors with affiliations
#             r'(?m)^((?:[A-Z][a-zäöüß.-]+(?:\s+[A-Z][a-zäöüß.-]+)*,?\s*)+)(?:\d*\s*)?(?:and |& )?[^@\n]*?(?=\s*(?:University|Institute|@|$))',
#             # Pattern for authors with email addresses
#             r'([A-Z][a-zäöüß.-]+(?:\s+[A-Z][a-zäöüß.-]+)*)\s*(?:<[^>]+>|\([^)]+\)|[\w.+-]+@[\w.-]+)',
#         ]
#         self.title_starters = [
#             "A ", "An ", "The ", "On ", "Towards ", "Analysis ", "Design ",
#             "Implementation ", "Survey ", "Review ", "Study ", "Data ", "Machine ",
#             "Deep ", "Artificial ", "Learning "
#         ]
        
#         # Publishers and common boundaries
#         self.boundaries = r'(?:IEEE|ACM|BIOKDD|Abstract|Author|©|\n)'

#         self.title_patterns = [
#             # Standard academic title pattern
#             rf'^([^@\n]+?)(?=\s*(?:{self.boundaries}))',
#             # Title with colon (subtitle)
#             r'^([^:\n]+:[^@\n]+?)(?=\s*\n)',
#             # Bracketed title
#             r'\[(.*?)\]',
#             # Explicit title marker
#             r'(?i)title:\s*([^\n]+)'
#         ]

#     def extract_metadata(self, text: str, filename: str) -> Dict[str, Any]:
#         """Extract metadata from document text"""
#         try:
#             # Get first 2000 characters for faster processing
#             sample_text = text[:2000]
            
#             # Extract title first
#             title = self._extract_title(sample_text, filename)
            
#             # Get text after title for author extraction
#             title_end = sample_text.find(title) + len(title)
#             author_text = sample_text[title_end:title_end + 500]
            
#             metadata = {
#                 'title': title,
#                 'authors': self._extract_authors(author_text),
#                 'year': self._extract_year(sample_text),
#                 'filename': filename
#             }
            
#             return metadata
            
#         except Exception as e:
#             logger.error(f"Metadata extraction error: {str(e)}")
#             return {
#                 'title': self._format_filename(filename),
#                 'authors': [],
#                 'year': None,
#                 'filename': filename
#             }

#     def _extract_title(self, text: str, filename: str) -> str:
#         """Extract document title with improved accuracy"""
#         try:
#             # Clean text
#             text = text.strip()
#             lines = [line.strip() for line in text.split('\n') if line.strip()]
#             search_text = '\n'.join(lines[:5])
            
#             potential_titles = []
            
#             # Try each pattern
#             for pattern in self.title_patterns:
#                 match = re.search(pattern, search_text, re.MULTILINE)
#                 if match:
#                     title = match.group(1).strip()
#                     title = self._clean_title(title)
#                     if self._is_complete_title(title):
#                         potential_titles.append(title)
            
#             # If we found any valid titles, use the best one
#             if potential_titles:
#                 return self._select_best_title(potential_titles)
            
#             # Fallback: try first line if it looks like a title
#             if lines and self._is_complete_title(lines[0]):
#                 return self._clean_title(lines[0])
            
#             return ""
            
#         except Exception as e:
#             logger.error(f"Title extraction error: {str(e)}")
#             return ""
        
#     def _remove_trailing_authors(self, text: str) -> str:
#         """Remove author names from the end of text"""
#         words = text.split()
#         for i in range(len(words) - 1, -1, -1):
#             potential_name = ' '.join(words[i:min(i + 3, len(words))])
#             if self._is_valid_author_name(potential_name):
#                 return ' '.join(words[:i]).strip()
#         return text
        
#     def _is_valid_title(self, title: str) -> bool:
#         """Validate title format"""
#         if not title:
#             return False
        
#         # Length constraints
#         if len(title) < 10 or len(title) > 300:
#             return False
        
#         # Should not be just a filename
#         if re.match(r'^[\w-]+$', title):
#             return False
        
#         # Should not contain email addresses
#         if '@' in title:
#             return False
        
#         # Should not be just common words
#         if title.lower() in ['title', 'abstract', 'introduction', 'ieee']:
#             return False
        
#         # Should have proper capitalization
#         words = title.split()
#         if not words[0][0].isupper():
#             return False
        
#         # Should not end with common file extensions
#         if re.search(r'\.(?:pdf|doc|docx|txt)$', title, re.IGNORECASE):
#             return False
        
#         return True
    
#     def _clean_title(self, title: str) -> str:
#         """Clean and format title"""
#         try:
#             # Remove author names that follow the title
#             for pattern in [
#                 # Name with affiliation
#                 r'\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:University|Institute|Department)',
#                 # Single or multiple names
#                 r'\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s*,\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)*\s*(?:and|&)?\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*$',
#                 # Single name at end
#                 r'\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*$'
#             ]:
#                 title = re.sub(pattern, '', title)
            
#             # Remove unwanted prefixes/suffixes
#             title = re.sub(r'^(?:Title:|TITLE:|\d+\.\s*)', '', title)
#             title = re.sub(r'\.(?:pdf|doc|docx|txt)$', '', title, flags=re.IGNORECASE)
            
#             # Clean whitespace
#             title = ' '.join(title.split())
            
#             # Remove trailing punctuation except colons in middle
#             if ':' not in title:
#                 title = re.sub(r'[.,;!?]+$', '', title)
            
#             return title.strip()
            
#         except Exception as e:
#             logger.error(f"Title cleaning error: {str(e)}")
#             return title

#     def _is_complete_title(self, title: str) -> bool:
#         """Check if title is complete and valid"""
#         if not title or len(title) < 10 or len(title) > 300:
#             return False
            
#         # Check for common title starters
#         if not any(title.startswith(starter) for starter in self.title_starters):
#             first_word = title.split()[0]
#             if not first_word[0].isupper():
#                 return False
        
#         # Must contain at least 3 words
#         words = title.split()
#         if len(words) < 3:
#             return False
        
#         # Check for incomplete titles
#         if any(title.lower().endswith(end) for end in ['in', 'of', 'on', 'and', 'the', 'a', 'an']):
#             return False
        
#         # Should not contain email addresses
#         if '@' in title:
#             return False
        
#         # Should not be just common words
#         if title.lower() in ['title', 'abstract', 'introduction', 'ieee']:
#             return False
        
#         return True

#     def _select_best_title(self, titles: List[str]) -> str:
#         """Select the best title from multiple candidates"""
#         if not titles:
#             return ""
            
#         # Score each title
#         scored_titles = []
#         for title in titles:
#             score = 0
            
#             # Prefer titles that start with common academic words
#             if any(title.startswith(starter) for starter in self.title_starters):
#                 score += 2
            
#             # Prefer titles with subtitles (containing colon)
#             if ':' in title:
#                 score += 1
            
#             # Prefer longer titles (up to a point)
#             words = title.split()
#             if 5 <= len(words) <= 15:
#                 score += 1
            
#             scored_titles.append((score, title))
        
#         # Return the highest scoring title
#         return max(scored_titles, key=lambda x: x[0])[1]
        
#     def _format_filename(self, filename: str) -> str:
#         """Format filename as title if needed"""
#         # Remove extension
#         name = filename.rsplit('.', 1)[0]
        
#         # Replace underscores and hyphens with spaces
#         name = name.replace('_', ' ').replace('-', ' ')
        
#         # Proper case words (but keep acronyms)
#         words = name.split()
#         formatted_words = []
#         for word in words:
#             if not word.isupper():  # Don't change acronyms
#                 word = word.capitalize()
#             formatted_words.append(word)
        
#         return ' '.join(formatted_words)

#     def _extract_authors(self, text: str) -> List[str]:
#         """Extract author names with improved handling"""
#         try:
#             authors = []
#             lines = text.split('\n')
            
#             # Look at first few lines where authors typically appear
#             search_text = '\n'.join(lines[:10])
            
#             for pattern in self.author_patterns:
#                 matches = re.finditer(pattern, search_text)
#                 for match in matches:
#                     author_text = match.group(1).strip()
#                     # Split on common separators
#                     author_list = re.split(r',|\band\b|&', author_text)
                    
#                     for author in author_list:
#                         author = author.strip()
#                         # Validate author name format
#                         if self._is_valid_author_name(author):
#                             # Remove any trailing numbers or affiliations
#                             author = re.sub(r'\d+$', '', author).strip()
#                             authors.append(author)
            
#             # Remove duplicates while preserving order
#             seen = set()
#             unique_authors = []
#             for author in authors:
#                 if author not in seen:
#                     seen.add(author)
#                     unique_authors.append(author)
            
#             return unique_authors
            
#         except Exception as e:
#             logger.error(f"Author extraction error: {str(e)}")
#             return []
        
#     def _is_valid_author_name(self, text: str) -> bool:
#         """Check if text looks like an author name"""
#         if not text:
#             return False
            
#         # Should be 2-4 words, each capitalized
#         words = text.split()
#         if not (2 <= len(words) <= 4):
#             return False
            
#         # Each word should be properly capitalized
#         return all(
#             word[0].isupper() and word[1:].islower() 
#             for word in words 
#             if len(word) > 1
#         )

#     def _extract_year(self, text: str) -> Optional[int]:
#         """Extract publication year with validation"""
#         try:
#             # Find all 4-digit years
#             years = re.findall(self.year_pattern, text)
#             if years:
#                 # Convert to integers and filter valid years
#                 current_year = datetime.now().year
#                 valid_years = [
#                     int(year) for year in years 
#                     if 1900 <= int(year) <= current_year
#                 ]
                
#                 if valid_years:
#                     # Return the most likely publication year
#                     return min(valid_years)  # Usually the first/earliest year
#             return None
            
#         except Exception as e:
#             logger.error(f"Year extraction error: {str(e)}")
#             return None