# 📚 SmartScholar

A powerful document analysis and research assistant powered by AI, built with Streamlit and advanced language models.

[![Made with Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io)

## 🌟 Features

- **📄 Multi-Format Document Processing**: Support for PDF, DOCX, DOC, and TXT files
- **🤖 AI-Powered Analysis**: 
  - Document classification
  - Summary generation
  - Topic extraction
  - Similarity analysis
- **💬 Interactive Chat Interface**: Chat with your documents using Google's Gemini 1.5 Pro
- **📊 Research Analysis Dashboard**:
  - Research trends visualization
  - Geographic analysis of research output
  - Impact metrics and citations analysis
- **🎨 Document Visualization**: AI-generated visual representations of document content
- **📈 Advanced Metrics**: Citation analysis, impact factors, and research growth patterns

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- GPU support (optional but recommended)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/SmartScholar-CS/SmartScholar.git
cd SmartScholar
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.streamlit/secrets.toml` file with:
```toml
GOOGLE_API_KEY = "your_google_api_key"
HF_API_KEY = "your_huggingface_api_key"
```

4. Run the application:
```bash
streamlit run streamlit_app.py
```

## 🛠️ Technologies Used

- **Frontend**: Streamlit
- **AI/ML**:
  - Google Gemini 1.5 Pro for LLM
  - HuggingFace Transformers
  - Sentence Transformers
  - PyTorch
- **Document Processing**:
  - PyPDF2
  - python-docx
  - docx2txt
- **Visualization**:
  - Plotly
  - Matplotlib
  - Seaborn

## 📊 Project Structure

```
SmartScholar/
├── document_assistant/
│   ├── __init__.py
│   ├── analysis_components.py
│   ├── core.py
│   ├── document_processor.py
│   ├── metadata_extractor.py
│   ├── models.py
│   ├── processors.py
│   └── ui_components.py
├── requirements.txt
└── streamlit_app.py
```

## 🔧 System Requirements

- RAM: 8GB minimum, 16GB recommended
- Storage: 2GB free space
- GPU: Optional but recommended for better performance
- Internet connection for API access

## 📝 Features in Detail

### Document Processing
- Automatic text extraction from multiple file formats
- Metadata extraction including title, authors, and publication date
- Document structure analysis and section extraction

### AI Analysis
- Zero-shot document classification
- Extractive and abstractive summarization
- Citation and reference extraction
- Document similarity calculation

### Research Analytics
- Publication trend analysis
- Geographic research distribution
- Impact factor calculation
- Citation network analysis

### Interactive Interface
- Real-time document chat
- Dynamic visualization updates
- Interactive research dashboards
- Multi-document comparison

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Google's Gemini 1.5 Pro for advanced language processing
- HuggingFace for providing state-of-the-art NLP models
- Streamlit for the amazing web framework

## 📞 Support

For support, please open an issue in the GitHub repository or contact the maintainers.