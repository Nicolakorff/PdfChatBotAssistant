# PdfChatbotAssistant

**Try it!!** https://pdfchatbotassistant.streamlit.app/

This is a chatbot assistant that allows you to converse with a chatbot about the content of your PDF files. Simply upload your PDFs, it processes the content, and then you can ask it any questions about the information contained within.
I must say the answers are not what expected, but I love them profoundly. They make me think and laugh. Still learning what each model can do.

## Key Features

- **Multiple PDF Upload:** Allows you to upload multiple PDF files at once.
- **Text Processing:** Extracts text from PDFs and cleans it for better understanding.
- **Text Chunking:** Divides text into smaller chunks to facilitate processing by the language model.
- **Knowledge Base Creation:** Uses embeddings to create a vector knowledge base of PDF content.
- **Interactive Conversation:** Allows you to ask questions about PDF content and receive relevant answers.
- **Conversation History:** Maintains a history of the conversation for richer context.
- **Simple User Interface:** Built with Streamlit for an intuitive user experience.

## How to Use

1. **Clone the repository (if necessary):**
```bash
git clone https://github.com/Nicolakorff/PdfChatbotAssistant
cd PdfChatbotAssistant
```
2. **Install the dependencies:**
```bash
pip install -r requirements.txt
```
3. **Run the Streamlit application:**
```bash
streamlit run app.py
```
4. **Upload your PDF files:** In the left sidebar, you will find a section to upload your PDF documents. Click "Choose your PDF Files and Press OK" and select the files you want to process.
5. **Process the PDFs:** After selecting the files, click the "Process PDF" button. The application will process the PDFs and create a knowledge base. You will see a progress bar indicating the processing status.
6. **Start chatting:** Once the PDFs have been successfully processed, a text box will appear at the top where you can type your questions about the PDF content.
7. **View the conversation:** The user's questions and the chatbot's responses will be displayed in the main interface.

## Technologies Used

- **Streamlit:** For creating the interactive user interface.
- **Langchain:** Framework for developing language model-driven applications.
- **PyPDF2:** For reading and extracting text from PDF files.
- **Hugging Face Transformers:** For obtaining text embeddings (`all-MiniLM-L6-v2`) and interacting with language models (`gpt2`).
- **FAISS:** For efficient creation and searching of the vector knowledge base.
- **dotenv:** For managing environment variables (such as API keys, although this project does not use them directly in the current version).
- **Sentence Transformers:** (Although commented out in the current code, this could have been an alternative for embeddings.)

  ## Upcoming Improvements

- Implementation of different language models.
- Options to adjust text fragmentation parameters.
- Support for other file types (e.g., text documents).
- Better error handling and user feedback.
- Ability to save and load knowledge bases.

## Contributions

Contributions to this project are welcome. If you have ideas for improving the application or have found a bug, please feel free to open an issue or submit a pull request.

![Captura de pantalla 2025-04-25 a las 22 25 27](https://github.com/user-attachments/assets/f6932d85-d46f-4aba-9fdc-1d3f4c7e2359)
![Captura de pantalla 2025-04-25 a las 22 24 51](https://github.com/user-attachments/assets/e8289f79-fd32-41a4-b8f1-2ae6026709dc)
![Captura de pantalla 2025-04-26 a las 14 32 16](https://github.com/user-attachments/assets/3f61afb0-8919-42b2-bd3c-3b9222f1b2bb)

