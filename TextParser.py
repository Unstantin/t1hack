import PyPDF2
from docx import Document
import pandas as pd
import os


class TextParser:
    def __init__(self):
        pass

    def parse_pdf(self, file_path):
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ''
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text()
            return text

    def parse_word(self, file_path):
        doc = Document(file_path)
        text = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
        return text

    def parse_txt(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        return text

    def parse_csv(self, file_path):
        df = pd.read_csv(file_path)
        text = df.to_string(index=False)
        return text

    def parse_md(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        return text

    def parse_file(self, file_path):
        _, ext = os.path.splitext(file_path)
        print(os.path.splitext(file_path))
        if ext.lower() == '.pdf':
            return self.parse_pdf(file_path)
        elif ext.lower() == '.docx':
            return self.parse_word(file_path)
        elif ext.lower() == '.txt':
            return self.parse_txt(file_path)
        elif ext.lower() == '.csv':
            return self.parse_csv(file_path)
        elif ext.lower() == '.md':
            return self.parse_md(file_path)
        else:
            return None