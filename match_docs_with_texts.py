import os
import re
import glob
import shutil
import docx
import pdfplumber
from pdf2image import convert_from_path
from collections import Counter
from win32com.client import Dispatch
import time

# Путь к папке с документами
docs_path = "docs"  # Укажи свою папку
text_files_path = "data_with_docs"  # Укажи папку с fileN.txt


# Функция для конвертации .doc в .docx (перезапускаем Word для каждого файла)
def convert_doc_to_docx(doc_path):
    try:
        word = Dispatch("Word.Application")
        word.Visible = False
        doc = word.Documents.Open(os.path.abspath(doc_path))
        new_path = doc_path + "x"
        doc.SaveAs(new_path, FileFormat=16)
        doc.Close()
        word.Quit()
        time.sleep(1)  # Даем системе обработать закрытие процесса
        return new_path
    except Exception as e:
        print(f"Ошибка конвертации {doc_path}: {e}")
        return None


# Функция для извлечения текста из DOCX
def extract_text_from_docx(filepath):
    doc = docx.Document(filepath)
    return " ".join([p.text for p in doc.paragraphs])


# Функция для извлечения текста из PDF
def extract_text_from_pdf(filepath):
    text = ""
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + " "
    return text.strip()



# Функция для предобработки текста
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^а-яa-z0-9]+', ' ', text)
    return set(text.split())


# Читаем все fileN.txt
txt_files = glob.glob(os.path.join(text_files_path, "file*.txt"))
txt_data = {}

for txt_file in txt_files:
    with open(txt_file, "r", encoding="utf-8") as f:
        text = f.read()
    txt_data[txt_file] = preprocess_text(text)

# Обрабатываем документы
doc_files = glob.glob(os.path.join(docs_path, "*.docx")) + \
            glob.glob(os.path.join(docs_path, "*.doc")) + \
            glob.glob(os.path.join(docs_path, "*.pdf"))

unprocessed_files = []

for doc_file in doc_files:
    if doc_file.endswith(".doc"):
        converted_path = convert_doc_to_docx(doc_file)
        if not converted_path:
            unprocessed_files.append(doc_file)
            continue
        doc_file = converted_path

    if doc_file.endswith(".docx"):
        doc_text = extract_text_from_docx(doc_file)
    elif doc_file.endswith(".pdf"):
        doc_text = extract_text_from_pdf(doc_file)
    else:
        continue

    doc_words = preprocess_text(doc_text)

    # Поиск наилучшего совпадения
    best_match = max(txt_data.items(), key=lambda x: len(doc_words & x[1]), default=(None, set()))
    best_txt_file, best_match_words = best_match

    if best_txt_file:
        new_name = f"{os.path.basename(best_txt_file).replace('.txt', '')}{os.path.splitext(doc_file)[-1]}"
        new_path = os.path.join(docs_path, new_name)
        shutil.move(doc_file, new_path)
        print(f"Переименован: {doc_file} -> {new_name}")
    else:
        unprocessed_files.append(doc_file)

print("Необработанные файлы:", unprocessed_files)
