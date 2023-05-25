from sentence_transformers import SentenceTransformer, util
import PyPDF2
import time
import fitz

# Set up SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

def get_text_from_file(file_path):

    # Read the PDF file and extract text

    pdfFileObj = open(file_path, 'rb')
    pdfReader = PyPDF2.PdfReader(pdfFileObj)
    number_pages = len(pdfReader.pages)
    pageObj = pdfReader.pages[0]

    return pdfReader, number_pages

def find_solution_in_page(path, description):
    # Find the page with the solution in the PDF file
    document, pages_number = get_text_from_file(path)

    # Encode all sentences
    embeddings_problem_description = model.encode(description)

    highest_sim = []
    for i in range(3, pages_number):
        # Extracting text from page
        page = document.pages[i].extract_text()

        # Encode all sentences from page
        page = model.encode(page)
        cos_sim_2 = util.cos_sim(page, embeddings_problem_description)
        highest_sim.append([cos_sim_2[0][0], i])

    page_with_solution = sorted(highest_sim, key=lambda x: x[0], reverse=True)

    # Check if the solution is on one page only
    if page_with_solution[0][1] + 1 != page_with_solution[1][1]:
        one_page_only = 1
    else:
        one_page_only = 0

    return page_with_solution[0][1] + 1, one_page_only

def get_text_from_file_2(file_path):
    # Read the PDF file and extract text
    pdf_file = fitz.open(file_path)
    number_pages = len(pdf_file)

    return pdf_file, number_pages

def find_solution_in_page_2(path, description):
    # Find the page with the solution in the PDF file
    document, pages_number = get_text_from_file_2(path)

    # Encode all sentences
    embeddings_problem_description = model.encode(description)

    highest_sim = []
    for i in range(3, pages_number):
        # Extracting text from page
        page = document.load_page(i)
        page_text = page.get_text("text")

        # Encode all sentences from page
        page = model.encode(page_text)
        cos_sim_2 = util.cos_sim(page, embeddings_problem_description)
        highest_sim.append([cos_sim_2[0][0], i])

    page_with_solution = sorted(highest_sim, key=lambda x: x[0], reverse=True)

    # Check if the solution is on one page only
    if page_with_solution[0][1] + 1 != page_with_solution[1][1]:
        one_page_only = 1
    else:
        one_page_only = 0

    return page_with_solution[0][1] + 1, one_page_only

path = "/home/piotr/PycharmProjects/Innential/SMA-Categorization/Website/Database/Manual/STP50-ACRLY-AST-RM-xx-10.pdf"
description = "how to disconnect inverter from voltage source"

start_time = time.time()
get_text_from_file(path)
elapsed_time = time.time() - start_time

# Print the elapsed time
print("PyPDF2:", elapsed_time, "seconds")

start_time = time.time()
find_solution_in_page(path, description)
elapsed_time = time.time() - start_time

# Print the elapsed time
print("PyPDF2 find solution:", elapsed_time, "seconds")

#############################################################################################################
#PyMuPDF

start_time = time.time()
get_text_from_file_2(path)
elapsed_time = time.time() - start_time

# Print the elapsed time
print("PyMuPDF2:", elapsed_time, "seconds")

start_time = time.time()
find_solution_in_page_2(path, description)
elapsed_time = time.time() - start_time

# Print the elapsed time
print("PyMuPDF2 find solution:", elapsed_time, "seconds")