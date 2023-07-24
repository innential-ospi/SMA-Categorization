from flask import Flask, send_file, render_template, request, jsonify, Response, session
from flask_session import Session  # Import Session
from sentence_transformers import SentenceTransformer, util
from werkzeug.security import generate_password_hash, check_password_hash
from flask_httpauth import HTTPBasicAuth
from env import OPENAI_KEY, LOGIN, PASS
import openai
import os
import fitz
import time
import torch






if torch.cuda.is_available():
    print('CUDA is available')
    device = torch.device('cuda')
else:
    print("CUDA is not available, using CPU instead")
    device = torch.device('cpu')

# Set up SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2',device = device)

# Set up OpenAI API credentials
openai.api_key = OPENAI_KEY

def chat(message):
    # Use OpenAI Chat API to generate a response
    # The structure is different as with davinci-003
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        temperature = 0,
        max_tokens = 400,
        top_p=0.6,
        frequency_penalty=0.5,
        presence_penalty=0,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": message},
        ]
    )
    return response['choices'][0]['message']['content'].strip()

def find_path_to_database():
    # Get the path to the database directory
    path = os.path.join(os.getcwd(), 'Database', 'Manual')
    return path


def get_name_of_files():
    # Get the names of all files in the database directory
    path = find_path_to_database()
    files = os.listdir(path)
    return files


def get_text_from_file(file_path):
    # Read the PDF file and extract text
    pdf_file = fitz.open(file_path)
    number_pages = len(pdf_file)

    return pdf_file, number_pages

def find_solution_in_page(path, description):
    # Find the page with the solution in the PDF file
    document, pages_number = get_text_from_file(path)

    # Encode all sentences
    embeddings_problem_description = model.encode(description)

    highest_sim = []

    pages = [document.load_page(i).get_text("text") for i in range(3, pages_number)]
    start = time.time()
    page_embeddings = model.encode(pages)
    cos_sim = util.cos_sim(page_embeddings, embeddings_problem_description)
    end = time.time()
    print("Encode time: ", end - start)
    page_with_solution = sorted(enumerate(cos_sim), key=lambda x: x[1], reverse=True)[0][0] + 3

    print(page_with_solution)
    # Check if the solution is on one page only
    #if page_with_solution[0][1] + 1 != page_with_solution[1][1]:
    #    one_page_only = 1
    #else:
    #    one_page_only = 0
    #
    return page_with_solution+1, 1
def find_similarities(manuals, sensor):
    # Find similarities between manuals and a given sensor

    # Encode all sentences
    embeddings_manuals = model.encode(manuals)
    embeddings_sensor = model.encode(sensor)

    # Compute cosine similarity between all pairs
    cos_sim = util.cos_sim(embeddings_manuals, embeddings_sensor)

    # Add all pairs to a list with their cosine similarity score
    similar_docs = []
    for i in range(len(cos_sim)):
        similar_docs.append([cos_sim[i][0], i, 0])

    # Sort list by the highest cosine similarity score
    similar_docs = sorted(similar_docs, key=lambda x: x[0], reverse=True)

    sorted_manuals = []
    for score, i, j in similar_docs:
        if score > 0.30:
            sorted_manuals.append(manuals[i])

    return similar_docs, sorted_manuals

def ask_gpt(description, path, page, one_page):
    # Use GPT-3.5 to generate step-by-step instructions
    document, pages_number = get_text_from_file(path)
    language = "Englosh"
    user_input = "Extract answer for this problem (point by point without changing anything, translate to" + language +"): " + description + ", from text:"

    if one_page == 1:
        for i in range(page - 1, page):
            page_obj = document.load_page(i)
            page_text = page_obj.get_text("text")
            user_input += ". " + page_text
    else:
        for i in range(page, page + 2):
            page_obj = document.load_page(i)
            page_text = page_obj.get_text("text")
            user_input += ". " + page_text

    start = time.time()
    response = chat(user_input)
    end = time.time()

    print("GPT response time: ", end - start)

    return response


def gpt_step_by_step(text, device):
    description_global = text
    filename_global = "STP50-4x-BE-en-23.pdf"

    print("Des: ",description_global)

    if description_global:
        description = description_global
        filename = filename_global

        print("GPT 3.5 step by step")
        print("Problem description:",description,"Filename:", filename)

        path = find_path_to_database() + '/' + filename

        solution, one_page = find_solution_in_page(path, description)
        response = ask_gpt(description, path, solution, one_page)
    else:
        response = "No code description found"

    print(response)

#Request to get the pdf file

def open_pdf(filename):
    file_path = find_path_to_database() + '/' + filename
    print("Path to pdf manual:", file_path)
    return send_file(file_path, as_attachment=True)


def index():
    return render_template("index.html")


def search(text, device):
    start = time.time()
    # Handle the POST request for searching manuals

    model = device

    print("Language:", "English")
    description = text

    # Check if the model exists in the manuals database
    manuals = get_name_of_files()
    documents, sorted_manuals = find_similarities(manuals, model)

    print("Similar documents: ", sorted_manuals)

    path = find_path_to_database() + '/' + sorted_manuals[0]
    print(path)
    solution = "Description not found"
    step_by_step = "Description not found"

    # Check if the description exists
    if description:
        solution, one_page = find_solution_in_page(path, description)
        solution = "Page " + str(solution)

    exact_file = sorted_manuals[0]
    similar_files = sorted_manuals[1:]



    end = time.time()

    print("Total time: ", end - start)

text = "i want to disconnect inverter form voltage source"

device = "spt50 4x be"

search(text, device)

gpt_step_by_step(text, device)

