from flask import Flask, send_file, render_template, request, jsonify, send_from_directory
from sentence_transformers import SentenceTransformer, util
import openai
import os

# Initialize Flask app
app = Flask(__name__)

# Set up SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Set up OpenAI API credentials
openai.api_key = ''


def chat(message):
    # Use OpenAI Chat API to generate a response
    response = openai.Completion.create(
        engine='text-davinci-003',
        prompt=message,
        max_tokens=500,
        n=1,
        stop=None,
        temperature=0.4,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )
    return response.choices[0].text.strip()


def find_path_to_database():
    # Get the path to the database directory
    path = os.getcwd()
    path = path.replace('\\', '/')
    path = path.replace('/SMA_Innential', '')
    path = path + '/Database/Manual'
    return path


def get_name_of_files():
    # Get the names of all files in the database directory
    path = find_path_to_database()
    files = os.listdir(path)
    return files


def get_text_from_file(file_path):
    # Read the PDF file and extract text
    import PyPDF2

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
        if score > 0.3:
            sorted_manuals.append(manuals[i])

    return similar_docs, sorted_manuals

def ask_gpt(description, path, page, one_page):
    # Use GPT-3.5 to generate step-by-step instructions
    document, pages_number = get_text_from_file(path)
    user_input = "Extract answer for this problem (point by point): " + description + ", from text:"

    if one_page == 1:
        for i in range(page - 1, page):
            page = document.pages[i].extract_text()
            user_input = user_input + ". " + page
    else:
        for i in range(page, page + 2):
            page = document.pages[i].extract_text()
            user_input = user_input + ". " + page

    response = chat(user_input)
    return response

@app.route("/gpt-step-by-step", methods=["POST"])
def gpt_step_by_step():
    global description_global
    global filename_global

    description = description_global
    filename = filename_global

    print("GPT 3.5 step by step")
    print("Problem describtion:",description,"Filename:", filename)

    path = find_path_to_database() + '/' + filename

    solution, one_page = find_solution_in_page(path, description)
    response = ask_gpt(description, path, solution, one_page)

    return jsonify(response=response)

#Request to get the pdf file
@app.route("/open-pdf/<filename>")
def open_pdf(filename):
    file_path = find_path_to_database() + '/' + filename
    print("Path to pdf manual:", file_path)
    return send_file(file_path, as_attachment=True)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/search", methods=["POST"])
def search():
    global description_global
    global filename_global

    # Handle the POST request for searching manuals
    data = request.get_json()
    model = data.get("serialNumber")
    error = data.get("eventCode")
    description = data.get("eventDescription")

    # Check if the model exists in the manuals database
    manuals = get_name_of_files()
    documents, sorted_manuals = find_similarities(manuals, model)

    print("Similar documents: ", sorted_manuals)

    path = find_path_to_database() + '/' + sorted_manuals[0]
    solution = "No code describtion found"
    step_by_step = "No describtion found"

    # Check if the description exists
    if description:
        solution, one_page = find_solution_in_page(path, description)

    exact_file = sorted_manuals[0]
    similar_files = sorted_manuals[1:]

    # Set global variables
    description_global = description
    filename_global = exact_file

    response = {
        "exact_file": exact_file,
        "similar_files": similar_files, # Get the first 5 similar files
        "path": path,
        "solution": solution,
        "step_by_step": step_by_step,

    }
    return jsonify(response)

if __name__ == "__main__":
    app.run()