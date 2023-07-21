# SMA-Categorization
Simple Flask application which automatically finds Manuals in database based on the serial number of the device. It is also connected to GPT3.5 to provide summary for a given problem description.

### Directories

- `app` - contains the Flask application
- `Database\manual` - contains the manuals to read by the application
- `templates` - contains the html templates

To run application, install the requirements and run `python app.py` in 
the `app` directory. Provide your OpenAI api key`openai.api_key = `

Then open http://127.0.0.1:5000 or other local server 
given in terminal. 

### Interface

User interface is divided into two separate blocks:
- Service console - technician can enter the serial number of the device and the problem description
- Results - the application will display the results of the search: manual for a given device, similar documents, page where the answer for the problem might be.
- Step by step button - the application will display the summary of the problem description based on the GPT model.


### CUDA installation
- Installation Docs: https://linuxhint.com/install-cuda-on-ubuntu-22-04-lts/
- The program will print if GPU is available