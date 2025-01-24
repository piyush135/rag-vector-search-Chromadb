# **Dynamic Visualization with OpenAI and Streamlit**

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=OpenAI&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-000000?style=for-the-badge&logo=ChromaDB&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=Plotly&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=Pandas&logoColor=white)

This repository contains a **Streamlit-based web application** that integrates **OpenAI's GPT** and **ChromaDB** to dynamically visualize data based on user prompts. The application allows users to upload Excel files, store data in ChromaDB collections, and generate insights or visualizations using OpenAI's GPT models.

---

## **Features**

- **Dynamic Data Visualization**:
  - Generate visualizations (e.g., bar charts, pie charts, scatter plots, heatmaps) based on user queries and retrieved data.
  - Visualizations are rendered using **Plotly** for interactivity.

- **ChromaDB Integration**:
  - Store and query data in ChromaDB collections with embeddings for efficient retrieval.
  - Supports multiple collections (e.g., products, distributors, orders).

- **OpenAI GPT Integration**:
  - Use OpenAI's GPT models to generate responses and visualization code dynamically.
  - Supports **GPT-3.5-turbo** for natural language processing and code generation.

- **Streamlit UI**:
  - User-friendly interface for uploading data, querying, and viewing results.
  - Includes a **Prompt History** section to track previous queries and responses.

- **Configurable Settings**:
  - Manage API keys, paths, and model configurations via a `config.yml` file.

---

## **Technologies Used**

- **Streamlit**: For building the web application interface.
- **OpenAI API**: For generating responses and visualization code.
- **ChromaDB**: For storing and querying data embeddings.
- **Plotly**: For rendering interactive visualizations.
- **Pandas**: For handling Excel file uploads and data manipulation.
- **YAML**: For managing configuration settings.

---

## **How It Works**

1. **Data Upload**:
   - Users upload Excel files containing data (e.g., products, distributors, orders).
   - Data is stored in ChromaDB collections with embeddings for efficient querying.

2. **Query and Retrieve**:
   - Users enter prompts to retrieve relevant data from ChromaDB collections.
   - The application queries the collections and retrieves the most relevant records.

3. **Generate Insights**:
   - OpenAI's GPT models generate responses based on the retrieved data and user prompts.
   - The application also generates visualization code or descriptions using OpenAI.

4. **Render Visualizations**:
   - The generated visualization code is executed to render interactive charts using Plotly.

---

## **Setup**

### **Prerequisites**

- Python 3.8+
- OpenAI API key
- Streamlit, ChromaDB, Plotly, Pandas, PyYAML

### **Installation**

1. Clone the repository:
   ```bash
   git clone https://github.com/piyush135/rag-vector-search-Chromadb.git
   cd dynamic-visualization-app
Install dependencies:

bash
Copy
pip install -r requirements.txt
Configure the application:

Open the config.yml file and add your OpenAI API key:

yaml
Copy
openai_api_key: "your_openai_api_key_here"
Modify other settings (e.g., ChromaDB path, model names) as needed.

Run the Streamlit app:

bash
Copy
streamlit run app.py
Open your browser and navigate to http://localhost:8501 to access the application.

Usage
Upload Data:

Navigate to the Upload Data tab.

Upload Excel files for products, distributors, geo, or orders.

Query Data:

Navigate to the Chat tab.

Enter a prompt (e.g., "Show me the distribution of product categories").

View the generated response and visualization.

View History:

Use the Prompt History sidebar to view previous queries and responses.

Click on a prompt to regenerate its visualization.

Examples
Example 1: Upload Data
Upload an Excel file containing product data.

The data is stored in the products collection in ChromaDB.

Example 2: Generate Visualization
Enter a prompt: "Show me a pie chart of product categories."

The application retrieves relevant data, generates a response, and renders a pie chart using Plotly.

Example 3: Query History
View previous prompts and responses in the Prompt History sidebar.

Click on a prompt to regenerate its visualization.


   
