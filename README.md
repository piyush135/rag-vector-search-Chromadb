Dynamic Visualization with OpenAI and Streamlit
This repository contains a Streamlit application that integrates OpenAI's GPT and ChromaDB to dynamically visualize data based on user prompts. The application allows users to upload Excel files, store data in ChromaDB collections, and generate insights or visualizations using OpenAI's GPT models.

Key Features:
Dynamic Data Visualization: Generate visualizations (e.g., bar charts, pie charts, scatter plots, heatmaps) based on user queries and retrieved data.

ChromaDB Integration: Store and query data in ChromaDB collections for efficient retrieval.

OpenAI GPT Integration: Use OpenAI's GPT models to generate responses and visualization code dynamically.

Streamlit UI: User-friendly interface for uploading data, querying, and viewing results.

Configurable Settings: Manage API keys, paths, and model configurations via a config.yml file.

Technologies Used:
Streamlit: For building the web application interface.

OpenAI API: For generating responses and visualization code.

ChromaDB: For storing and querying data embeddings.

Plotly: For rendering interactive visualizations.

Pandas: For handling Excel file uploads and data manipulation.

YAML: For managing configuration settings.

How It Works:
Users upload Excel files containing data (e.g., products, distributors, orders).

Data is stored in ChromaDB collections with embeddings for efficient querying.

Users enter prompts to retrieve relevant data and generate insights or visualizations.

OpenAI's GPT models generate responses and visualization code, which are rendered using Plotly.

Setup:
Clone the repository.

Install dependencies: pip install -r requirements.txt.

Add your OpenAI API key and other settings in config.yml.

Run the Streamlit app: streamlit run app.py.

Use Cases:
Data Exploration: Quickly explore and visualize datasets.

Business Intelligence: Generate insights from business data (e.g., sales, orders, inventory).

Interactive Dashboards: Build dynamic dashboards for real-time data analysis.

Repository Structure:
Copy
dynamic-visualization-app/
├── app.py                  # Main Streamlit application
├── config.yml              # Configuration file for API keys and settings
├── requirements.txt        # List of Python dependencies
├── chroma_db/              # Directory for ChromaDB persistent storage
└── README.md               # Repository documentation
Requirements:
Python 3.8+

OpenAI API key

Streamlit, ChromaDB, Plotly, Pandas, PyYAML