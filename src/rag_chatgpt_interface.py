import streamlit as st
import pandas as pd
import plotly.express as px
import json
import os
import yaml
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions
from datetime import datetime

# Load configuration from config.yml
def load_config():
    """Load configuration from config.yml."""
    config_path = os.path.join(os.path.dirname(__file__), "config.yml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

# Load config
config = load_config()

# Use config values
CHROMA_DB_PATH = config["chroma_db_path"]
OPENAI_API_KEY = config["openai_api_key"]
EMBEDDING_MODEL = config["embedding_model"]
GPT_MODEL = config["gpt_model"]
COLLECTION_NAMES = config["collections"]

# Initialize ChromaDB client with persistent storage
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

# Define embedding function (using OpenAI's text-embedding-ada-002)
embedding_fn = embedding_functions.OpenAIEmbeddingFunction(api_key=OPENAI_API_KEY, model_name=EMBEDDING_MODEL)

# Initialize the OpenAI client
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Function to get or create a collection
def get_or_create_collection(name, embedding_function=None):
    """Get an existing collection or create a new one if it doesn't exist."""
    try:
        collection = client.get_collection(name=name, embedding_function=embedding_function)
        print(f"Collection '{name}' already exists. Reusing it.")
    except Exception:
        print(f"Collection '{name}' does not exist. Creating it.")
        collection = client.create_collection(name=name, embedding_function=embedding_function)
    return collection

# Initialize collections
collections = {name: get_or_create_collection(name, embedding_fn) for name in COLLECTION_NAMES}

# Function to add data to ChromaDB collections
def add_data_to_collection(data, collection):
    """Add data to a ChromaDB collection."""
    for idx, row in data.iterrows():
        row_dict = row.to_dict()
        document = json.dumps(row_dict)
        metadata = {"id": str(row[0])}
        collection.add(documents=[document], metadatas=[metadata], ids=[str(row[0])])

# Function to upload Excel files and update collections
def upload_excel(file, collection_name):
    """Upload data from an Excel file and update a ChromaDB collection."""
    try:
        df = pd.read_excel(file)
        collection = collections.get(collection_name)
        if not collection:
            st.error(f"Invalid collection name: {collection_name}")
            return
        add_data_to_collection(df, collection)
        st.success(f"Data uploaded successfully to the '{collection_name}' collection.")
    except Exception as e:
        st.error(f"An error occurred while uploading the file: {str(e)}")

# Function to retrieve and generate responses
def retrieve_and_generate(prompt):
    """Retrieve relevant data from ChromaDB and generate a response using OpenAI's GPT."""
    try:
        retrieved_data = {
            name: collection.query(query_texts=[prompt], n_results=5)["documents"]
            for name, collection in collections.items()
        }
        response = openai_client.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Based on the following data: {retrieved_data}, answer the following question: {prompt}"}
            ],
            max_tokens=500
        )
        return response.choices[0].message.content.strip(), retrieved_data
    except Exception as e:
        return f"An error occurred: {str(e)}", None

# Function to generate visualization code or description using OpenAI
def generate_visualization_code(prompt, retrieved_data):
    """Use OpenAI to generate visualization code or description based on the prompt and retrieved data."""
    try:
        input_text = f"Based on the following data: {retrieved_data}, generate a visualization for the following prompt: {prompt}. Provide the code or description for the visualization."
        response = openai_client.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates visualization code or descriptions."},
                {"role": "user", "content": input_text}
            ],
            max_tokens=500
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"An error occurred while generating visualization: {str(e)}"

# Function to render the visualization based on the generated code or description
def render_visualization(visualization_code, retrieved_data):
    """Render the visualization based on the generated code or description."""
    try:
        if "plt." in visualization_code or "sns." in visualization_code:
            chart_type = None
            if "plt.pie" in visualization_code:
                chart_type = "pie"
            elif "plt.bar" in visualization_code:
                chart_type = "bar"
            elif "plt.plot" in visualization_code:
                chart_type = "line"
            elif "plt.hist" in visualization_code:
                chart_type = "histogram"
            elif "plt.scatter" in visualization_code:
                chart_type = "scatter"
            elif "sns.heatmap" in visualization_code or "plt.imshow" in visualization_code:
                chart_type = "heatmap"

            if chart_type == "heatmap":
                heatmap_data = []
                for product in retrieved_data["products"][0]:
                    try:
                        product_dict = json.loads(product) if isinstance(product, str) else product
                        if all(key in product_dict for key in ["State", "ProductCategory", "Volume"]):
                            heatmap_data.append({
                                "State": product_dict["State"],
                                "ProductCategory": product_dict["ProductCategory"],
                                "Volume": product_dict["Volume"]
                            })
                    except json.JSONDecodeError as e:
                        st.error(f"Error parsing JSON: {e}")
                        continue

                if heatmap_data:
                    df = pd.DataFrame(heatmap_data)
                    heatmap_df = df.pivot(index="State", columns="ProductCategory", values="Volume")
                    fig = px.imshow(heatmap_df, labels=dict(x="Product Category", y="State", color="Volume"),
                                    x=heatmap_df.columns, y=heatmap_df.index, text_auto=True, color_continuous_scale="YlGnBu")
                    fig.update_layout(title="Order Volume by State and Product Category", xaxis_title="Product Category", yaxis_title="State")
                    st.plotly_chart(fig)
                else:
                    st.write("No valid data available for heatmap.")
            else:
                categories = []
                attributes = []
                for product in retrieved_data["products"][0]:
                    try:
                        product_dict = json.loads(product) if isinstance(product, str) else product
                        if "ProductCategory" in product_dict:
                            categories.append(product_dict["ProductCategory"])
                        if all(key in product_dict for key in ["AttributeX", "AttributeY"]):
                            attributes.append((product_dict["AttributeX"], product_dict["AttributeY"]))
                    except json.JSONDecodeError as e:
                        st.error(f"Error parsing JSON: {e}")
                        continue

                if chart_type == "scatter" and attributes:
                    x_values = [attr[0] for attr in attributes]
                    y_values = [attr[1] for attr in attributes]
                    scatter_data = pd.DataFrame({"AttributeX": x_values, "AttributeY": y_values})
                    fig = px.scatter(scatter_data, x="AttributeX", y="AttributeY", title="Scatter Plot: Product Attributes")
                    st.plotly_chart(fig)
                elif categories:
                    category_counts = pd.Series(categories).value_counts().reset_index()
                    category_counts.columns = ["ProductCategory", "Count"]
                    if chart_type == "pie":
                        fig = px.pie(category_counts, values="Count", names="ProductCategory", title="Product Category Distribution")
                    elif chart_type == "bar":
                        fig = px.bar(category_counts, x="ProductCategory", y="Count", title="Product Category Distribution")
                    elif chart_type == "line":
                        fig = px.line(category_counts, x="ProductCategory", y="Count", title="Product Category Distribution")
                    elif chart_type == "histogram":
                        fig = px.histogram(category_counts, x="ProductCategory", y="Count", title="Product Category Distribution")
                    st.plotly_chart(fig)
                else:
                    st.write("No valid data available for visualization.")
        else:
            st.write("### Visualization Description")
            st.write("Visualization is available for line, pie, and bar chart.")
    except Exception as e:
        st.error(f"An error occurred while rendering the visualization: {str(e)}")

# Function to visualize data using OpenAI
def visualize_data(retrieved_data, prompt):
    """Dynamically visualize retrieved data using OpenAI and Plotly."""
    if not retrieved_data:
        st.write("No data available for visualization.")
        return
    visualization_code = generate_visualization_code(prompt, retrieved_data)
    render_visualization(visualization_code, retrieved_data)

# Streamlit UI
def main():
    st.title("Dynamic Visualization with OpenAI and Streamlit")
    st.write("Enter your prompt below to generate a visualization based on the retrieved data.")

    if "prompt_history" not in st.session_state:
        st.session_state.prompt_history = []

    with st.sidebar:
        st.header("Prompt History")
        for entry in st.session_state.prompt_history:
            with st.expander(f"{entry['prompt']}"):
                st.write(f"**Prompt:** {entry['prompt']}")
                st.write(f"**Response:** {entry['response']}")
                if st.button(f"Show Visualization for: {entry['prompt']}"):
                    st.session_state.selected_prompt = entry['prompt']
                    st.session_state.selected_retrieved_data = entry['retrieved_data']
                st.write("---")
        if st.button("Clear Prompt History"):
            st.session_state.prompt_history = []
            st.write("Prompt history cleared.")

    tab1, tab2 = st.tabs(["Chat", "Upload Data"])

    with tab1:
        prompt = st.text_input("Enter your prompt:")
        if prompt:
            response, retrieved_data = retrieve_and_generate(prompt)
            st.session_state.prompt_history.append({
                "prompt": prompt,
                "response": response,
                "retrieved_data": retrieved_data,
                "timestamp": datetime.now()
            })
            st.write("### Response:")
            st.write(response)
            st.write("### Visualizations")
            visualize_data(retrieved_data, prompt)

    with tab2:
        st.header("Upload Excel Files")
        uploaded_files = {
            "products": st.file_uploader("Upload Products Excel", type=["xlsx"]),
            "distributors": st.file_uploader("Upload Distributors Excel", type=["xlsx"]),
            "geo": st.file_uploader("Upload Geo Excel", type=["xlsx"]),
            "orders": st.file_uploader("Upload Orders Excel", type=["xlsx"])
        }
        for collection_name, uploaded_file in uploaded_files.items():
            if uploaded_file:
                upload_excel(uploaded_file, collection_name)

    if "selected_prompt" in st.session_state:
        st.write("### Selected Prompt:")
        st.write(st.session_state.selected_prompt)
        st.write("### Response:")
        selected_entry = next(entry for entry in st.session_state.prompt_history if entry['prompt'] == st.session_state.selected_prompt)
        st.write(selected_entry['response'])
        st.write("### Visualizations")
        visualize_data(st.session_state.selected_retrieved_data, st.session_state.selected_prompt)

if __name__ == "__main__":
    main()