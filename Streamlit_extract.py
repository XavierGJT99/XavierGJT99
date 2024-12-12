import streamlit as st
import os
from tika import parser
import fitz
import io
from PIL import Image
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import spacy
import re

# Load the SpaCy model
nlp = spacy.load("en_core_web_sm")

def summarize_text(text):
    # Load the T5 model and tokenizer
    model = T5ForConditionalGeneration.from_pretrained('t5-base')
    tokenizer = T5Tokenizer.from_pretrained('t5-base')

    # Prepare the input text
    input_text = "summarize: " + text
    inputs = tokenizer.encode_plus(input_text, 
                                    add_special_tokens=True, 
                                    max_length=1024, 
                                    return_attention_mask=True, 
                                    return_tensors='pt',
                                    truncation=True)  # Add truncation to handle long input texts

    # Generate the summary
    summary_ids = model.generate(inputs['input_ids'], 
                                 attention_mask=inputs['attention_mask'], 
                                 max_length=250,  
                                 min_length=50, 
                                 length_penalty=1.5,  
                                 early_stopping=True,
                                 num_beams=4)  # generate multiple summaries and select the best one
                                 

    # Decode the summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary

# Entity extractions
def extract_entities(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents if ent.label_ in ['PERSON']]
    return entities

# removes any repeated entity
def remove_duplicates(entities):
    unique_entities = list(set(entities))
    return unique_entities

# Search function for texts
def search_text(text, search_query):
    # Split the text into sentences
    sentences = re.findall(r"[^.!?]+", text)

    # Find the sentences where the search_query is found
    matching_sentences = [sentence for sentence in sentences if re.search(search_query, sentence, re.IGNORECASE)]

    if matching_sentences:
        st.write("Matches found:")
        for sentence in matching_sentences:
            st.write(sentence)
    else:
        st.write("No matches found")

# Image extraction
def extract_images(file_path):
    pdf_file = fitz.open(file_path)
    file_dir = os.path.dirname(file_path)

    image_list = []
            
    for page_number in range(len(pdf_file)): 
        page = pdf_file[page_number]
        for image_index, img in enumerate(page.get_images(), start=1):
            xref = img[0] 
            base_image = pdf_file.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            pil_image = Image.open(io.BytesIO(image_bytes))
            image_path = os.path.join(file_dir, f"image_{page_number}_{image_index}.{image_ext}")
            pil_image.save(image_path)
            image_list.append(image_path)

    return image_list

#combined app
def main():
    st.title("PDF Summarizer")

    uploaded_file = st.file_uploader("Select a PDF file", type=["pdf"])

    if uploaded_file:
        file_path = os.path.join(os.getcwd(), uploaded_file.name)

        # Extract the text from the PDF file using Tika
        parsed_data = parser.from_file(file_path)
        text = parsed_data['content'].strip()

        selected_function = st.selectbox("Select a function", ["Extract Text", "Summarize Text", "Extract Entities", "Extract Images", "Search PDF"])

        if selected_function == "Extract Text":
            st.header("Extracted Text:")
            st.text_area(label="", value=text, height=500)

        elif selected_function == "Summarize Text":
            summary = summarize_text(text)
            st.header("Summary:")
            st.text_area(label="", value=summary, height=200)

        elif selected_function == "Extract Entities":
            entities = extract_entities(text)
            unique_entities = remove_duplicates(entities)
            st.header("Extracted Entities:")
            for entity in unique_entities:
                st.write(f"Text: {entity[0]}, Label: {entity[1]}")

        elif selected_function == "Extract Images":
            image_list = extract_images(file_path)
            st.header("Extracted Images:")
            for image_path in image_list:
                st.image(image_path)

        elif selected_function == "Search PDF":
            search_query = st.text_input("Enter a word or phrase to search for:")
            url = "https://regex-generator-lyzr.streamlit.app/" 
            st.write(f"If you are unsure of how to search using Regex, click the [link]({url})")
            if search_query:
                search_text(text, search_query)

    else:
        st.write("No file selected")


if __name__ == "__main__":
    main()

# to run the streamlit app
# python -m streamlit run Streamlit_extract.py
