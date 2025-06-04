import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from flask_session import Session
from flask import Flask, render_template, request, session, redirect, url_for
from datetime import timedelta
import time # Import time for potential retry logic

application = Flask(__name__)

application.config['SESSION_TYPE'] = 'filesystem'
application.config['SESSION_FILE_DIR'] = './.flask_session/'
application.config['SESSION_PERMANENT'] = False
Session(application)

FAISS_INDEX_PATH = "endometriosis_faiss.index"
CHUNKS_DATA_PATH = "endometriosis_chunks_with_ids.json"
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise EnvironmentError(
        "GOOGLE_API_KEY environment variable not set. Please set it before running."
    )

genai.configure(api_key=GOOGLE_API_KEY)
GEMINI_LLM_MODEL = 'gemini-2.5-flash-preview-05-20'

embedding_model = None
faiss_index = None
chunks_data = None
id_to_chunk_map = None


def load_rag_components():
    global embedding_model, faiss_index, chunks_data, id_to_chunk_map

    if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(CHUNKS_DATA_PATH):
        print("Error: FAISS index or chunks data not found. Please run 'create_vector_db.py' first.")
        return False

    try:
        print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print("Embedding model loaded.")

        print(f"Loading FAISS index from {FAISS_INDEX_PATH}...")
        faiss_index = faiss.read_index(FAISS_INDEX_PATH)
        print(f"FAISS index loaded. Contains {faiss_index.ntotal} embeddings.")

        print(f"Loading chunks data from {CHUNKS_DATA_PATH}...")
        with open(CHUNKS_DATA_PATH, "r", encoding="utf-8") as f:
            chunks_data = json.load(f)
        print(f"Loaded {len(chunks_data)} chunks.")

        id_to_chunk_map = {chunk['id']: chunk for chunk in chunks_data}

        return True
    except Exception as e:
        print(f"Error loading RAG components: {e}")
        return False


def retrieve_and_generate(query, conversation_history_raw, top_k=3):
    if embedding_model is None or faiss_index is None or chunks_data is None:
        return "RAG components not loaded. Please restart the application.", [], conversation_history_raw

    # Convert history from session format to API-compatible format
    api_compatible_history = []
    for turn in conversation_history_raw:
        # Crucial: For user turns, only add the original query, not the full prompt.
        # For model turns, add its parts.
        if turn['role'] == 'user':
            api_compatible_history.append({'role': 'user', 'parts': [{'text': turn['parts'][0]}]})
        else: # role is 'model'
            parts = []
            for part_text in turn.get('parts', []):
                parts.append({'text': part_text})
            api_compatible_history.append({'role': turn['role'], 'parts': parts})

    query_embedding = embedding_model.encode([query]).astype('float32')

    D, I = faiss_index.search(query_embedding, top_k)

    retrieved_chunks = []
    if I.size > 0:
        for idx in I[0]:
            if 0 <= idx < len(chunks_data):
                retrieved_chunks.append(chunks_data[idx])
            else:
                print(f"Warning: Invalid index {idx} returned from FAISS. Skipping.")
    else:
        print("No relevant chunks found.")

    context = "\n\n".join(
        [f"Source PMID: {c['source']['pmid']}\nTitle: {c['source']['title']}\nText: {c['text']}"
         for c in retrieved_chunks]
    )

    if not context:
        final_prompt_for_llm_call = f"No relevant information found in the knowledge base for the query: '{query}'. Please try rephrasing or ask a different question."
        print("\n--- LLM Input (No Context) ---")
        print(final_prompt_for_llm_call)
    else:
        final_prompt_for_llm_call = f"""
        You are an expert in endometriosis research. Answer the following question, primarily referencing the provided context.
        **Do NOT directly quote large blocks of text from the context. Summarize and synthesize the information in your own words.**
        If the context is insufficient, you may draw upon your general knowledge to provide a comprehensive answer, but clearly distinguish between information from the context and your general expertise (e.g., by starting a general knowledge section with "Beyond the provided sources,..." or similar).
        You can also suggest potential hypotheses or avenues for further research if appropriate.

        --- Context ---
        {context}

        --- Question ---
        {query}

        --- Answer ---
        """
        print("\n--- LLM Input (with Context) ---")
        print(final_prompt_for_llm_call)

    try:
        model = genai.GenerativeModel(GEMINI_LLM_MODEL)
        chat_session = model.start_chat(history=api_compatible_history)
        
        # Add retry logic for 429 errors
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = chat_session.send_message(final_prompt_for_llm_call)
                break # If successful, break out of the retry loop
            except Exception as e:
                error_message_str = str(e)
                if "429" in error_message_str:
                    delay_seconds = 25 # Default delay, or parse from error. For testing, you might use a lower value like 5
                    print(f"Rate limit hit. Retrying in {delay_seconds} seconds (Attempt {attempt + 1}/{max_retries})...")
                    time.sleep(delay_seconds)
                else:
                    # Re-raise other errors immediately
                    raise e
        else: # This 'else' block executes if the loop completes without a 'break' (i.e., all retries failed)
            return "Exceeded maximum retries due to rate limits. Please try again later.", [], conversation_history_raw
        
        # --- End of retry logic ---

        # The history from chat_session already includes the *full* LLM prompt/response structure.
        # We need to reconstruct it to only store the user's original query and the model's clean response.
        updated_history_for_session = []
        # Add the user's original query first for this turn
        updated_history_for_session.append({'role': 'user', 'parts': [query]}) # Store only the original query

        # Then add the model's response for this turn
        model_response_parts = []
        for part in response.parts:
            model_response_parts.append(part.text if hasattr(part, 'text') else str(part))
        updated_history_for_session.append({'role': 'model', 'parts': model_response_parts})


        print("\n--- Raw LLM Response ---")
        try:
            print(response.text)
        except Exception as e:
            print(f"Couldn't print response text directly: {e}")
        
        # Extract response text safely for returning
        response_text = ""
        try:
            response_text = response.text
        except AttributeError:
            try:
                response_text = "\n".join([p.text if hasattr(p, "text") else str(p) for p in response.parts])
            except Exception as e:
                print(f"Error parsing response parts: {e}")
                response_text = str(response)
        
        return response_text, retrieved_chunks, updated_history_for_session


    except Exception as e:
        print(f"Error calling Gemini LLM: {e}")
        error_message = str(e)
        if "404" in error_message and "models/" in error_message:
            error_message += f". Check if '{GEMINI_LLM_MODEL}' is the correct model name for your API key and region."
        print(f"Ensure your GOOGLE_API_KEY is correctly set as an environment variable and you have network access. Error details: {error_message}")
        return f"An error occurred while generating the response: {error_message}", [], conversation_history_raw


@application.route('/', methods=['GET', 'POST'])
def index():
    answer = ""
    sources_info = []
    user_query = ""
    # Retrieve current history for display
    conversation_history = session.get('conversation_history', [])

    if request.method == 'POST':
        user_query = request.form.get('query', '').strip()
        if user_query:
            # Pass the *current* conversation history to the RAG function
            answer, sources, new_turn_history = retrieve_and_generate(user_query, conversation_history)

            # Manually append the new user query and model response to the history for the session
            # This is crucial for keeping the history clean
            conversation_history.append({'role': 'user', 'parts': [user_query]}) # Append only the user's actual query
            if new_turn_history and new_turn_history[-1]['role'] == 'model': # Ensure a model response exists
                conversation_history.append(new_turn_history[-1]) # Append the clean model response

            session['conversation_history'] = conversation_history # Save updated history back to session

            if sources:
                for src in sources:
                    sources_info.append({
                        'pmid': src['source']['pmid'],
                        'title': src['source']['title'],
                        'chunk_id': src['id']
                    })
            else:
                sources_info = ["(No specific sources were retrieved for this query.)"]
        else:
            answer = "Please enter a question."

    return render_template('index.html', answer=answer, sources=sources_info, query=user_query, history=conversation_history)


@application.route('/new_chat')
def new_chat():
    session.pop('conversation_history', None)
    return redirect(url_for('index'))


if __name__ == '__main__':
    print("Starting Flask application...")
    if load_rag_components():
        print("\n--- Endometriosis Robot Librarian Web App Ready! ---")
        print(f"Using Google Gemini LLM: {GEMINI_LLM_MODEL}. Using Embedding Model: {EMBEDDING_MODEL_NAME}.")
        print("Open your web browser and go to: http://127.0.0.1:5000/")
        application.run(debug=False)
    else:
        print("Failed to load RAG components. Exiting web application.")