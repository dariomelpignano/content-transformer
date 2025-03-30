import os
import gradio as gr
from dotenv import load_dotenv
import openai
from anthropic import Anthropic
import PyPDF2
import google.generativeai as genai
import speech_recognition as sr
from pydub import AudioSegment
from typing import Optional, Tuple

# Load environment variables
load_dotenv()

# Constants
SUPPORTED_FILE_TYPES = [".pdf", ".txt", ".md"]
TEMP_AUDIO_FILE = "temp_audio.wav"
ERROR_MESSAGES = {
    "no_context": "⚠️ Per favore, carica e processa prima almeno un file di contesto.",
    "no_conversation": "⚠️ Per favore, inserisci del testo nella cronologia conversazione.",
    "unsupported_file": "Formato file non supportato. Si prega di caricare un file PDF, TXT o MD.",
    "transcription_error": "Errore nella trascrizione: {}",
    "api_error": "❌ {}",
    "no_file": "Nessun file caricato.",
    "success": "✅ Contesto caricato e processato correttamente",
    "gemini_not_available": "⚠️ Gemini non disponibile. Inserisci una Google API Key nel file .env per utilizzarlo.",
    "api_keys_updated": "Chiavi API aggiornate con successo!"
}

# Configure API keys
GEMINI_AVAILABLE = bool(os.getenv("GOOGLE_API_KEY"))
OPENAI_AVAILABLE = bool(os.getenv("OPENAI_API_KEY"))
ANTHROPIC_AVAILABLE = bool(os.getenv("ANTHROPIC_API_KEY"))

if GEMINI_AVAILABLE:
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
if OPENAI_AVAILABLE:
    openai.api_key = os.getenv("OPENAI_API_KEY")

# Global variables
context_text = ""

# Prompt template
PROMPT_TEMPLATE = """Contesto del documento:
{context}

Contenuto della cronologia conversazione:
{conversation}

Istruzioni per il suggerimento:
1. Analizza il contesto del documento per trovare informazioni rilevanti sul prodotto/servizio
2. Usa le informazioni dalla cronologia conversazione per capire il contesto della vendita
3. Formula un suggerimento che:
   - Evidenzia i punti di forza del prodotto/servizio menzionati nel contesto
   - Si allinea con il tono e il focus della conversazione
   - È specifico e pertinente al contesto
4. Se il contesto contiene informazioni specifiche, usale per rendere il suggerimento più convincente
5. Mantieni il suggerimento breve e diretto

Fornisci UNA frase significativa e persuasiva da dire al cliente, basata sulle informazioni disponibili nel contesto e nella cronologia conversazione."""

def create_prompt(context: str, conversation: str) -> str:
    """Create a formatted prompt with the given context and conversation."""
    return PROMPT_TEMPLATE.format(context=context, conversation=conversation)

def get_claude_response(prompt: str) -> str:
    """Get a response from Claude model."""
    try:
        client = Anthropic()
        message = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=150,
            temperature=0.3,
            messages=[{
                "role": "user",
                "content": prompt
            }]
        )
        return message.content
    except Exception as e:
        raise Exception(f"Errore nella chiamata a Claude: {str(e)}")

def get_openai_response(prompt: str) -> str:
    """Get a response from OpenAI model."""
    try:
        response = openai.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[{
                "role": "user",
                "content": prompt
            }],
            max_tokens=150,
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        raise Exception(f"Errore nella chiamata a OpenAI: {str(e)}")

def get_gemini_response(prompt: str) -> str:
    """Get a response from Gemini model."""
    try:
        gemini_model = genai.GenerativeModel('gemini-2.0-flash')
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        raise Exception(f"Errore nella chiamata a Gemini: {str(e)}")

def load_context(file_obj: gr.File) -> Tuple[str, str]:
    """Load and process context from file."""
    global context_text
    try:
        if file_obj is None:
            return ERROR_MESSAGES["no_file"], ""
            
        file_path = file_obj.name
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension not in SUPPORTED_FILE_TYPES:
            return ERROR_MESSAGES["unsupported_file"], ""
            
        print(f"Debug - Loading file: {file_path}")
        
        if file_extension == ".pdf":
            with open(file_path, "rb") as file:
                pdf_reader = PyPDF2.PdfReader(file)
                context_text = ""
                for page in pdf_reader.pages:
                    context_text += page.extract_text() + "\n"
                print(f"Debug - PDF content length: {len(context_text)}")
        else:
            with open(file_path, "r", encoding="utf-8") as file:
                context_text = file.read()
                print(f"Debug - Text file content length: {len(context_text)}")
                
        return ERROR_MESSAGES["success"], context_text
    except Exception as e:
        return f"❌ Errore nel caricamento del file: {str(e)}", ""

def process_uploaded_file(file) -> str:
    """Process an uploaded file and store its content."""
    global context_text
    try:
        if file is None:
            return ERROR_MESSAGES["no_file"]
        
        status, content = load_context(file)
        if status == ERROR_MESSAGES["success"]:
            context_text = content
            return status
        return status
    except Exception as e:
        return ERROR_MESSAGES["api_error"].format(str(e))

def update_api_keys(openai_key: str, anthropic_key: str, google_key: str) -> str:
    """Update API keys for the models."""
    global GEMINI_AVAILABLE
    try:
        if openai_key:
            os.environ["OPENAI_API_KEY"] = openai_key
            openai.api_key = openai_key
        if anthropic_key:
            os.environ["ANTHROPIC_API_KEY"] = anthropic_key
        if google_key:
            os.environ["GOOGLE_API_KEY"] = google_key
            genai.configure(api_key=google_key)
            GEMINI_AVAILABLE = True
        return ERROR_MESSAGES["api_keys_updated"]
    except Exception as e:
        return ERROR_MESSAGES["api_error"].format(str(e))

def get_recommendation(conversation_text: str, model_choice: str) -> str:
    """Get AI recommendation based on context and conversation."""
    global context_text
    try:
        if not context_text:
            print("Debug - No context available")
            return ERROR_MESSAGES["no_context"]
            
        if not conversation_text:
            print("Debug - No conversation text available")
            return ERROR_MESSAGES["no_conversation"]
            
        prompt = create_prompt(context_text, conversation_text)
        print(f"Debug - Context length: {len(context_text)}")
        print(f"Debug - Conversation length: {len(conversation_text)}")
        print(f"Debug - First 100 chars of context: {context_text[:100]}")
        print(f"Debug - Selected model: {model_choice}")
            
        if model_choice == "Google (Gemini 2.0 Flash)":
            if not GEMINI_AVAILABLE:
                return ERROR_MESSAGES["gemini_not_available"]
            return get_gemini_response(prompt)
        elif model_choice == "Claude (3.7 Sonnet)":
            if not ANTHROPIC_AVAILABLE:
                return "⚠️ Claude non disponibile. Inserisci una Anthropic API Key nel file .env per utilizzarlo."
            return get_claude_response(prompt)
        elif model_choice == "OpenAI (GPT-4 Turbo)":
            if not OPENAI_AVAILABLE:
                return "⚠️ OpenAI non disponibile. Inserisci una OpenAI API Key nel file .env per utilizzarlo."
            return get_openai_response(prompt)
    except Exception as e:
        print(f"Debug - Error occurred: {str(e)}")
        return ERROR_MESSAGES["api_error"].format(str(e))

def transcribe_audio(audio_path: Optional[str], conversation_history: str) -> str:
    """Transcribe audio and update conversation history."""
    try:
        if audio_path is None:
            return conversation_history
            
        # Convert audio to WAV format
        audio = AudioSegment.from_file(audio_path)
        audio.export(TEMP_AUDIO_FILE, format="wav")
        
        # Initialize recognizer and transcribe
        recognizer = sr.Recognizer()
        with sr.AudioFile(TEMP_AUDIO_FILE) as source:
            audio_data = recognizer.record(source)
            transcribed_text = recognizer.recognize_google(audio_data, language="it-IT")
        
        # Clean up temporary file
        os.remove(TEMP_AUDIO_FILE)
        
        # Update conversation history
        return conversation_history + "\n" + transcribed_text if conversation_history else transcribed_text
    except Exception as e:
        return ERROR_MESSAGES["transcription_error"].format(str(e))

# Create Gradio interface
with gr.Blocks(css="""
    .suggestion-box {
        background-color: #f7f7f7;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        border: 1px solid #ddd;
    }
""") as demo:
    gr.Markdown("# 🤖 Assistente Vendite AI")
    
    # Suggestions Section (at the top)
    with gr.Row():
        with gr.Column(scale=2):
            available_models = []
            # Add models in preferred order: Google first, then Claude, then OpenAI
            if GEMINI_AVAILABLE:
                available_models.append("Google (Gemini 2.0 Flash)")
            if ANTHROPIC_AVAILABLE:
                available_models.append("Claude (3.7 Sonnet)")
            if OPENAI_AVAILABLE:
                available_models.append("OpenAI (GPT-4 Turbo)")
            
            if not available_models:
                available_models = ["⚠️ Nessun modello disponibile. Inserisci le API Key nel file .env"]
            
            model_choice = gr.Radio(
                choices=available_models,
                value=available_models[0],  # First model (Google) will be default
                label="Seleziona il modello AI"
            )
            
            if not GEMINI_AVAILABLE:
                gr.Markdown(ERROR_MESSAGES["gemini_not_available"])
            if not ANTHROPIC_AVAILABLE:
                gr.Markdown("⚠️ Claude non disponibile. Inserisci una Anthropic API Key nel file .env per utilizzarlo.")
            if not OPENAI_AVAILABLE:
                gr.Markdown("⚠️ OpenAI non disponibile. Inserisci una OpenAI API Key nel file .env per utilizzarlo.")
                
            suggest_button = gr.Button("Ottieni Suggerimenti")
            suggestions_output = gr.Markdown(label="Suggerimenti", elem_classes="suggestion-box")
    
    # Context Document Section
    with gr.Row():
        with gr.Column():
            file_input = gr.File(
                label="Carica Documento di Contesto",
                file_types=SUPPORTED_FILE_TYPES
            )
            process_button = gr.Button("Processa File")
            context_status = gr.Markdown(label="Status")
            
            process_button.click(
                process_uploaded_file,
                inputs=file_input,
                outputs=context_status
            )
    
    # Audio and Conversation Section
    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(sources=["microphone"], type="filepath", label="Registra Audio")
            conversation_history = gr.Textbox(label="Cronologia Conversazione", lines=10, interactive=True)
            
            audio_input.change(
                transcribe_audio,
                inputs=[audio_input, conversation_history],
                outputs=conversation_history
            )
    
    # Connect the suggestion button
    suggest_button.click(
        get_recommendation,
        inputs=[conversation_history, model_choice],
        outputs=suggestions_output
    )

if __name__ == "__main__":
    demo.launch() 