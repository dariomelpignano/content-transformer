"""
Content Transformer: A tool to transcribe media content using Faster Whisper and AI.
"""
import os
import logging
import tempfile
from pathlib import Path
from datetime import datetime
import gradio as gr
from dotenv import load_dotenv
import openai
import anthropic
import yt_dlp
import subprocess
from pydub import AudioSegment
from faster_whisper import WhisperModel
import requests
import zipfile
import io

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# SECURITY NOTE: API keys are loaded from .env file which should NEVER be committed to version control.
# See .env.example for the expected structure.
load_dotenv()

# Create output directory if it doesn't exist
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

class ContentTransformer:
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        # Initialize Faster Whisper model (using CPU for better compatibility)
        self.model = WhisperModel("base", device="cpu", compute_type="int8")
        logger.info("Faster Whisper model loaded successfully")
        self._processing = False
    
    def cancel_processing(self):
        """Cancel the current processing operation"""
        self._processing = False
        return gr.update(value="Processing cancelled by user."), None, None, gr.update(visible=False), gr.update(visible=False), gr.update(value=0), gr.update(value=0), gr.update(value=0), gr.update(value="")
    
    def save_api_keys(self, openai_key, anthropic_key):
        """Save API keys to .env file"""
        try:
            with open('.env', 'w') as f:
                if openai_key:
                    f.write(f"OPENAI_API_KEY={openai_key}\n")
                if anthropic_key:
                    f.write(f"ANTHROPIC_API_KEY={anthropic_key}\n")
            return "API keys saved successfully!"
        except Exception as e:
            return f"Error saving API keys: {str(e)}"
    
    def download_youtube_audio(self, url):
        """Download audio from YouTube URL using yt-dlp"""
        try:
            # Validate URL format
            if not url or not isinstance(url, str):
                raise ValueError("Invalid URL provided")
            
            # Clean the URL
            url = url.strip()
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
            
            # Create temporary file for WAV output
            temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            
            try:
                # Configure yt-dlp to download and extract audio directly to WAV
                ydl_opts = {
                    'format': 'bestaudio/best',
                    'outtmpl': temp_wav.name[:-4],  # Remove .wav extension as yt-dlp will add it
                    'postprocessors': [{
                        'key': 'FFmpegExtractAudio',
                        'preferredcodec': 'wav',
                        'preferredquality': '192',
                    }],
                    'quiet': True,
                    'no_warnings': True,
                }
                
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=True)
                    logger.info(f"Successfully downloaded audio from {url}")
                    
                    # Convert to 16kHz mono WAV for Whisper
                    temp_converted = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
                    subprocess.run([
                        'ffmpeg',
                        '-i', f"{temp_wav.name[:-4]}.wav",  # Input file (yt-dlp adds .wav)
                        '-acodec', 'pcm_s16le',
                        '-ar', '16000',
                        '-ac', '1',
                        '-y',
                        temp_converted.name
                    ], check=True, capture_output=True)
                    
                    # Clean up the original WAV file
                    os.unlink(f"{temp_wav.name[:-4]}.wav")
                    
                    return temp_converted.name, info.get('title', 'untitled')
                
            except Exception as e:
                # Clean up temp files if download fails
                if os.path.exists(f"{temp_wav.name[:-4]}.wav"):
                    os.unlink(f"{temp_wav.name[:-4]}.wav")
                if os.path.exists(temp_wav.name):
                    os.unlink(temp_wav.name)
                raise Exception(f"Failed to download/convert audio: {str(e)}")
                
        except Exception as e:
            logger.error(f"Error in download_youtube_audio: {str(e)}")
            raise Exception(f"Error downloading YouTube video: {str(e)}")
    
    def extract_audio_from_video(self, video_path):
        """Extract audio from video file using ffmpeg"""
        try:
            # Ensure video_path is a string and exists
            video_path = str(video_path)
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")
            
            logger.info(f"Extracting audio from video: {video_path}")
            
            # Create a temporary file with a unique name
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
            temp_path = temp_file.name
            temp_file.close()
            
            logger.info(f"Created temporary file: {temp_path}")
            
            # Extract audio to MP3 format
            cmd = [
                'ffmpeg', '-i', video_path,
                '-vn',  # No video
                '-acodec', 'libmp3lame',  # Use MP3 codec
                '-q:a', '2',  # High quality MP3
                '-y',  # Overwrite output file if it exists
                temp_path
            ]
            logger.info(f"Running command: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info(f"FFmpeg output: {result.stdout}")
            
            if result.stderr:
                logger.warning(f"FFmpeg warnings: {result.stderr}")
            
            # Verify the output file exists and has content
            if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
                raise Exception("Failed to extract audio: output file is empty or missing")
            
            return temp_path
        except Exception as e:
            logger.error(f"Error in extract_audio_from_video: {str(e)}")
            raise Exception(f"Error extracting audio from video: {str(e)}")
    
    def transcribe_audio(self, audio_path, language="en"):
        """Transcribe audio using Faster Whisper"""
        try:
            # Ensure audio_path is a string and exists
            audio_path = str(audio_path)
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
            logger.info(f"Transcribing audio in language: {language}")
            logger.info(f"Audio file path: {audio_path}")
            
            # If language is 'auto', let Faster Whisper detect it
            if language == "auto":
                language = None
            
            # Transcribe with Faster Whisper
            segments, info = self.model.transcribe(
                audio_path,
                language=language,
                beam_size=5,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500)
            )
            
            # Get total duration from info
            total_duration = info.duration if hasattr(info, 'duration') else 0
            logger.info(f"Total duration: {total_duration} seconds")
            
            # Combine all segments into a single text and track progress
            transcription = ""
            last_progress = 0
            last_yield_time = datetime.now()
            
            try:
                for segment in segments:
                    if not self._processing:
                        logger.info("Transcription cancelled by user")
                        # Save partial transcription if we have any
                        if transcription.strip():
                            yield 0, transcription.strip()
                        return transcription.strip()
                        
                    transcription += segment.text + " "
                    # Calculate progress based on segment end time
                    if total_duration > 0:
                        current_progress = min(1.0, segment.end / total_duration)
                        # Yield if progress has increased by at least 5% or 5 seconds have passed
                        current_time = datetime.now()
                        if (current_progress - last_progress >= 0.05 or 
                            (current_time - last_yield_time).total_seconds() >= 5):
                            last_progress = current_progress
                            last_yield_time = current_time
                            yield current_progress, transcription.strip()
                
                transcription = transcription.strip()
                logger.info(f"Transcription completed for {audio_path}")
                return transcription
                
            except KeyboardInterrupt:
                logger.warning("Transcription interrupted by user")
                # Save partial transcription if we have any
                if transcription.strip():
                    yield 0, transcription.strip()
                return transcription.strip()
            except Exception as e:
                logger.error(f"Error during transcription: {str(e)}")
                # Save partial transcription if we have any
                if transcription.strip():
                    yield 0, transcription.strip()
                return transcription.strip()
            
        except Exception as e:
            logger.error(f"Error in transcribe_audio: {str(e)}")
            raise Exception(f"Error transcribing audio: {str(e)}")
    
    def process_content(self, file_path, youtube_urls, language):
        """Process uploaded file or YouTube URLs"""
        try:
            self._processing = True
            results = []
            current_item = 0
            total_items = 0
            temp_files = []  # Track temporary files for cleanup
            
            # Count total items to process
            if youtube_urls:
                urls = [url.strip() for url in youtube_urls.split('\n') if url.strip()]
                total_items += len(urls)
            if file_path:
                if isinstance(file_path, list):
                    total_items += len([f for f in file_path if f is not None])
                else:
                    total_items += 1
            
            if total_items == 0:
                return gr.update(value="Please upload a file or enter YouTube URLs"), None, gr.update(interactive=True), gr.update(visible=False), gr.update(visible=False), gr.update(value=0), gr.update(value=0), gr.update(value=0), gr.update(value="")
            
            try:
                # Handle YouTube URLs
                if youtube_urls and self._processing:
                    urls = [url.strip() for url in youtube_urls.split('\n') if url.strip()]
                    if urls:
                        status = f"Processing {len(urls)} YouTube URLs..."
                        results.append(status)
                        logger.info(status)
                        yield gr.update(value="\n".join(results)), None, gr.update(interactive=False), gr.update(visible=True), gr.update(visible=True), gr.update(value=0), gr.update(value=0), gr.update(value=0), gr.update(value="")
                        
                        for url in urls:
                            if not self._processing:
                                break
                            current_item += 1
                            status = f"Processing YouTube video {current_item}/{total_items}: {url}"
                            results.append(status)
                            logger.info(status)
                            yield gr.update(value="\n".join(results)), None, gr.update(interactive=False), gr.update(visible=True), gr.update(visible=True), gr.update(value=int((current_item/total_items)*100)), gr.update(value=0), gr.update(value=0), gr.update(value=url)
                            try:
                                # Download YouTube audio
                                temp_file, content_name = self.download_youtube_audio(url)
                                temp_files.append(temp_file)  # Track for cleanup
                                yield gr.update(value="\n".join(results)), None, gr.update(interactive=False), gr.update(visible=True), gr.update(visible=True), gr.update(value=int((current_item/total_items)*100)), gr.update(value=0), gr.update(value=30), gr.update(value=f"Downloading: {content_name}")
                                
                                # Process file with progress updates
                                transcription = ""
                                try:
                                    for progress, partial_text in self.transcribe_audio(temp_file, language):
                                        transcription = partial_text
                                        yield gr.update(value="\n".join(results)), None, gr.update(interactive=False), gr.update(visible=True), gr.update(visible=True), gr.update(value=int((current_item/total_items)*100)), gr.update(value=0), gr.update(value=int(progress*100)), gr.update(value=f"Transcribing: {content_name}")
                                except Exception as e:
                                    logger.error(f"Error during transcription of {content_name}: {str(e)}")
                                    if transcription.strip():
                                        # Save partial transcription if we have any
                                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                        output_file = OUTPUT_DIR / f"{content_name}_{timestamp}_partial.md"
                                        with open(output_file, "w", encoding="utf-8") as f:
                                            f.write(f"# {content_name} (Partial Transcription)\n\n")
                                            f.write(f"## URL\n\n{url}\n\n")
                                            f.write("## Transcription\n\n")
                                            f.write(transcription)
                                        results.append(f"Saved partial transcription for {content_name} to {output_file}")
                                    raise
                                
                                # Save transcription to file
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                output_file = OUTPUT_DIR / f"{content_name}_{timestamp}.md"
                                with open(output_file, "w", encoding="utf-8") as f:
                                    f.write(f"# {content_name}\n\n")
                                    f.write(f"## URL\n\n{url}\n\n")
                                    f.write("## Transcription\n\n")
                                    f.write(transcription)
                                
                                result = f"Completed: {content_name} (saved to {output_file})"
                                results.append(result)
                                logger.info(result)
                                yield gr.update(value="\n".join(results)), None, gr.update(interactive=False), gr.update(visible=True), gr.update(visible=True), gr.update(value=int((current_item/total_items)*100)), gr.update(value=0), gr.update(value=100), gr.update(value=f"Completed: {content_name}")
                            except Exception as e:
                                error_msg = f"Error processing YouTube video {url}: {str(e)}"
                                logger.error(error_msg)
                                results.append(error_msg)
                                yield gr.update(value="\n".join(results)), None, gr.update(interactive=False), gr.update(visible=True), gr.update(visible=True), gr.update(value=int((current_item/total_items)*100)), gr.update(value=0), gr.update(value=0), gr.update(value=f"Error: {content_name}")
                
                # Handle file upload
                if file_path and self._processing:
                    # Handle multiple files
                    if isinstance(file_path, list):
                        status = f"Processing {len([f for f in file_path if f is not None])} uploaded files..."
                        results.append(status)
                        logger.info(status)
                        yield gr.update(value="\n".join(results)), None, gr.update(interactive=False), gr.update(visible=True), gr.update(visible=True), gr.update(value=0), gr.update(value=0), gr.update(value=0), gr.update(value="")
                        
                        for file in file_path:
                            if not self._processing:
                                break
                            if file is None:
                                continue
                            current_item += 1
                            content_name = Path(file.name).stem
                            status = f"Processing file {current_item}/{total_items}: {content_name}"
                            results.append(status)
                            logger.info(status)
                            yield gr.update(value="\n".join(results)), None, gr.update(interactive=False), gr.update(visible=True), gr.update(visible=True), gr.update(value=int((current_item/total_items)*100)), gr.update(value=0), gr.update(value=0), gr.update(value=content_name)
                            try:
                                # Extract audio from video
                                temp_audio = self.extract_audio_from_video(file.name)
                                temp_files.append(temp_audio)  # Track for cleanup
                                yield gr.update(value="\n".join(results)), None, gr.update(interactive=False), gr.update(visible=True), gr.update(visible=True), gr.update(value=int((current_item/total_items)*100)), gr.update(value=0), gr.update(value=30), gr.update(value=f"Extracting audio: {content_name}")
                                
                                # Process file with progress updates
                                transcription = ""
                                try:
                                    for progress, partial_text in self.transcribe_audio(temp_audio, language):
                                        transcription = partial_text
                                        yield gr.update(value="\n".join(results)), None, gr.update(interactive=False), gr.update(visible=True), gr.update(visible=True), gr.update(value=int((current_item/total_items)*100)), gr.update(value=0), gr.update(value=int(progress*100)), gr.update(value=f"Transcribing: {content_name}")
                                except Exception as e:
                                    logger.error(f"Error during transcription of {content_name}: {str(e)}")
                                    if transcription.strip():
                                        # Save partial transcription if we have any
                                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                        output_file = OUTPUT_DIR / f"{content_name}_{timestamp}_partial.md"
                                        with open(output_file, "w", encoding="utf-8") as f:
                                            f.write(f"# {content_name} (Partial Transcription)\n\n")
                                            f.write("## Transcription\n\n")
                                            f.write(transcription)
                                        results.append(f"Saved partial transcription for {content_name} to {output_file}")
                                    raise
                                
                                # Save transcription to file
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                output_file = OUTPUT_DIR / f"{content_name}_{timestamp}.md"
                                with open(output_file, "w", encoding="utf-8") as f:
                                    f.write(f"# {content_name}\n\n")
                                    f.write("## Transcription\n\n")
                                    f.write(transcription)
                                
                                result = f"Completed: {content_name} (saved to {output_file})"
                                results.append(result)
                                logger.info(result)
                                yield gr.update(value="\n".join(results)), None, gr.update(interactive=False), gr.update(visible=True), gr.update(visible=True), gr.update(value=int((current_item/total_items)*100)), gr.update(value=0), gr.update(value=100), gr.update(value=f"Completed: {content_name}")
                            except Exception as e:
                                error_msg = f"Error processing file {content_name}: {str(e)}"
                                logger.error(error_msg)
                                results.append(error_msg)
                                yield gr.update(value="\n".join(results)), None, gr.update(interactive=False), gr.update(visible=True), gr.update(visible=True), gr.update(value=int((current_item/total_items)*100)), gr.update(value=0), gr.update(value=0), gr.update(value=f"Error: {content_name}")
                
                # Final status
                if self._processing:
                    yield gr.update(value="\n".join(results)), None, gr.update(interactive=True), gr.update(visible=False), gr.update(visible=False), gr.update(value=100), gr.update(value=100), gr.update(value=100), gr.update(value="Processing completed!")
                else:
                    yield gr.update(value="\n".join(results)), None, gr.update(interactive=True), gr.update(visible=False), gr.update(visible=False), gr.update(value=100), gr.update(value=0), gr.update(value=0), gr.update(value="Processing cancelled!")
                    
            finally:
                # Clean up temporary files
                for temp_file in temp_files:
                    try:
                        if os.path.exists(temp_file):
                            os.unlink(temp_file)
                            logger.info(f"Cleaned up temporary file: {temp_file}")
                    except Exception as e:
                        logger.warning(f"Error cleaning up temporary file {temp_file}: {str(e)}")
            
        except Exception as e:
            logger.error(f"Error in process_content: {str(e)}")
            yield gr.update(value=f"Error: {str(e)}"), None, gr.update(interactive=True), gr.update(visible=False), gr.update(visible=False), gr.update(value=0), gr.update(value=0), gr.update(value=0), gr.update(value="Error occurred!")

def create_interface():
    """Create the Gradio interface"""
    transformer = ContentTransformer()
    
    with gr.Blocks(title="Content Transformer") as app:
        gr.Markdown("# 🎙️ Content Transformer 🎙️")
        gr.Markdown("Transform your media content into text using AI-powered transcription")
        
        with gr.Tab("Setup"):
            gr.Markdown("### 🔑 API Configuration")
            with gr.Row():
                with gr.Column():
                    openai_key = gr.Textbox(
                        label="OpenAI API Key",
                        type="password",
                        value=os.getenv("OPENAI_API_KEY", ""),
                        placeholder="Enter your OpenAI API key"
                    )
                with gr.Column():
                    anthropic_key = gr.Textbox(
                        label="Claude API Key",
                        type="password",
                        value=os.getenv("ANTHROPIC_API_KEY", ""),
                        placeholder="Enter your Anthropic API key"
                    )
            save_keys_btn = gr.Button("Save API Keys")
            setup_output = gr.Textbox(label="Status")
            
            save_keys_btn.click(
                transformer.save_api_keys,
                inputs=[openai_key, anthropic_key],
                outputs=setup_output
            )
        
        with gr.Tab("Transcribe"):
            gr.Markdown("### 📝 Transcribe Content")
            with gr.Row():
                with gr.Column():
                    file_input = gr.File(
                        label="Upload Media File",
                        file_types=["video", "audio"],
                        file_count="multiple"
                    )
                with gr.Column():
                    youtube_urls = gr.Textbox(
                        label="YouTube URLs",
                        placeholder="Enter YouTube video URLs (one per line)",
                        lines=5
                    )
            
            language = gr.Radio(
                choices=["auto", "en", "it", "fr", "de", "es"],
                value="auto",  # Default to auto-detection
                label="Language",
                info="Select the language of the content or use auto-detection"
            )
            
            with gr.Row():
                transcribe_btn = gr.Button("Transcribe", variant="primary")
                cancel_btn = gr.Button("Cancel Processing", variant="stop", visible=False)
            
            with gr.Row():
                status = gr.Textbox(label="Status", lines=10)  # Increased lines for better visibility
                processing_indicator = gr.HTML(
                    """
                    <div style="text-align: center; padding: 10px; background-color: #f0f0f0; border-radius: 5px;">
                        <div class="spinner"></div>
                        <p>Processing in progress...</p>
                    </div>
                    """,
                    visible=False
                )
            
            with gr.Row():
                progress_bar = gr.Slider(
                    minimum=0,
                    maximum=100,
                    value=0,
                    label="Overall Progress (%)",
                    interactive=False
                )
                memory_bar = gr.Slider(
                    minimum=0,
                    maximum=100,
                    value=0,
                    label="Memory Usage (%)",
                    interactive=False
                )
            
            with gr.Row():
                current_file_progress = gr.Slider(
                    minimum=0,
                    maximum=100,
                    value=0,
                    label="Current File Progress (%)",
                    interactive=False
                )
                current_file_name = gr.Textbox(
                    label="Current File",
                    value="",
                    interactive=False
                )
            
            # Add CSS for the spinner
            gr.HTML("""
                <style>
                .spinner {
                    width: 40px;
                    height: 40px;
                    margin: 0 auto;
                    border: 4px solid #f3f3f3;
                    border-top: 4px solid #3498db;
                    border-radius: 50%;
                    animation: spin 1s linear infinite;
                }
                @keyframes spin {
                    0% { transform: rotate(0deg); }
                    100% { transform: rotate(360deg); }
                }
                </style>
            """)
            
            # Update the click events to handle UI state
            def update_ui_state(is_processing):
                return [
                    gr.update(interactive=not is_processing),  # file_input
                    gr.update(interactive=not is_processing),  # youtube_urls
                    gr.update(interactive=not is_processing),  # transcribe_btn
                    gr.update(visible=is_processing),  # cancel_btn
                    gr.update(visible=is_processing)   # processing_indicator
                ]
            
            transcribe_btn.click(
                lambda: update_ui_state(True),
                outputs=[file_input, youtube_urls, transcribe_btn, cancel_btn, processing_indicator]
            ).then(
                transformer.process_content,
                inputs=[file_input, youtube_urls, language],
                outputs=[status, file_input, youtube_urls, processing_indicator, cancel_btn, progress_bar, memory_bar, current_file_progress, current_file_name]
            ).then(
                lambda: update_ui_state(False),
                outputs=[file_input, youtube_urls, transcribe_btn, cancel_btn, processing_indicator]
            )
            
            cancel_btn.click(
                transformer.cancel_processing,
                outputs=[status, file_input, youtube_urls, processing_indicator, cancel_btn, progress_bar, memory_bar, current_file_progress, current_file_name]
            ).then(
                lambda: update_ui_state(False),
                outputs=[file_input, youtube_urls, transcribe_btn, cancel_btn, processing_indicator]
            )
    
    return app

if __name__ == "__main__":
    app = create_interface()
    app.queue()  # Enable queue for the entire app
    app.launch() 