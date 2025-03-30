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
        # Return 8 values to match the expected outputs
        return gr.update(value="Processing cancelled by user."), None, None, gr.update(visible=False), gr.update(visible=False), gr.update(value=100), gr.update(value=0), gr.update(value="Processing cancelled!")

    def save_api_keys(self, openai_key, anthropic_key):
        """Save API keys to .env file"""
        try:
            with open('.env', 'w') as f:
                if openai_key:
                    f.write(f"OPENAI_API_KEY={openai_key}\\n")
                if anthropic_key:
                    f.write(f"ANTHROPIC_API_KEY={anthropic_key}\\n")
            # Reload keys after saving
            self.openai_api_key = os.getenv("OPENAI_API_KEY", openai_key)
            self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", anthropic_key)
            logger.info("API keys saved and reloaded.")
            return "API keys saved successfully!"
        except Exception as e:
            logger.error(f"Error saving API keys: {str(e)}")
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
                    downloaded_file_path = f"{temp_wav.name[:-4]}.wav" # yt-dlp adds .wav

                    # Check if the downloaded file exists before proceeding
                    if not os.path.exists(downloaded_file_path):
                         raise FileNotFoundError(f"yt-dlp failed to create the audio file: {downloaded_file_path}")

                    # Convert to 16kHz mono WAV for Whisper
                    temp_converted = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
                    ffmpeg_command = [
                        'ffmpeg',
                        '-i', downloaded_file_path,  # Input file
                        '-acodec', 'pcm_s16le',
                        '-ar', '16000',
                        '-ac', '1',
                        '-y',
                        temp_converted.name
                    ]
                    logger.info(f"Running ffmpeg conversion: {' '.join(ffmpeg_command)}")
                    result = subprocess.run(ffmpeg_command, check=True, capture_output=True, text=True)
                    logger.info(f"FFmpeg conversion output: {result.stdout}")
                    if result.stderr:
                         logger.warning(f"FFmpeg conversion warnings: {result.stderr}")


                    # Clean up the original WAV file from yt-dlp
                    os.unlink(downloaded_file_path)
                    logger.info(f"Cleaned up intermediate file: {downloaded_file_path}")

                    return temp_converted.name, info.get('title', 'untitled')

            except Exception as e:
                logger.error(f"Error during yt-dlp download/conversion for {url}: {str(e)}")
                # Clean up temp files if download/conversion fails
                potential_yt_dlp_file = f"{temp_wav.name[:-4]}.wav"
                if os.path.exists(potential_yt_dlp_file):
                    try:
                        os.unlink(potential_yt_dlp_file)
                        logger.info(f"Cleaned up failed yt-dlp file: {potential_yt_dlp_file}")
                    except Exception as cleanup_e:
                         logger.warning(f"Could not clean up {potential_yt_dlp_file}: {cleanup_e}")
                if os.path.exists(temp_wav.name):
                     try:
                         os.unlink(temp_wav.name) # The initial temp file object
                         logger.info(f"Cleaned up initial temp file: {temp_wav.name}")
                     except Exception as cleanup_e:
                         logger.warning(f"Could not clean up {temp_wav.name}: {cleanup_e}")

                # Re-raise the exception to be caught by the outer handler
                raise

        except Exception as e:
            logger.error(f"Error in download_youtube_audio for {url}: {str(e)}")
            # Ensure temp_wav.name is cleaned up if it exists and wasn't handled above
            if 'temp_wav' in locals() and os.path.exists(temp_wav.name):
                 try:
                     os.unlink(temp_wav.name)
                     logger.info(f"Cleaned up initial temp file in outer exception handler: {temp_wav.name}")
                 except Exception as cleanup_e:
                     logger.warning(f"Outer handler could not clean up {temp_wav.name}: {cleanup_e}")
            raise Exception(f"Error downloading YouTube video: {str(e)}")


    def extract_audio_from_video(self, video_path):
        """Extract audio from video file using ffmpeg"""
        try:
            # Ensure video_path is a string and exists
            video_path = str(video_path)
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")

            logger.info(f"Extracting audio from video: {video_path}")

            # Create a temporary file with a unique name for the MP3 output
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
            temp_path = temp_file.name
            temp_file.close() # Close the file handle immediately

            logger.info(f"Created temporary file for MP3 output: {temp_path}")

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

            # Convert the extracted MP3 to 16kHz mono WAV for Whisper
            temp_converted_wav = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            ffmpeg_convert_cmd = [
                 'ffmpeg',
                 '-i', temp_path, # Input MP3
                 '-acodec', 'pcm_s16le',
                 '-ar', '16000',
                 '-ac', '1',
                 '-y',
                 temp_converted_wav.name
            ]
            logger.info(f"Running ffmpeg conversion to WAV: {' '.join(ffmpeg_convert_cmd)}")
            convert_result = subprocess.run(ffmpeg_convert_cmd, check=True, capture_output=True, text=True)
            logger.info(f"FFmpeg conversion output: {convert_result.stdout}")
            if convert_result.stderr:
                 logger.warning(f"FFmpeg conversion warnings: {convert_result.stderr}")

            # Clean up the intermediate MP3 file
            os.unlink(temp_path)
            logger.info(f"Cleaned up intermediate MP3 file: {temp_path}")

            return temp_converted_wav.name # Return the path to the final WAV file

        except Exception as e:
            logger.error(f"Error in extract_audio_from_video: {str(e)}")
            # Clean up temporary files on error
            if 'temp_path' in locals() and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                    logger.info(f"Cleaned up temp MP3 file on error: {temp_path}")
                except Exception as cleanup_e:
                     logger.warning(f"Could not clean up {temp_path} on error: {cleanup_e}")
            if 'temp_converted_wav' in locals() and os.path.exists(temp_converted_wav.name):
                try:
                    os.unlink(temp_converted_wav.name)
                    logger.info(f"Cleaned up temp WAV file on error: {temp_converted_wav.name}")
                except Exception as cleanup_e:
                     logger.warning(f"Could not clean up {temp_converted_wav.name} on error: {cleanup_e}")
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
            logger.info(f"Detected language: {info.language} with probability {info.language_probability:.2f}")
            logger.info(f"Total duration: {total_duration} seconds")

            # Combine all segments into a single text and track progress
            transcription = ""
            last_progress = 0
            last_yield_time = datetime.now()

            try:
                segment_list = list(segments) # Convert generator to list to allow cancelling
                for i, segment in enumerate(segment_list):
                    if not self._processing: # Check for cancellation flag
                        logger.info("Transcription cancelled by user flag.")
                        # Save partial transcription if we have any
                        if transcription.strip():
                            # Yield 0 progress and the partial text
                             yield 0, transcription.strip()
                        return transcription.strip() # Return whatever was transcribed

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
                            # Yield progress and the current partial transcription
                            yield current_progress, transcription.strip()

                transcription = transcription.strip()
                logger.info(f"Transcription completed for {audio_path}")
                # Yield final progress and full transcription before returning
                yield 1.0, transcription
                return transcription

            except KeyboardInterrupt:
                logger.warning("Transcription interrupted by KeyboardInterrupt")
                # Save partial transcription if we have any
                if transcription.strip():
                     yield 0, transcription.strip()
                return transcription.strip() # Return partial if interrupted
            except Exception as e:
                logger.error(f"Error during transcription segment processing: {str(e)}")
                # Save partial transcription if we have any
                if transcription.strip():
                    yield 0, transcription.strip()
                # Re-raise the exception to be handled by the outer try-except
                raise

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
            urls_list = []
            if youtube_urls:
                urls_list = [url.strip() for url in youtube_urls.split('\\n') if url.strip()]
                total_items += len(urls_list)

            files_list = []
            if file_path:
                if isinstance(file_path, list):
                    files_list = [f for f in file_path if f is not None]
                else:
                    files_list = [file_path] # Treat single file as a list
                total_items += len(files_list)

            if total_items == 0:
                # Return 8 values
                return gr.update(value="Please upload a file or enter YouTube URLs"), None, None, gr.update(visible=False), gr.update(visible=False), gr.update(value=0), gr.update(value=0), gr.update(value="")


            # -------- Process YouTube URLs --------
            if urls_list and self._processing:
                status = f"Processing {len(urls_list)} YouTube URLs..."
                results.append(status)
                logger.info(status)
                # Yield 8 values
                yield gr.update(value="\\n".join(results)), None, None, gr.update(visible=True), gr.update(visible=True), gr.update(value=0), gr.update(value=0), gr.update(value="")

                for url in urls_list:
                    if not self._processing: break
                    current_item += 1
                    content_name = "YouTube Video" # Placeholder
                    status = f"Processing YouTube video {current_item}/{total_items}: {url}"
                    results.append(status)
                    logger.info(status)
                    # Yield 8 values
                    yield gr.update(value="\\n".join(results)), None, None, gr.update(visible=True), gr.update(visible=True), gr.update(value=int((current_item/total_items)*100)), gr.update(value=0), gr.update(value=url)
                    try:
                        # Download YouTube audio
                        temp_file, content_name = self.download_youtube_audio(url)
                        temp_files.append(temp_file)  # Track for cleanup
                        # Yield 8 values
                        yield gr.update(value="\\n".join(results)), None, None, gr.update(visible=True), gr.update(visible=True), gr.update(value=int((current_item/total_items)*100)), gr.update(value=30), gr.update(value=f"Downloading: {content_name}")

                        # Process file with progress updates
                        transcription = ""
                        try:
                            for progress, partial_text in self.transcribe_audio(temp_file, language):
                                transcription = partial_text
                                # Yield 8 values
                                yield gr.update(value="\\n".join(results)), None, None, gr.update(visible=True), gr.update(visible=True), gr.update(value=int((current_item/total_items)*100)), gr.update(value=int(progress*100)), gr.update(value=f"Transcribing: {content_name}")
                        except Exception as e:
                            logger.error(f"Error during transcription of {content_name}: {str(e)}")
                            if transcription.strip():
                                # Save partial transcription if we have any
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                output_file = OUTPUT_DIR / f"{content_name}_{timestamp}_partial.md"
                                with open(output_file, "w", encoding="utf-8") as f:
                                    f.write(f"# {content_name} (Partial Transcription)\n\n")
                                    if 'url' in locals(): # Check if url exists (for YouTube videos)
                                        f.write(f"## URL\n\n{url}\n\n")
                                    f.write("## Transcription\n\n")
                                    f.write(transcription)
                                results.append(f"Saved partial transcription for {content_name} to {output_file}")
                            raise # Re-raise to be caught by the outer exception handler for this item

                        # Save full transcription to file
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        output_file = OUTPUT_DIR / f"{content_name}_{timestamp}.md"
                        with open(output_file, "w", encoding="utf-8") as f:
                            f.write(f"# {content_name}\n\n")
                            if 'url' in locals(): # Check if url exists (for YouTube videos)
                                f.write(f"## URL\n\n{url}\n\n")
                            f.write("## Transcription\n\n")
                            f.write(transcription)

                        result = f"Completed: {content_name} (saved to {output_file})"
                        results.append(result)
                        logger.info(result)
                        # Yield 8 values
                        yield gr.update(value="\\n".join(results)), None, None, gr.update(visible=True), gr.update(visible=True), gr.update(value=int((current_item/total_items)*100)), gr.update(value=100), gr.update(value=f"Completed: {content_name}")
                    except Exception as e:
                        error_msg = f"Error processing YouTube video {url}: {str(e)}"
                        logger.error(error_msg)
                        results.append(error_msg)
                        # Yield 8 values
                        yield gr.update(value="\\n".join(results)), None, None, gr.update(visible=True), gr.update(visible=True), gr.update(value=int((current_item/total_items)*100)), gr.update(value=0), gr.update(value=f"Error: {content_name or url}")


            # -------- Process File Uploads --------
            if files_list and self._processing:
                status = f"Processing {len(files_list)} uploaded files..."
                results.append(status)
                logger.info(status)
                 # Yield 8 values
                yield gr.update(value="\\n".join(results)), None, None, gr.update(visible=True), gr.update(visible=True), gr.update(value=0), gr.update(value=0), gr.update(value="")

                for file in files_list:
                    if not self._processing: break
                    current_item += 1
                    try:
                         # Use file.name for Gradio File object path
                         file_actual_path = file.name
                         content_name = Path(file_actual_path).stem
                    except AttributeError:
                         logger.error("Uploaded item is not a valid Gradio File object.")
                         results.append(f"Error: Item {current_item} is not a valid file.")
                         continue # Skip this item

                    status = f"Processing file {current_item}/{total_items}: {content_name}"
                    results.append(status)
                    logger.info(status)
                    # Yield 8 values
                    yield gr.update(value="\\n".join(results)), None, None, gr.update(visible=True), gr.update(visible=True), gr.update(value=int((current_item/total_items)*100)), gr.update(value=0), gr.update(value=content_name)
                    try:
                        # Extract audio (now returns WAV path)
                        temp_audio = self.extract_audio_from_video(file_actual_path)
                        temp_files.append(temp_audio)  # Track for cleanup
                        # Yield 8 values
                        yield gr.update(value="\\n".join(results)), None, None, gr.update(visible=True), gr.update(visible=True), gr.update(value=int((current_item/total_items)*100)), gr.update(value=30), gr.update(value=f"Extracting audio: {content_name}")

                        # Process file with progress updates
                        transcription = ""
                        try:
                            for progress, partial_text in self.transcribe_audio(temp_audio, language):
                                transcription = partial_text
                                # Yield 8 values
                                yield gr.update(value="\\n".join(results)), None, None, gr.update(visible=True), gr.update(visible=True), gr.update(value=int((current_item/total_items)*100)), gr.update(value=int(progress*100)), gr.update(value=f"Transcribing: {content_name}")
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
                            raise # Re-raise

                        # Save full transcription to file
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        output_file = OUTPUT_DIR / f"{content_name}_{timestamp}.md"
                        with open(output_file, "w", encoding="utf-8") as f:
                            f.write(f"# {content_name}\n\n")
                            f.write("## Transcription\n\n")
                            f.write(transcription)

                        result = f"Completed: {content_name} (saved to {output_file})"
                        results.append(result)
                        logger.info(result)
                        # Yield 8 values
                        yield gr.update(value="\\n".join(results)), None, None, gr.update(visible=True), gr.update(visible=True), gr.update(value=int((current_item/total_items)*100)), gr.update(value=100), gr.update(value=f"Completed: {content_name}")
                    except Exception as e:
                        error_msg = f"Error processing file {content_name}: {str(e)}"
                        logger.error(error_msg)
                        results.append(error_msg)
                        # Yield 8 values
                        yield gr.update(value="\\n".join(results)), None, None, gr.update(visible=True), gr.update(visible=True), gr.update(value=int((current_item/total_items)*100)), gr.update(value=0), gr.update(value=f"Error: {content_name}")


            # -------- Final Status --------
            if self._processing: # Processing completed normally
                 final_status_msg = "Processing completed!"
                 final_progress = 100
            else: # Processing was cancelled
                 final_status_msg = "Processing cancelled!"
                 final_progress = int((current_item / total_items) * 100) if total_items > 0 else 0

            logger.info(f"Finished processing. Final status: {final_status_msg}")
            # Yield 8 values
            yield gr.update(value="\\n".join(results)), None, None, gr.update(visible=False), gr.update(visible=False), gr.update(value=final_progress), gr.update(value=100), gr.update(value=final_status_msg)


        except Exception as e:
            logger.error(f"Critical error in process_content: {str(e)}", exc_info=True)
             # Yield 8 values
            yield gr.update(value=f"Critical Error: {str(e)}"), None, None, gr.update(visible=False), gr.update(visible=False), gr.update(value=0), gr.update(value=0), gr.update(value="Critical Error!")

        finally:
            self._processing = False # Ensure processing flag is reset
            # Clean up temporary files
            logger.info(f"Cleaning up {len(temp_files)} temporary files...")
            for temp_file_path in temp_files:
                try:
                    if temp_file_path and os.path.exists(temp_file_path):
                        os.unlink(temp_file_path)
                        logger.info(f"Cleaned up temporary file: {temp_file_path}")
                except Exception as e:
                    logger.warning(f"Error cleaning up temporary file {temp_file_path}: {str(e)}")
            logger.info("Cleanup complete.")

# Note: Replaced the entire conflicting create_interface with the local version below
def create_interface():
    """Create the Gradio interface for Content Transformer"""
    transformer = ContentTransformer() # Instantiate our class

    with gr.Blocks(title="Content Transformer") as app:
        gr.Markdown("# 🎙️ Content Transformer 🎙️")
        gr.Markdown("Transform your media content into text using AI-powered transcription")

        with gr.Tab("Setup"):
            gr.Markdown("### 🔑 API Configuration")
            gr.Markdown("Enter your API keys here. They will be stored in a local `.env` file.")
            with gr.Row():
                with gr.Column():
                    openai_key = gr.Textbox(
                        label="OpenAI API Key",
                        type="password",
                        value=os.getenv("OPENAI_API_KEY", ""),
                        placeholder="Enter your OpenAI API key (optional)"
                    )
                with gr.Column():
                    anthropic_key = gr.Textbox(
                        label="Claude API Key",
                        type="password",
                        value=os.getenv("ANTHROPIC_API_KEY", ""),
                        placeholder="Enter your Anthropic API key (optional)"
                    )
            save_keys_btn = gr.Button("Save API Keys")
            setup_output = gr.Textbox(label="Status", interactive=False)

            save_keys_btn.click(
                transformer.save_api_keys,
                inputs=[openai_key, anthropic_key],
                outputs=setup_output
            )

        with gr.Tab("Transcribe"):
            gr.Markdown("### 📝 Transcribe Content")
            with gr.Row():
                with gr.Column(scale=1):
                     # Define file input accepting video/audio, multiple files allowed
                    file_input = gr.File(
                        label="Upload Media Files (Video/Audio)",
                        file_types=["video", "audio"],
                        file_count="multiple"
                    )
                with gr.Column(scale=1):
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
                status = gr.Textbox(label="Status Log", lines=10, interactive=False) # Increased lines for better visibility


            with gr.Row():
                 # Overall Progress Bar
                progress_bar = gr.Slider(
                    minimum=0,
                    maximum=100,
                    value=0,
                    label="Overall Progress (%)",
                    interactive=False
                )

            with gr.Row():
                # Current File Progress Bar
                current_file_progress = gr.Slider(
                    minimum=0,
                    maximum=100,
                    value=0,
                    label="Current File Progress (%)",
                    interactive=False
                )
                # Current File Name Display
                current_file_name = gr.Textbox(
                    label="Current Item",
                    value="",
                    interactive=False
                )

            # Spinner HTML/CSS (kept from previous merge)
            processing_indicator = gr.HTML(
                 """
                 <div id="processing-spinner" style="text-align: center; padding: 10px; background-color: #f0f0f0; border-radius: 5px; display: none;">
                      <style>
                      .spinner {
                          width: 20px; /* Smaller spinner */
                          height: 20px;
                          display: inline-block; /* Keep it inline */
                          vertical-align: middle; /* Align with text */
                          border: 3px solid #f3f3f3;
                          border-top: 3px solid #3498db;
                          border-radius: 50%;
                          animation: spin 1s linear infinite;
                          margin-right: 10px; /* Space between spinner and text */
                      }
                      @keyframes spin {
                          0% { transform: rotate(0deg); }
                          100% { transform: rotate(360deg); }
                      }
                      </style>
                      <div class="spinner"></div>
                      <span style="vertical-align: middle;">Processing...</span>
                 </div>
                 """,
                 visible=False # Initially hidden
             )


            # UI State Update Function
            def update_ui_on_process_start():
                return {
                    file_input: gr.update(interactive=False),
                    youtube_urls: gr.update(interactive=False),
                    transcribe_btn: gr.update(interactive=False),
                    cancel_btn: gr.update(visible=True),
                    processing_indicator: gr.update(visible=True),
                    progress_bar: gr.update(value=0), # Reset progress bars
                    current_file_progress: gr.update(value=0),
                    current_file_name: gr.update(value="")
                }

            def update_ui_on_process_end():
                 return {
                    file_input: gr.update(interactive=True),
                    youtube_urls: gr.update(interactive=True),
                    transcribe_btn: gr.update(interactive=True),
                    cancel_btn: gr.update(visible=False),
                    processing_indicator: gr.update(visible=False)
                 }


            # Connect Transcribe Button
            transcribe_event = transcribe_btn.click(
                update_ui_on_process_start,
                outputs=[file_input, youtube_urls, transcribe_btn, cancel_btn, processing_indicator, progress_bar, current_file_progress, current_file_name]
            ).then(
                transformer.process_content,
                inputs=[file_input, youtube_urls, language],
                outputs=[status, file_input, youtube_urls, processing_indicator, cancel_btn, progress_bar, current_file_progress, current_file_name],
                show_progress="hidden" # Use custom progress UI
            )
            # Update UI after process finishes or errors out
            transcribe_event.then(
                 update_ui_on_process_end,
                 outputs=[file_input, youtube_urls, transcribe_btn, cancel_btn, processing_indicator]
            )


            # Connect Cancel Button
            cancel_event = cancel_btn.click(
                 transformer.cancel_processing,
                 outputs=[status, file_input, youtube_urls, processing_indicator, cancel_btn, progress_bar, current_file_progress, current_file_name],
                 cancels=[transcribe_event] # Attempt to cancel the transcribe process
            )
            # Update UI after cancel finishes
            cancel_event.then(
                 update_ui_on_process_end,
                 outputs=[file_input, youtube_urls, transcribe_btn, cancel_btn, processing_indicator]
            )

    return app


if __name__ == "__main__":
    # Setup basic logging to console
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    app = create_interface()
    # Enable queue for handling multiple requests and long processes
    app.queue()
    # Launch the Gradio app
    app.launch() # share=True for public link (use with caution) 