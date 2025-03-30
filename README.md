# Content Transformer

A powerful tool for transcribing media content using AI. This application can process both local media files and YouTube videos, supporting multiple languages and automatic language detection.

## Features

* Transcribe audio from local media files (MP3, WAV, MP4, AVI, MOV, MKV)
* Download and transcribe audio from YouTube videos
* Support for multiple languages:  
   * English (en)  
   * Italian (it)  
   * French (fr)  
   * German (de)  
   * Spanish (es)  
   * Auto-detection
* Real-time status updates during processing
* Markdown output with timestamps
* Clean and intuitive web interface

## Setup

1. Clone this repository
2. Create and activate a virtual environment:  
   ```bash
   python -m venv venv  
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```
4. Copy `.env.example` to `.env`:  
   ```bash
   cp .env.example .env
   ```
5. Add your API keys to the `.env` file
6. Run the application:  
   ```bash
   python app.py
   ```

## API Keys

This project requires the following API keys:

* Google API key: Used for Gemini model
* OpenAI API key: Used for GPT-4 model
* Anthropic API key: Used for Claude model
* YouTube API key (optional): Used for YouTube video processing

Never commit your actual API keys to version control! The `.env` file is already in `.gitignore` to prevent accidental commits.

## Usage

1. Open your web browser and navigate to `http://127.0.0.1:7860`
2. Choose between uploading local files or entering YouTube URLs
3. Select the language (or use auto-detection)
4. Click "Transcribe" and wait for the results
5. Find your transcriptions in markdown files with timestamps

## Requirements

* Python 3.8 or higher
* FFmpeg installed on your system
* Internet connection for YouTube downloads and API access

## Development

### Project Structure

```
content-transformer/
├── app.py              # Main application file
├── requirements.txt    # Python dependencies
├── .env.example       # Environment variables template
├── .gitignore         # Git ignore rules
├── README.md          # Project documentation
└── LICENSE            # MIT License
```

### Running Tests

```bash
# Activate virtual environment first
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Run tests (when implemented)
python -m pytest
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines

* Follow PEP 8 style guide for Python code
* Add docstrings to all functions and classes
* Update documentation for any new features
* Add tests for new functionality
* Keep commits clean and well-documented

## License

This project is licensed under the MIT License - see the LICENSE file for details. 