﻿# Auto-Subtitle-Generator

This Flask-based web application allows users to upload a video, extract silent frames, transcribe audio, and generate a combined subtitle file (SRT) that includes both audio transcriptions and captions for silent frames using a deep learning image captioning model.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Web Interface](#webinterface)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Features
- Upload video files and extract audio.
- Detect silent periods in audio and extract frames from those periods.
- Transcribe audio using Whisper model to generate subtitles.
- Generate image captions for frames extracted during silent periods using a VisionTransformer-GPT2 model.
- Combine audio transcriptions and image captions into a single SRT file for download.
- A user-friendly web interface built with HTML, CSS, and JavaScript.

## Installation

### Prerequisites
- Python 3.8 or higher
- `ffmpeg` installed on your system (used for video-to-audio conversion)

### Steps

1. Clone the repository:

- git clone https://github.com/samanvitha-k/Auto-Subtitle-Generator.git
- cd Auto-Subtitle-Generator

2. Install the python packages:
   pip install -r requirements.txt


3. Install ffmpeg :

- For Ubuntu :
  sudo apt update
  sudo apt install ffmpeg

- For Windows: 
  Download from FFmpeg official website, and add it to your system's PATH.

4. Start the application:
  python app.py
- The application will run locally at http://127.0.0.1:5000/.

- Optional GPU Support:
   If you have a CUDA-compatible GPU and PyTorch installed with CUDA, the application will automatically utilize GPU acceleration for faster model inference.

## Usage
- Open your browser and navigate to http://127.0.0.1:5000/.
- Upload a video file through the form.
- Wait for the app to process the video:
- It extracts audio from the video.
- Detects silent periods in the audio.
- Transcribes the audio to generate subtitles using the Whisper model.
- Extracts frames from silent periods and generates captions for those frames using a Vision-Transformer-GPT2 model.
- Once the processing is complete, the application provides a link to download the generated SRT subtitle file.

## Web Interface
- The app includes an HTML page for uploading video files.
- It also uses CSS for basic styling and JavaScript for handling client-side operations and interaction with the    server.



## Contributing
Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a new branch (git checkout -b feature/new-feature).
3. Make your changes and commit (git commit -m 'Add new feature').
4. Push to the branch (git push origin feature/new-feature).
5. Open a pull request.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Contact
If you have any questions or issues with the project, feel free to reach out:

- GitHub: your-username
- Email: your-email@example.com











