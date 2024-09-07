from flask import Flask, request, jsonify, send_file, render_template
import os
import subprocess
import whisper
import srt
import uuid
from pydub import AudioSegment
from pydub.silence import detect_silence
from moviepy.editor import VideoFileClip
import numpy as np
from PIL import Image
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch


# Initialize Flask app
app = Flask(__name__)

# Set up upload folder
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Create upload folder if it doesn't exist
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Set maximum file size for uploads (e.g., 500 MB)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500 MB

# Load the image captioning model
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

# Image caption prediction function
def predict_step(image_paths):
    try:
        images = []
        for image_path in image_paths:
            i_image = Image.open(image_path)
            if i_image.mode != "RGB":
                i_image = i_image.convert(mode="RGB")
            images.append(i_image)

        pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(device)

        output_ids = model.generate(pixel_values, **gen_kwargs)

        preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        preds = [pred.strip() for pred in preds]
        return preds
    except Exception as e:
        print(f"Error during prediction step: {e}")
        return []

# Extract frames from silent periods in the video
def extract_silent_frames(video_path, silent_periods, output_folder):
    try:
        clip = VideoFileClip(video_path)
        frame_id = 0

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for start_time, end_time in silent_periods:
            for t in np.arange(start_time / 1000, end_time / 1000, 1.0):  # Extract a frame every second
                frame = clip.get_frame(t)
                image_path = os.path.join(output_folder, f"frame_{frame_id}.jpg")
                Image.fromarray(frame).save(image_path)
                frame_id += 1

        clip.close()
    except Exception as e:
        print(f"Error during frame extraction: {e}")

# Main video processing function
def process_video(video_path):
    try:
        # Generate unique IDs for filenames
        unique_id = str(uuid.uuid4())
        
        # Step 1: Convert video to audio
        audio_path = "output_audio.wav"
        ffmpeg_command = ['ffmpeg', '-i', video_path, '-b:a', '192K', audio_path, '-y']
        subprocess.run(ffmpeg_command, check=True)
        print("Audio extraction complete.")

        # Step 2: Detect silent periods in the audio
        audio = AudioSegment.from_wav(audio_path)
        silent_periods = detect_silence(audio, min_silence_len=1000, silence_thresh=audio.dBFS - 16)
        print(f"Silent periods detected: {silent_periods}")

        # Step 3: Transcribe audio to generate subtitles
        whisper_model = whisper.load_model("base")
        result = whisper_model.transcribe(audio_path)
        segments = result['segments']
        print(f"Transcription complete. Segments: {segments}")

        # Step 4: Create SRT subtitles
        subtitles = []
        for i, segment in enumerate(segments):
            start = segment['start']
            end = segment['end']
            text = segment['text']
            subtitle = srt.Subtitle(index=i + 1, start=srt.timedelta(seconds=start),
                                    end=srt.timedelta(seconds=end), content=text)
            subtitles.append(subtitle)

        # Step 5: Extract frames during silent periods and generate captions
        output_folder = f"frames_{unique_id}"
        extract_silent_frames(video_path, silent_periods, output_folder)
        frame_paths = [os.path.join(output_folder, frame) for frame in os.listdir(output_folder) if frame.endswith('.jpg')]
        captions = predict_step(frame_paths)
        print(f"Captions generated: {captions}")

        # Step 6: Combine subtitles and captions
        for idx, (start_time, end_time) in enumerate(silent_periods):
            if idx < len(captions):
                caption_text = captions[idx]
                subtitle = srt.Subtitle(index=len(subtitles) + 1 + idx, start=srt.timedelta(seconds=start_time / 1000),
                                        end=srt.timedelta(seconds=end_time / 1000), content=caption_text)
                subtitles.append(subtitle)

        # Step 7: Write combined SRT file
        srt_output = srt.compose(subtitles)
        srt_file_path = f"output_subtitles_{unique_id}.srt"
        with open(srt_file_path, "w") as srt_file:
            srt_file.write(srt_output)

        print(f"SRT file created successfully at {srt_file_path}")
        return srt_file_path
    except Exception as e:
        print(f"Error during video processing: {e}")
        raise

# Main route to display index page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle video upload, process it, and generate subtitles
@app.route('/generate-subtitles', methods=['POST'])
def generate_subtitles():
    try:
        video_file = request.files['video']
        if video_file:
            # Generate a unique filename for the video file
            unique_id = str(uuid.uuid4())

            # Create a unique filename
            video_filename = f"{unique_id}_{video_file.filename}"
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_filename)

              # Save the file with unique name
            video_file.save(video_path)
            print(f"Uploaded video saved at {video_path}")
            
            # Process video and generate subtitles
            srt_file_path = process_video(video_path)
            print(f"Generated SRT file at {srt_file_path}")
            
            # Check if SRT file was created
            if not os.path.isfile(srt_file_path):
                print("Failed to create subtitles.")
                return jsonify({'error': 'Failed to create subtitles.'}), 500
            
            # Send the file for download
            return send_file(srt_file_path, as_attachment=True)
        else:
            print("No video uploaded.")
            return jsonify({'error': 'No video uploaded'}), 400
    except Exception as e:
        print(f"Error during processing: {e}")
        return jsonify({'error': 'Failed to process video.'}), 500

if __name__ == "__main__":
    app.run(debug=True)   