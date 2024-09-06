document.addEventListener('DOMContentLoaded', function () {
    const videoFileInput = document.getElementById('video-file');
    const processBtn = document.getElementById('process-btn');
    const videoPreview = document.getElementById('video-preview');
    const progressBar = document.getElementById('progress-bar');
    const progressBarContainer = document.getElementById('progress-bar-container');
    const progressText = document.getElementById('progress-text');

    videoFileInput.addEventListener('change', (event) => {
        const file = event.target.files[0];
        if (file) {
            // Ensure the file is a video type
            if (!file.type.startsWith('video/')) {
                alert('Please upload a valid video file.');
                return;
            }

            // Create a URL for the video file
            const videoURL = URL.createObjectURL(file);

            // Set the video source to the URL
            videoPreview.src = videoURL;
            videoPreview.load(); // Reload the video player with the new source

            // Adjust the video dimensions based on its metadata
            videoPreview.onloadedmetadata = function () {
                const videoWidth = this.videoWidth;
                const videoHeight = this.videoHeight;

                // Set the width and height dynamically
                videoPreview.width = videoWidth;
                videoPreview.height = videoHeight;

                // Adjust the progress bar container width to match video width
                progressBarContainer.style.width = videoWidth + 'px';

                // Show the video preview and progress bar
                videoPreview.style.display = 'block';
                progressBar.style.display = 'none';
                progressText.textContent = '0%'; // Reset progress text
            };

            videoPreview.onerror = function () {
                alert('Failed to load the video. Please try a different file.');
                videoPreview.src = ''; // Clear the source to reset the player
            };
        }
    });

    processBtn.addEventListener('click', () => {
        const videoFile = videoFileInput.files[0];
        if (!videoFile) {
            alert('Please upload a video file.');
            return;
        }

        const formData = new FormData();
        formData.append('video', videoFile);

        const xhr = new XMLHttpRequest();
        xhr.open('POST', '/generate-subtitles', true);
        xhr.onload = function () {
            if (xhr.status === 200) {
                const response = JSON.parse(xhr.responseText);
                if (response.result) {
                    progressText.textContent = 'Subtitles ready';
                    progressBar.style.width = '100%';

                    // Provide a link to download the subtitles
                    const downloadLink = document.createElement('a');
                    downloadLink.href = `/download/${response.result.split('/').pop()}`;
                    downloadLink.textContent = 'Download Subtitles';
                    downloadLink.style.display = 'block';
                    document.body.appendChild(downloadLink);
                    downloadLink.click();    // Automtically click the download link 
                } else {
                    alert('Failed to process video.');
                }
            } else {
                alert('Failed to start processing video.');
            }
        };
        xhr.onerror = function () {
            alert('Request failed.');
        };
        xhr.upload.onprogress = function (event) {
            if (event.lengthComputable) {
                const percentage = (event.loaded / event.total) * 100;
                progressBar.style.width = percentage + '%';
                progressText.textContent = 'Uploading... ' + Math.round(percentage) + '%';
                progressBar.style.display = 'block';
            }
        };
        xhr.send(formData);
    });
});


