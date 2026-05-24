const inputVideo = document.getElementById('input-video');
const videoPreview = document.getElementById('video-preview');
const uploadText = document.getElementById('upload-text');

const mediapipeButton = document.getElementById('mediapipe');
const mmposeButton = document.getElementById('mmpose');
const yolo26Button = document.getElementById('yolo26');
const showProcessButton = document.getElementById('show-process');

const startButton = document.getElementById('start-button');

let videoFile = null;
let selectedModel = null;
let showProcess = false;

inputVideo.addEventListener('change', (event) => {
    const file = event.target.files[0];
    if (file) {
        videoFile = file;
        videoPreview.src = URL.createObjectURL(videoFile);
        videoPreview.style.display = 'block';
        uploadText.textContent = 'Video Preview:';
    }
});

mediapipeButton.addEventListener('click', () => {
   if (selectedModel === 'mediapipe') {
      mediapipeButton.style.backgroundColor = '';
      mediapipeButton.style.transform = 'scale(1)';
      selectedModel = null;
      return;
   }
   mediapipeButton.style.backgroundColor = 'rgb(117, 198, 198)';
   mediapipeButton.style.transform = 'scale(1.05)';
   yolo26Button.style.backgroundColor = '';
   yolo26Button.style.transform = 'scale(1)';
   mmposeButton.style.backgroundColor = '';
   mmposeButton.style.transform = 'scale(1)';
   selectedModel = 'mediapipe';
});

mmposeButton.addEventListener('click', () => {
    if (selectedModel === 'mmpose') {
       mmposeButton.style.backgroundColor = '';
       mmposeButton.style.transform = 'scale(1)';
       selectedModel = null;
       return;
    }
   mmposeButton.style.backgroundColor = 'rgba(70, 70, 226, 1)';
   mmposeButton.style.transform = 'scale(1.05)';
   mediapipeButton.style.backgroundColor = '';
   mediapipeButton.style.transform = 'scale(1)';
   yolo26Button.style.backgroundColor = '';
   yolo26Button.style.transform = 'scale(1)';
   selectedModel = 'mmpose';
});

yolo26Button.addEventListener('click', () => {
    if (selectedModel === 'yolo26') {
       yolo26Button.style.backgroundColor = '';
       yolo26Button.style.transform = 'scale(1)';
       selectedModel = null;
       return;
    }
   yolo26Button.style.backgroundColor = 'rgba(143, 157, 217, 1)';
   yolo26Button.style.transform = 'scale(1.05)';
   mediapipeButton.style.backgroundColor = '';
   mediapipeButton.style.transform = 'scale(1)';
   mmposeButton.style.backgroundColor = '';
   mmposeButton.style.transform = 'scale(1)';
   selectedModel = 'yolo26';
});

showProcessButton.addEventListener('click', () => {
   showProcess = !showProcess;
    if (showProcess) {
        showProcessButton.style.backgroundColor = 'rgba(59, 230, 219, 1)';
        showProcessButton.style.transform = 'scale(1.05)';
    } else {
        showProcessButton.style.backgroundColor = '';
        showProcessButton.style.transform = 'scale(1)';
    }
});

startButton.addEventListener('click', (event) => {
    event.preventDefault();
    if (!(videoFile && selectedModel)) {
        // alert('Please upload a video and select a model before starting the analysis.');
        return;
    }
    sendDataToBackend(videoFile, selectedModel, showProcess);
});

async function sendDataToBackend(videoFile, selectedModel, showProcess) {
    const processingScreen = document.getElementById('processing-screen');
    const optionsScreen = document.getElementById('options-screen');
    const processingText = document.getElementById('processing-text');
    const processingPreview = document.getElementById('processing-preview');
    alert(`Video File: ${videoFile.name}`);

    processingPreview.src = `http://127.0.0.1:8000/processing_preview?filename=${videoFile.name}`;

    optionsScreen.classList.add('hidden');
    optionsScreen.style.opacity = '0';
    processingScreen.classList.remove('hidden');

    const userInput = new FormData();
    userInput.append('video_file', videoFile);
    userInput.append('model', selectedModel);
    userInput.append('show', showProcess);

    try {
        const response = await fetch('http://127.0.0.1:8000/inputs', {
            method: 'POST',
            body: userInput
        });

        const results = await response.json();
        // add results displaying here

    } catch (error) {
        console.error('Error sending data to backend:', error);
        alert(error.message);
    }

}

