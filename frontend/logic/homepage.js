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
    const resultsScreen = document.getElementById('results-screen');
    const processingPreview = document.getElementById('processing-preview');

    const loadingCounter = document.getElementById('loading-counter');    
    const webSocket = new WebSocket('ws://127.0.0.1:8000/ws');
    webSocket.binaryType = 'blob';

    let currentUrl = null;
    let frameCount = 0;
    let totalFrames = '0';

    webSocket.onmessage = (event) => {
        if (typeof event.data === 'string') {
            const message = JSON.parse(event.data);
            totalFrames = message.frame_count;
            return;
        }
        if (currentUrl) {
            URL.revokeObjectURL(currentUrl);
        }

        currentUrl = URL.createObjectURL(event.data);

        processingPreview.src = currentUrl;
        frameCount++;
        loadingCounter.innerText = `${frameCount}/${totalFrames} frames processed`;

        if (frameCount == totalFrames) {
            processingText = document.getElementById('processing-text');
            processingText.innerText = 'Processing complete! Constructing results...';
        }
    };

    webSocket.onclose = () => {
        console.log('WebSocket closed');
    };

    webSocket.onerror = (error) => {
        console.log(error);
    };

    optionsScreen.classList.add('hidden');
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
        processingScreen.classList.add('hidden');
        resultsScreen.classList.remove('hidden');

    } catch (error) {
        console.error('Error sending data to backend:', error);
        alert(error.message);
    }

};

const features = [
    'RIGHT SHOULDER ANGLE',
    'LEFT SHOULDER ANGLE',
    'RIGHT ELBOW ANGLE',
    'LEFT ELBOW ANGLE',
    'RIGHT HIP ANGLE',
    'LEFT HIP ANGLE',
    'RIGHT KNEE ANGLE',
    'LEFT KNEE ANGLE',
    'RIGHT ANKLE ANGLE',
    'LEFT ANKLE ANGLE',
    'RIGHT SHOULDER ANGLE VELOCITY',
    'LEFT SHOULDER ANGLE VELOCITY',
    'RIGHT ELBOW ANGLE VELOCITY',
    'LEFT ELBOW ANGLE VELOCITY',
    'RIGHT HIP ANGLE VELOCITY',
    'LEFT HIP ANGLE VELOCITY',
    'RIGHT KNEE ANGLE VELOCITY',
    'LEFT KNEE ANGLE VELOCITY',
    'RIGHT ANKLE ANGLE VELOCITY',
    'LEFT ANKLE ANGLE VELOCITY',
    'NOSE X',
    'NOSE Y',
    'LEFT SHOULDER X',
    'LEFT SHOULDER Y',
    'RIGHT SHOULDER X',
    'RIGHT SHOULDER Y',
    'LEFT ELBOW X',
    'LEFT ELBOW Y',
    'RIGHT ELBOW X',
    'RIGHT ELBOW Y',
    'LEFT WRIST X',
    'LEFT WRIST Y',
    'RIGHT WRIST X',
    'RIGHT WRIST Y',
    'LEFT HIP X',
    'LEFT HIP Y',
    'RIGHT HIP X',
    'RIGHT HIP Y',
    'LEFT KNEE X',
    'LEFT KNEE Y',
    'RIGHT KNEE X',
    'RIGHT KNEE Y',
    'LEFT ANKLE X',
    'LEFT ANKLE Y',
    'RIGHT ANKLE X',
    'RIGHT ANKLE Y',
    'LEFT FOOT X',
    'LEFT FOOT Y',
    'RIGHT FOOT X',
    'RIGHT FOOT Y',
];

const resultsContainer = document.getElementById('results-container');
const graphSearchInput = document.getElementById('site-search');
graphSearchInput.addEventListener('input', () => {
    const query = graphSearchInput.value.toLowerCase().trim();
    const queryWords = query.split(/\s+/);
    const allGraphs = document.querySelectorAll('.graph');

    allGraphs.forEach(graph => {
        const graphId = graph.id.toLowerCase();
        const graphWords = graphId.split('-');
        if (graphWords.some(item => queryWords.includes(item)) || graphId.includes(query)) {
            graph.style.display = 'block';
        } 
        else {
            graph.style.display = 'none';
        }
    });

    const matchedFeatures = features.filter(item =>
        item.toLowerCase().includes(query)
    );

    document.querySelectorAll('.feature').forEach(feature => {
        feature.remove();
    });

    matchedFeatures.forEach(feature => {
        const newLink = document.createElement('a');
        const linkButton = document.createElement('div');

        newLink.href = `http://127.0.0.1:8000/outputs/graphs/Z-Scores/${encodeURIComponent(feature)}.png`;

        newLink.target = "_blank";
        newLink.rel = "noopener noreferrer";
        newLink.classList.add('feature')

        linkButton.classList.add('choice');
        linkButton.classList.add('featureButton');
        linkButton.textContent = feature;

        newLink.appendChild(linkButton);
        resultsContainer.appendChild(newLink);
    });
});