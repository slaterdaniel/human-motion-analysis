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