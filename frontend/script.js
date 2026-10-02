const fileInput = document.getElementById('audio-file');
const browseBtn = document.getElementById('browse-btn');
const dropZone = document.getElementById('drop-zone');
const selectedFile = document.getElementById('selected-file');
const fileName = document.getElementById('file-name');
const predictBtn = document.getElementById('predict-btn');
const resetBtn = document.getElementById('reset-btn');
const status = document.getElementById('status');
const statusText = document.getElementById('status-text');
const chipsContainer = document.getElementById('emotion-chips');
const result = document.getElementById('result');
const resultEmotion = document.getElementById('result-emotion');
const resultEmotionIcon = document.getElementById('result-emotion-icon');
const resultConfidence = document.getElementById('result-confidence');
const resultFile = document.getElementById('result-file');
const probabilities = document.getElementById('probabilities');
const errorBox = document.getElementById('error');

// const emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad'];
const emotions = [
  {
      id: 'angry',
      label: 'Angry',
      icon: '<i class="fa-solid fa-face-angry"></i>'
  },
  {
      id: 'happy',
      label: 'Happy',
      icon: '<i class="fa-solid fa-face-smile"></i>'
  },
  {
      id: 'neutral',
      label: 'Neutral',
      icon: '<i class="fa-solid fa-face-meh"></i>'
  },
  {
      id: 'sad',
      label: 'Sad',
      icon: '<i class="fa-solid fa-face-sad-tear"></i>'
  }
];


let selectedAudio = null;
let shuffleTimer = null;
let shuffleIndex = 0;

function showError(message) {
  errorBox.textContent = message;
  errorBox.classList.remove('hidden');
}

function clearError() {
  errorBox.classList.add('hidden');
  errorBox.textContent = '';
}

function setFile(file) {
  clearError();
  if (!file) return;
  if (!file.type.startsWith('audio/') && !/\.(wav|mp3|flac|ogg|m4a|aac)$/i.test(file.name)) {
    showError('Please select an audio file.');
    return;
  }
  if (file.size > 25 * 1024 * 1024) {
    showError('Audio file must be smaller than 25 MB.');
    return;
  }
  selectedAudio = file;
  fileName.textContent = file.name;
  selectedFile.classList.remove('hidden');
  result.classList.add('hidden');
}

function renderChips(active = '') {
  chipsContainer.innerHTML = emotions.map(emotion =>
      `<span class="chip ${emotion.id === active ? 'active' : ''}" data-emotion="${emotion.id}">
          ${emotion.icon}
          <span>${emotion.label}</span>
      </span>`
  ).join('');
}

function startShuffle() {
  stopShuffle();
  shuffleIndex = 0;
  shuffleTimer = setInterval(() => {
      const emotion = emotions[shuffleIndex % emotions.length];
      renderChips(emotion.id);
      shuffleIndex += 1;
  }, 320);
}

function stopShuffle(finalEmotion = '') {
  if (shuffleTimer) clearInterval(shuffleTimer);
  shuffleTimer = null;
  chipsContainer.innerHTML = emotions.map(emotion =>
      `<span class="chip ${emotion.id === finalEmotion ? 'active' : ''}" data-emotion="${emotion.id}">
          ${emotion.icon}
          <span>${emotion.label}</span>
      </span>`
  ).join('');
}

async function predict() {
  if (!selectedAudio) return;
  clearError();
  predictBtn.disabled = true;
  predictBtn.textContent = 'Analyzing…';
  result.classList.add('hidden');
  status.classList.remove('hidden');
  statusText.textContent = 'Extracting acoustic features…';
  startShuffle();

  const formData = new FormData();
  formData.append('file', selectedAudio);

  try {
    const response = await fetch('/api/predict', { method: 'POST', body: formData });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || 'Prediction failed.');

    statusText.textContent = 'Prediction complete.';
    stopShuffle(data.emotion);
    showResult(data);
  } catch (error) {
    stopShuffle();
    status.classList.add('hidden');
    showError(error.message || 'Something went wrong while analyzing the audio.');
  } finally {
    predictBtn.disabled = false;
    predictBtn.textContent = 'Analyze audio';
  }
}

function showResult(data) {
  const emotion = emotions.find(item => item.id === data.emotion);
  resultEmotion.textContent = emotion ? emotion.label : data.emotion;
  resultEmotionIcon.innerHTML = emotion ? emotion.icon : '';

  resultConfidence.textContent = `${(data.confidence * 100).toFixed(1)}% confidence`;
  resultFile.textContent = data.filename || selectedAudio.name;
  probabilities.innerHTML = Object.entries(data.probabilities)
    .sort((a, b) => b[1] - a[1])
    .map(([emotionId, probability]) => {
      const emotion = emotions.find(item => item.id === emotionId);
      const label = emotion ? emotion.label : emotionId;

      return `
          <div class="prob-row">
              <span>${label}</span>
              <div class="bar">
                  <i style="width:${probability * 100}%"></i>
              </div>
              <span>${(probability * 100).toFixed(1)}%</span>
          </div>
      `;
    })
    .join('');
  result.classList.remove('hidden');
}

function reset() {
  selectedAudio = null;
  fileInput.value = '';
  selectedFile.classList.add('hidden');
  status.classList.add('hidden');
  result.classList.add('hidden');
  clearError();
  stopShuffle();
  document.getElementById('demo').scrollIntoView({ behavior: 'smooth' });
}

browseBtn.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('click', (event) => {
  if (!event.target.closest('button')) fileInput.click();
});
dropZone.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' || event.key === ' ') fileInput.click();
});
fileInput.addEventListener('change', () => setFile(fileInput.files[0]));
predictBtn.addEventListener('click', predict);
resetBtn.addEventListener('click', reset);
['dragenter', 'dragover'].forEach(eventName => dropZone.addEventListener(eventName, e => {
  e.preventDefault(); dropZone.classList.add('dragging');
}));
['dragleave', 'drop'].forEach(eventName => dropZone.addEventListener(eventName, e => {
  e.preventDefault(); dropZone.classList.remove('dragging');
}));
dropZone.addEventListener('drop', e => setFile(e.dataTransfer.files[0]));

renderChips();
