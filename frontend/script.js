/**
 * AI Pitch Coach - Client-Side JavaScript
 * Handles WebSocket communication, audio recording, visualization, and UI updates
 */

// ============================================================================
// Configuration
// ============================================================================

const CONFIG = {
    // WebSocket settings
    WS_URL: `ws://${window.location.host}/ws`,

    // Audio settings
    CHUNK_INTERVAL_MS: 250,  // Send audio chunks every 250ms
    SAMPLE_RATE: 16000,      // 16kHz for speech recognition

    // Reconnection settings
    RECONNECT_DELAY_MS: 2000,
    MAX_RECONNECT_ATTEMPTS: 5,

    // Visualizer settings
    FFT_SIZE: 256,
    VISUALIZER_COLOR: '#4f46e5',
    VISUALIZER_COLOR_ACTIVE: '#22c55e'
};

// ============================================================================
// State Management
// ============================================================================

const state = {
    // WebSocket
    ws: null,
    wsConnected: false,
    reconnectAttempts: 0,

    // Recording
    isRecording: false,
    mediaRecorder: null,
    audioChunks: [],
    recordingStartTime: null,
    timerInterval: null,

    // Audio visualization
    audioContext: null,
    analyser: null,
    microphone: null,
    mediaStream: null,
    animationFrameId: null,

    // Audio playback queue
    audioQueue: [],
    isPlayingAudio: false,

    // UI elements (cached)
    elements: {}
};

// ============================================================================
// Initialization
// ============================================================================

document.addEventListener('DOMContentLoaded', () => {
    // Cache DOM elements
    cacheElements();

    // Set up event listeners
    setupEventListeners();

    // Initialize canvas
    initCanvas();

    // Check system status
    checkSystemStatus();

    // Initialize WebSocket connection
    initWebSocket();
});

function cacheElements() {
    state.elements = {
        // Status indicators
        sttStatus: document.getElementById('sttStatus'),
        llmStatus: document.getElementById('llmStatus'),
        ttsStatus: document.getElementById('ttsStatus'),
        wsStatus: document.getElementById('wsStatus'),

        // Recording
        recordButton: document.getElementById('recordButton'),
        recordIcon: document.getElementById('recordIcon'),
        recordText: document.getElementById('recordText'),
        recordingHint: document.getElementById('recordingHint'),
        recordingTimer: document.getElementById('recordingTimer'),

        // Visualizer
        visualizerContainer: document.getElementById('visualizerContainer'),
        audioVisualizer: document.getElementById('audioVisualizer'),
        audioLevelBar: document.getElementById('audioLevelBar'),

        // Transcript
        transcriptBox: document.getElementById('transcriptBox'),

        // Scores
        clarityScore: document.getElementById('clarityScore'),
        languageScore: document.getElementById('languageScore'),
        confidenceScore: document.getElementById('confidenceScore'),
        relevanceScore: document.getElementById('relevanceScore'),
        clarityFill: document.getElementById('clarityFill'),
        languageFill: document.getElementById('languageFill'),
        confidenceFill: document.getElementById('confidenceFill'),
        relevanceFill: document.getElementById('relevanceFill'),
        fillerCount: document.getElementById('fillerCount'),
        fillerDetails: document.getElementById('fillerDetails'),

        // Feedback
        feedbackBox: document.getElementById('feedbackBox'),

        // Audio
        audioPlayer: document.getElementById('audioPlayer'),

        // Loading
        loadingOverlay: document.getElementById('loadingOverlay'),
        loadingText: document.getElementById('loadingText')
    };
}

function setupEventListeners() {
    // Record button click
    state.elements.recordButton.addEventListener('click', toggleRecording);

    // Audio player events
    state.elements.audioPlayer.addEventListener('ended', playNextAudio);
    state.elements.audioPlayer.addEventListener('error', handleAudioError);

    // Handle window resize for canvas
    window.addEventListener('resize', resizeCanvas);
}

function initCanvas() {
    const canvas = state.elements.audioVisualizer;
    if (canvas) {
        resizeCanvas();
    }
}

function resizeCanvas() {
    const canvas = state.elements.audioVisualizer;
    const container = state.elements.visualizerContainer;
    if (canvas && container) {
        canvas.width = Math.min(container.offsetWidth - 48, 400);
        canvas.height = 80;
    }
}

// ============================================================================
// System Status
// ============================================================================

async function checkSystemStatus() {
    try {
        const response = await fetch('/api/status');
        const data = await response.json();

        // Update status indicators
        updateStatusIndicator('sttStatus', data.stt.status === 'ready');
        updateStatusIndicator('llmStatus', data.llm.available);
        updateStatusIndicator('ttsStatus', data.tts.status === 'ready');

    } catch (error) {
        console.error('Failed to check system status:', error);
        updateStatusIndicator('sttStatus', false);
        updateStatusIndicator('llmStatus', false);
        updateStatusIndicator('ttsStatus', false);
    }
}

function updateStatusIndicator(elementId, isConnected) {
    const element = state.elements[elementId];
    if (element) {
        element.classList.remove('connected', 'warning', 'error');
        element.classList.add(isConnected ? 'connected' : 'error');
    }
}

// ============================================================================
// WebSocket Connection
// ============================================================================

function initWebSocket() {
    if (state.ws && state.ws.readyState === WebSocket.OPEN) {
        return;
    }

    console.log('Connecting to WebSocket...');
    state.ws = new WebSocket(CONFIG.WS_URL);

    state.ws.onopen = () => {
        console.log('WebSocket connected');
        state.wsConnected = true;
        state.reconnectAttempts = 0;
        updateStatusIndicator('wsStatus', true);
    };

    state.ws.onclose = () => {
        console.log('WebSocket disconnected');
        state.wsConnected = false;
        updateStatusIndicator('wsStatus', false);

        // Attempt reconnection
        if (state.reconnectAttempts < CONFIG.MAX_RECONNECT_ATTEMPTS) {
            state.reconnectAttempts++;
            console.log(`Reconnecting... (attempt ${state.reconnectAttempts})`);
            setTimeout(initWebSocket, CONFIG.RECONNECT_DELAY_MS);
        }
    };

    state.ws.onerror = (error) => {
        console.error('WebSocket error:', error);
    };

    state.ws.onmessage = handleWebSocketMessage;
}

function sendMessage(message) {
    if (state.ws && state.ws.readyState === WebSocket.OPEN) {
        state.ws.send(JSON.stringify(message));
    }
}

// ============================================================================
// WebSocket Message Handling
// ============================================================================

function handleWebSocketMessage(event) {
    try {
        const message = JSON.parse(event.data);

        switch (message.type) {
            case 'status':
                handleStatusMessage(message);
                break;

            case 'transcript':
                handleTranscriptMessage(message);
                break;

            case 'filler_words':
                handleFillerWordsMessage(message);
                break;

            case 'analysis':
                handleAnalysisMessage(message);
                break;

            case 'scores':
                handleScoresMessage(message);
                break;

            case 'audio':
                handleAudioMessage(message);
                break;

            case 'complete':
                handleCompleteMessage();
                break;

            case 'error':
                handleErrorMessage(message);
                break;

            case 'pong':
                // Keep-alive response
                break;

            default:
                console.log('Unknown message type:', message.type);
        }

    } catch (error) {
        console.error('Failed to parse WebSocket message:', error);
    }
}

function handleStatusMessage(message) {
    // Only show loading overlay for processing steps, not during recording
    const msg = message.message || '';
    if (!msg.toLowerCase().includes('recording')) {
        showLoading(msg);
    }
}

function handleTranscriptMessage(message) {
    const transcriptBox = state.elements.transcriptBox;

    if (message.final) {
        transcriptBox.innerHTML = `<p class="transcript-text">${escapeHtml(message.text)}</p>`;
    } else {
        // Streaming transcript (if implemented)
        const existing = transcriptBox.querySelector('.transcript-text');
        if (existing) {
            existing.textContent += message.text;
        } else {
            transcriptBox.innerHTML = `<p class="transcript-text">${escapeHtml(message.text)}</p>`;
        }
    }
}

function handleFillerWordsMessage(message) {
    state.elements.fillerCount.textContent = message.count;

    // Format filler details
    if (message.details && Object.keys(message.details).length > 0) {
        const details = Object.entries(message.details)
            .map(([word, count]) => `"${word}": ${count}`)
            .join(', ');
        state.elements.fillerDetails.textContent = `(${details})`;
    } else {
        state.elements.fillerDetails.textContent = '(none detected)';
    }
}

function handleAnalysisMessage(message) {
    const feedbackBox = state.elements.feedbackBox;

    if (message.streaming) {
        // Append streaming content
        const existing = feedbackBox.querySelector('.feedback-text');
        if (existing) {
            // Remove cursor, add text, re-add cursor
            existing.innerHTML = existing.innerHTML.replace('<span class="streaming-cursor"></span>', '');
            existing.innerHTML += escapeHtml(message.text) + '<span class="streaming-cursor"></span>';
        } else {
            feedbackBox.innerHTML = `<div class="feedback-text">${escapeHtml(message.text)}<span class="streaming-cursor"></span></div>`;
        }
    } else if (message.complete) {
        // Remove streaming cursor
        const cursor = feedbackBox.querySelector('.streaming-cursor');
        if (cursor) {
            cursor.remove();
        }
    }
}

function handleScoresMessage(message) {
    const scores = message.data;

    // Update score values and bars
    updateScore('clarity', scores.clarity);
    updateScore('language', scores.language);
    updateScore('confidence', scores.confidence);
    updateScore('relevance', scores.topic_relevance);
}

function updateScore(name, value) {
    const scoreEl = state.elements[`${name}Score`];
    const fillEl = state.elements[`${name}Fill`];

    if (scoreEl && fillEl) {
        scoreEl.textContent = value > 0 ? `${value}/10` : '-';
        fillEl.style.width = `${(value / 10) * 100}%`;
    }
}

function handleAudioMessage(message) {
    // Add audio to queue
    state.audioQueue.push(message.data);

    // Start playing if not already
    if (!state.isPlayingAudio) {
        playNextAudio();
    }
}

function handleCompleteMessage() {
    hideLoading();
    state.elements.recordingHint.textContent = 'Recording complete! Click to record again.';
}

function handleErrorMessage(message) {
    hideLoading();
    console.error('Server error:', message.message);
    alert(`Error: ${message.message}`);
}

// ============================================================================
// Audio Playback
// ============================================================================

function playNextAudio() {
    if (state.audioQueue.length === 0) {
        state.isPlayingAudio = false;
        return;
    }

    state.isPlayingAudio = true;
    const audioBase64 = state.audioQueue.shift();

    // Create blob URL and play
    const audioBlob = base64ToBlob(audioBase64, 'audio/wav');
    const audioUrl = URL.createObjectURL(audioBlob);

    state.elements.audioPlayer.src = audioUrl;
    state.elements.audioPlayer.play().catch(error => {
        console.error('Audio playback error:', error);
        playNextAudio();
    });
}

function handleAudioError(error) {
    console.error('Audio player error:', error);
    playNextAudio();
}

function base64ToBlob(base64, mimeType) {
    const byteCharacters = atob(base64);
    const byteNumbers = new Array(byteCharacters.length);

    for (let i = 0; i < byteCharacters.length; i++) {
        byteNumbers[i] = byteCharacters.charCodeAt(i);
    }

    const byteArray = new Uint8Array(byteNumbers);
    return new Blob([byteArray], { type: mimeType });
}

// ============================================================================
// Audio Recording with Visualization
// ============================================================================

async function toggleRecording() {
    if (state.isRecording) {
        stopRecording();
    } else {
        await startRecording();
    }
}

async function startRecording() {
    try {
        // Request microphone access
        const stream = await navigator.mediaDevices.getUserMedia({
            audio: {
                channelCount: 1,
                sampleRate: CONFIG.SAMPLE_RATE,
                echoCancellation: true,
                noiseSuppression: true
            }
        });

        state.mediaStream = stream;

        // Reset state
        state.audioChunks = [];
        state.isRecording = true;
        state.recordingStartTime = Date.now();

        // Update UI
        updateRecordingUI(true);
        resetResultsUI();

        // Initialize audio visualization
        initAudioVisualization(stream);

        // Create MediaRecorder
        const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
            ? 'audio/webm;codecs=opus'
            : 'audio/webm';

        state.mediaRecorder = new MediaRecorder(stream, {
            mimeType: mimeType,
            audioBitsPerSecond: 16000
        });

        // Handle recorded data
        state.mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                state.audioChunks.push(event.data);
            }
        };

        // Handle recording stop
        state.mediaRecorder.onstop = async () => {
            // Stop visualization
            stopAudioVisualization();

            // Stop all tracks
            stream.getTracks().forEach(track => track.stop());

            // Process recorded audio
            await processRecordedAudio();
        };

        // Start recording with timeslice for chunked data
        state.mediaRecorder.start(CONFIG.CHUNK_INTERVAL_MS);

        // Send start message to server
        sendMessage({ type: 'start' });

        // Start timer
        startTimer();

    } catch (error) {
        console.error('Failed to start recording:', error);
        alert('Failed to access microphone. Please ensure microphone permissions are granted.');
    }
}

function stopRecording() {
    if (state.mediaRecorder && state.isRecording) {
        state.isRecording = false;
        state.mediaRecorder.stop();

        // Update UI
        updateRecordingUI(false);
        stopTimer();
        showLoading('Processing audio...');
    }
}

// ============================================================================
// Audio Visualization
// ============================================================================

function initAudioVisualization(stream) {
    try {
        // Create audio context
        state.audioContext = new (window.AudioContext || window.webkitAudioContext)();

        // Create analyser node
        state.analyser = state.audioContext.createAnalyser();
        state.analyser.fftSize = CONFIG.FFT_SIZE;
        state.analyser.smoothingTimeConstant = 0.8;

        // Connect microphone to analyser
        state.microphone = state.audioContext.createMediaStreamSource(stream);
        state.microphone.connect(state.analyser);

        // Start visualization loop
        visualize();

    } catch (error) {
        console.error('Failed to initialize audio visualization:', error);
    }
}

function stopAudioVisualization() {
    // Cancel animation frame
    if (state.animationFrameId) {
        cancelAnimationFrame(state.animationFrameId);
        state.animationFrameId = null;
    }

    // Disconnect and close audio context
    if (state.microphone) {
        state.microphone.disconnect();
        state.microphone = null;
    }

    if (state.audioContext) {
        state.audioContext.close();
        state.audioContext = null;
    }

    state.analyser = null;

    // Clear canvas
    const canvas = state.elements.audioVisualizer;
    if (canvas) {
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
    }

    // Reset level bar
    if (state.elements.audioLevelBar) {
        state.elements.audioLevelBar.style.width = '0%';
    }
}

function visualize() {
    if (!state.analyser || !state.isRecording) {
        return;
    }

    state.animationFrameId = requestAnimationFrame(visualize);

    const canvas = state.elements.audioVisualizer;
    const levelBar = state.elements.audioLevelBar;

    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const bufferLength = state.analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);

    // Get frequency data for waveform
    state.analyser.getByteFrequencyData(dataArray);

    // Clear canvas
    ctx.fillStyle = 'rgba(26, 26, 46, 0.3)';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Calculate average volume for level bar
    let sum = 0;
    for (let i = 0; i < bufferLength; i++) {
        sum += dataArray[i];
    }
    const average = sum / bufferLength;
    const volumePercent = Math.min(100, (average / 128) * 100);

    // Update level bar
    if (levelBar) {
        levelBar.style.width = `${volumePercent}%`;
    }

    // Draw waveform bars
    const barWidth = (canvas.width / bufferLength) * 2.5;
    let x = 0;

    for (let i = 0; i < bufferLength; i++) {
        const barHeight = (dataArray[i] / 255) * canvas.height;

        // Color based on intensity
        const intensity = dataArray[i] / 255;
        const hue = 250 - (intensity * 50); // Purple to green
        const saturation = 70 + (intensity * 30);
        const lightness = 50 + (intensity * 20);

        ctx.fillStyle = `hsl(${hue}, ${saturation}%, ${lightness}%)`;

        // Draw bar from center
        const y = (canvas.height - barHeight) / 2;
        ctx.fillRect(x, y, barWidth - 1, barHeight);

        x += barWidth;
    }

    // Draw center line
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.2)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, canvas.height / 2);
    ctx.lineTo(canvas.width, canvas.height / 2);
    ctx.stroke();
}

// ============================================================================
// Audio Processing
// ============================================================================

async function processRecordedAudio() {
    if (state.audioChunks.length === 0) {
        hideLoading();
        alert('No audio recorded. Please try again.');
        return;
    }

    try {
        // Combine all chunks into a single blob
        const audioBlob = new Blob(state.audioChunks, { type: 'audio/webm' });

        // Convert to WAV for better compatibility
        const wavBlob = await convertToWav(audioBlob);

        // Convert to base64
        const base64 = await blobToBase64(wavBlob);

        // Send to server
        sendMessage({
            type: 'audio',
            data: base64.split(',')[1]  // Remove data URL prefix
        });

        // Signal end of recording
        sendMessage({ type: 'stop' });

    } catch (error) {
        console.error('Failed to process audio:', error);
        hideLoading();
        alert('Failed to process audio. Please try again.');
    }
}

async function convertToWav(blob) {
    // Create an AudioContext to process the audio
    const audioContext = new (window.AudioContext || window.webkitAudioContext)({
        sampleRate: CONFIG.SAMPLE_RATE
    });

    try {
        const arrayBuffer = await blob.arrayBuffer();
        const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);

        // Convert to WAV
        const wavBuffer = audioBufferToWav(audioBuffer);
        return new Blob([wavBuffer], { type: 'audio/wav' });

    } catch (error) {
        console.warn('WAV conversion failed, using original format:', error);
        return blob;
    } finally {
        audioContext.close();
    }
}

function audioBufferToWav(audioBuffer) {
    const numChannels = 1;  // Mono
    const sampleRate = audioBuffer.sampleRate;
    const format = 1;  // PCM
    const bitDepth = 16;

    const channelData = audioBuffer.getChannelData(0);
    const samples = new Int16Array(channelData.length);

    // Convert float samples to int16
    for (let i = 0; i < channelData.length; i++) {
        const s = Math.max(-1, Math.min(1, channelData[i]));
        samples[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
    }

    const dataSize = samples.length * 2;
    const buffer = new ArrayBuffer(44 + dataSize);
    const view = new DataView(buffer);

    // WAV header
    writeString(view, 0, 'RIFF');
    view.setUint32(4, 36 + dataSize, true);
    writeString(view, 8, 'WAVE');
    writeString(view, 12, 'fmt ');
    view.setUint32(16, 16, true);  // Subchunk1Size
    view.setUint16(20, format, true);
    view.setUint16(22, numChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * numChannels * bitDepth / 8, true);
    view.setUint16(32, numChannels * bitDepth / 8, true);
    view.setUint16(34, bitDepth, true);
    writeString(view, 36, 'data');
    view.setUint32(40, dataSize, true);

    // Write samples
    const offset = 44;
    for (let i = 0; i < samples.length; i++) {
        view.setInt16(offset + i * 2, samples[i], true);
    }

    return buffer;
}

function writeString(view, offset, string) {
    for (let i = 0; i < string.length; i++) {
        view.setUint8(offset + i, string.charCodeAt(i));
    }
}

function blobToBase64(blob) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onloadend = () => resolve(reader.result);
        reader.onerror = reject;
        reader.readAsDataURL(blob);
    });
}

// ============================================================================
// UI Updates
// ============================================================================

function updateRecordingUI(isRecording) {
    const button = state.elements.recordButton;
    const text = state.elements.recordText;
    const hint = state.elements.recordingHint;
    const visualizer = state.elements.visualizerContainer;

    if (isRecording) {
        button.classList.add('recording');
        text.textContent = 'Stop Recording';
        hint.textContent = 'Speak clearly into your microphone...';

        // Show visualizer
        if (visualizer) {
            visualizer.classList.add('visible');
            resizeCanvas();
        }
    } else {
        button.classList.remove('recording');
        text.textContent = 'Start Recording';
        hint.textContent = 'Click the button and speak your pitch clearly';

        // Hide visualizer
        if (visualizer) {
            visualizer.classList.remove('visible');
        }
    }
}

function resetResultsUI() {
    // Reset transcript
    state.elements.transcriptBox.innerHTML = '<p class="placeholder">Your speech will appear here...</p>';

    // Reset scores
    ['clarity', 'language', 'confidence', 'relevance'].forEach(name => {
        updateScore(name, 0);
    });

    // Reset filler words
    state.elements.fillerCount.textContent = '0';
    state.elements.fillerDetails.textContent = '';

    // Reset feedback
    state.elements.feedbackBox.innerHTML = '<p class="placeholder">AI analysis will appear here after you record...</p>';

    // Clear audio queue
    state.audioQueue = [];
    state.isPlayingAudio = false;
}

function startTimer() {
    const timerEl = state.elements.recordingTimer;

    state.timerInterval = setInterval(() => {
        const elapsed = Math.floor((Date.now() - state.recordingStartTime) / 1000);
        const minutes = Math.floor(elapsed / 60).toString().padStart(2, '0');
        const seconds = (elapsed % 60).toString().padStart(2, '0');
        timerEl.textContent = `${minutes}:${seconds}`;
    }, 1000);
}

function stopTimer() {
    if (state.timerInterval) {
        clearInterval(state.timerInterval);
        state.timerInterval = null;
    }
}

function showLoading(message = 'Processing...') {
    state.elements.loadingText.textContent = message;
    state.elements.loadingOverlay.classList.add('visible');
}

function hideLoading() {
    state.elements.loadingOverlay.classList.remove('visible');
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ============================================================================
// Keep-Alive
// ============================================================================

setInterval(() => {
    if (state.wsConnected) {
        sendMessage({ type: 'ping' });
    }
}, 30000);  // Ping every 30 seconds
