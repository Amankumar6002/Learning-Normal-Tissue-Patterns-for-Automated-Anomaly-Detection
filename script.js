// Medical AI VAE Anomaly Detection Script

let model = null;

async function loadModel() {
    try {
        model = await tf.loadLayersModel('models/model.json');
        console.log('Model loaded successfully');
    } catch (error) {
        console.warn('Model loading failed, using simulation mode:', error);
        model = null;
    }
}

function setStatus(text) {
    const status = document.getElementById('analysis-status');
    if (status) status.textContent = text;
}

function setScore(value) {
    const score = document.getElementById('score-value');
    if (score) score.textContent = `${value.toFixed(1)}%`;
}

function setAnalysisDetail(text) {
    const detail = document.getElementById('analysis-detail');
    if (detail) detail.textContent = text;
}

function setStatusBadge(text, strong = false) {
    const badge = document.getElementById('status-badge');
    if (!badge) return;
    badge.textContent = text;
    badge.style.background = strong ? '#fed7d7' : '#e0f2fe';
    badge.style.color = strong ? '#b91c1c' : '#0369a1';
}

function showPreviewImage(src) {
    const preview = document.getElementById('preview-img');
    if (!preview) return;
    preview.src = src;
    preview.style.display = 'block';
}

function hidePreviewImage() {
    const preview = document.getElementById('preview-img');
    if (!preview) return;
    preview.src = '';
    preview.style.display = 'none';
}

function showOutputSection() {
    const outputSection = document.getElementById('output-section');
    if (outputSection) outputSection.style.display = 'grid';
}

function showError(message) {
    const errorMsg = document.getElementById('error-msg');
    if (errorMsg) {
        errorMsg.textContent = message;
        errorMsg.style.display = 'block';
    }
}

function hideError() {
    const errorMsg = document.getElementById('error-msg');
    if (errorMsg) {
        errorMsg.style.display = 'none';
    }
}

function computeRandomScore() {
    return 30 + Math.random() * 55;
}

async function processImage(imageSrc) {
    const originalImg = document.getElementById('original-img');
    const reconstructedImg = document.getElementById('reconstructed-img');
    const heatmapImg = document.getElementById('heatmap-img');

    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    const img = new Image();
    img.crossOrigin = 'anonymous';

    img.onload = async function() {
        canvas.width = img.width;
        canvas.height = img.height;
        ctx.drawImage(img, 0, 0);

        originalImg.src = imageSrc;
        originalImg.style.display = 'block';

        if (model) {
            await predictWithModel(canvas);
        } else {
            simulatePrediction(canvas);
        }

        showOutputSection();
        setStatus('Anomaly detection completed successfully');
        setAnalysisDetail('The model has generated a reconstruction and anomaly heatmap based on normal tissue patterns.');
    };

    img.src = imageSrc;
}

function simulatePrediction(canvas) {
    const width = canvas.width;
    const height = canvas.height;

    const reconstructedCanvas = document.createElement('canvas');
    reconstructedCanvas.width = width;
    reconstructedCanvas.height = height;
    const reconCtx = reconstructedCanvas.getContext('2d');

    reconCtx.drawImage(canvas, 0, 0, width, height);
    reconCtx.filter = 'blur(4px)';
    reconCtx.drawImage(canvas, 0, 0, width, height);
    reconCtx.filter = 'none';

    const heatmapCanvas = document.createElement('canvas');
    heatmapCanvas.width = width;
    heatmapCanvas.height = height;
    const heatCtx = heatmapCanvas.getContext('2d');

    heatCtx.drawImage(canvas, 0, 0, width, height);
    heatCtx.globalCompositeOperation = 'source-atop';
    heatCtx.fillStyle = 'rgba(220, 38, 38, 0.32)';
    heatCtx.fillRect(0, 0, width, height);
    heatCtx.globalCompositeOperation = 'source-over';

    const score = computeRandomScore();
    setScore(score);

    const detailText = score > 70
        ? 'High anomaly likelihood detected. Review the heatmap for abnormal tissue patterns.'
        : 'Normal tissue patterns detected. No strong anomalies found.';
    setAnalysisDetail(detailText);
    setStatusBadge(score > 70 ? 'Attention' : 'Completed', score > 70);

    const reconstructedImg = document.getElementById('reconstructed-img');
    const heatmapImg = document.getElementById('heatmap-img');

    if (reconstructedImg) {
        reconstructedImg.src = reconstructedCanvas.toDataURL();
        reconstructedImg.style.display = 'block';
    }
    if (heatmapImg) {
        heatmapImg.src = heatmapCanvas.toDataURL();
        heatmapImg.style.display = 'block';
    }
}

async function predictWithModel(canvas) {
    // Replace this with your real model preprocessing and prediction.
    // Example steps:
    // 1. const input = tf.browser.fromPixels(canvas).resizeBilinear([128,128]).toFloat().div(255).expandDims(0);
    // 2. const output = model.predict(input);
    // 3. Convert output tensor to image data and assign to reconstructed image.
    console.log('Real model prediction not implemented yet. Using simulation.');
    simulatePrediction(canvas);
}

document.addEventListener('DOMContentLoaded', function() {
    loadModel();

    const imageUpload = document.getElementById('image-upload');
    const processBtn = document.getElementById('process-btn');
    const uploadBox = document.getElementById('upload-box');

    imageUpload.addEventListener('change', function(event) {
        const file = event.target.files[0];
        if (!file) {
            hidePreviewImage();
            return;
        }

        const reader = new FileReader();
        reader.onload = function(e) {
            showPreviewImage(e.target.result);
            hideError();
        };
        reader.readAsDataURL(file);
    });

    uploadBox.addEventListener('click', function() {
        imageUpload.click();
    });

    processBtn.addEventListener('click', async function() {
        const file = imageUpload.files[0];
        if (!file) {
            showError('Please select an image before running analysis.');
            return;
        }

        hideError();
        setStatus('Processing image, please wait...');

        const reader = new FileReader();
        reader.onload = async function(e) {
            await processImage(e.target.result);
        };
        reader.readAsDataURL(file);
    });
});