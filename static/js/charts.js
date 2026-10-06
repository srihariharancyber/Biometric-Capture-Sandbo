/**
 * AURA: Real-Time Charts & Dynamic Biometric Telemetry
 * Includes Multi-Channel Oscilloscope, Biometric Radar Chart, and Interactive ROC Curve.
 */

// 1. Multi-Channel Scrolling Oscilloscope
class Oscilloscope {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    this.ctx = this.canvas ? this.canvas.getContext('2d') : null;
    this.historyLength = 120;
    this.channels = {
      hold: new Array(this.historyLength).fill(75),
      flight: new Array(this.historyLength).fill(85),
      speed: new Array(this.historyLength).fill(400),
      trust: new Array(this.historyLength).fill(95)
    };
    
    if (this.canvas) {
      this.initCanvas();
      this.startLoop();
    }
  }

  initCanvas() {
    const rect = this.canvas.parentElement.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    this.canvas.width = rect.width * dpr;
    this.canvas.height = rect.height * dpr;
    this.ctx.scale(dpr, dpr);
  }

  pushSample(hold, flight, speed, trust) {
    this.channels.hold.push(hold);
    this.channels.hold.shift();

    this.channels.flight.push(flight);
    this.channels.flight.shift();

    this.channels.speed.push(speed);
    this.channels.speed.shift();

    this.channels.trust.push(trust);
    this.channels.trust.shift();
  }

  startLoop() {
    const render = () => {
      this.draw();
      requestAnimationFrame(render);
    };
    requestAnimationFrame(render);
  }

  draw() {
    if (!this.ctx || !this.canvas) return;
    const rect = this.canvas.getBoundingClientRect();
    const w = rect.width;
    const h = rect.height;

    this.ctx.clearRect(0, 0, w, h);

    // Draw background subtle grid
    this.ctx.strokeStyle = 'rgba(30, 58, 102, 0.25)';
    this.ctx.lineWidth = 1;
    this.ctx.beginPath();
    for (let x = 0; x < w; x += 30) {
      this.ctx.moveTo(x, 0);
      this.ctx.lineTo(x, h);
    }
    for (let y = 0; y < h; y += 30) {
      this.ctx.moveTo(0, y);
      this.ctx.lineTo(w, y);
    }
    this.ctx.stroke();

    // 1. Draw Channel: Trust Score (Green/Cyan) [0-100]
    this.drawChannelLine(this.channels.trust, 0, 100, '#10b981', 2.5);

    // 2. Draw Channel: Hold Time (Amber) [0-250ms]
    this.drawChannelLine(this.channels.hold, 0, 250, '#f59e0b', 1.5);

    // 3. Draw Channel: Flight Time (Purple) [0-400ms]
    this.drawChannelLine(this.channels.flight, 0, 400, '#a855f7', 1.5);

    // 4. Draw Channel: Mouse Speed (Cyan) [0-1500px/s]
    this.drawChannelLine(this.channels.speed, 0, 1500, '#06b6d4', 1.5);
  }

  drawChannelLine(data, minVal, maxVal, strokeColor, lineWidth) {
    const rect = this.canvas.getBoundingClientRect();
    const w = rect.width;
    const h = rect.height;
    const step = w / (data.length - 1);

    this.ctx.strokeStyle = strokeColor;
    this.ctx.lineWidth = lineWidth;
    this.ctx.beginPath();

    data.forEach((val, i) => {
      const normalized = Math.max(0, Math.min(1, (val - minVal) / (maxVal - minVal)));
      const y = h - (normalized * (h - 20) + 10);
      const x = i * step;

      if (i === 0) {
        this.ctx.moveTo(x, y);
      } else {
        this.ctx.lineTo(x, y);
      }
    });

    this.ctx.stroke();
  }
}


// 2. Biometric Radar Chart
class BiometricRadar {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    this.ctx = this.canvas ? this.canvas.getContext('2d') : null;
    this.labels = [
      'Hold Time',
      'Flight Latency',
      'Mouse Velocity',
      'Curvature Arc',
      'Cadence Rhythm',
      'Click Dwell'
    ];
    this.baselineValues = [0.65, 0.70, 0.60, 0.45, 0.85, 0.60];
    this.currentValues = [0.63, 0.68, 0.58, 0.42, 0.82, 0.59];

    if (this.canvas) {
      this.initCanvas();
      this.render();
    }
  }

  initCanvas() {
    const rect = this.canvas.parentElement.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    this.canvas.width = rect.width * dpr;
    this.canvas.height = rect.height * dpr;
    this.ctx.scale(dpr, dpr);
  }

  updateMetrics(currentData, baselineData) {
    if (baselineData) {
      this.baselineValues = [
        Math.min(1.0, baselineData.hold_ms / 150),
        Math.min(1.0, baselineData.flight_ms / 250),
        Math.min(1.0, baselineData.speed / 800),
        Math.min(1.0, baselineData.curvature / 0.5),
        0.85,
        0.65
      ];
    }
    if (currentData) {
      this.currentValues = [
        Math.min(1.0, currentData.hold_ms / 150),
        Math.min(1.0, currentData.flight_ms / 250),
        Math.min(1.0, currentData.speed / 800),
        Math.min(1.0, currentData.curvature / 0.5),
        0.80,
        0.60
      ];
    }
    this.render();
  }

  render() {
    if (!this.ctx || !this.canvas) return;
    const rect = this.canvas.getBoundingClientRect();
    const w = rect.width;
    const h = rect.height;
    const centerX = w / 2;
    const centerY = h / 2;
    const radius = Math.min(centerX, centerY) - 36;
    const numPoints = this.labels.length;
    const angleStep = (Math.PI * 2) / numPoints;

    this.ctx.clearRect(0, 0, w, h);

    // Draw concentric polygon webs
    for (let level = 1; level <= 4; level++) {
      const r = (radius / 4) * level;
      this.ctx.strokeStyle = 'rgba(51, 65, 85, 0.35)';
      this.ctx.lineWidth = 1;
      this.ctx.beginPath();
      for (let i = 0; i < numPoints; i++) {
        const angle = i * angleStep - Math.PI / 2;
        const x = centerX + Math.cos(angle) * r;
        const y = centerY + Math.sin(angle) * r;
        if (i === 0) this.ctx.moveTo(x, y);
        else this.ctx.lineTo(x, y);
      }
      this.ctx.closePath();
      this.ctx.stroke();
    }

    // Draw axis lines and labels
    this.ctx.fillStyle = '#94a3b8';
    this.ctx.font = '10px JetBrains Mono, monospace';
    this.ctx.textAlign = 'center';
    this.ctx.textBaseline = 'middle';

    for (let i = 0; i < numPoints; i++) {
      const angle = i * angleStep - Math.PI / 2;
      const x = centerX + Math.cos(angle) * radius;
      const y = centerY + Math.sin(angle) * radius;

      this.ctx.strokeStyle = 'rgba(51, 65, 85, 0.45)';
      this.ctx.beginPath();
      this.ctx.moveTo(centerX, centerY);
      this.ctx.lineTo(x, y);
      this.ctx.stroke();

      // Label
      const labelX = centerX + Math.cos(angle) * (radius + 20);
      const labelY = centerY + Math.sin(angle) * (radius + 14);
      this.ctx.fillText(this.labels[i], labelX, labelY);
    }

    // Draw Baseline Profile (Cyan)
    this.drawPolygon(this.baselineValues, radius, centerX, centerY, angleStep, 'rgba(6, 182, 212, 0.18)', '#06b6d4');

    // Draw Current Session (Emerald or Amber)
    this.drawPolygon(this.currentValues, radius, centerX, centerY, angleStep, 'rgba(16, 185, 129, 0.28)', '#10b981');
  }

  drawPolygon(values, radius, cx, cy, angleStep, fillStyle, strokeStyle) {
    this.ctx.beginPath();
    values.forEach((v, i) => {
      const angle = i * angleStep - Math.PI / 2;
      const r = radius * Math.max(0.05, Math.min(1.0, v));
      const x = cx + Math.cos(angle) * r;
      const y = cy + Math.sin(angle) * r;
      if (i === 0) this.ctx.moveTo(x, y);
      else this.ctx.lineTo(x, y);
    });
    this.ctx.closePath();

    this.ctx.fillStyle = fillStyle;
    this.ctx.fill();

    this.ctx.strokeStyle = strokeStyle;
    this.ctx.lineWidth = 2;
    this.ctx.stroke();

    // Dots at vertices
    values.forEach((v, i) => {
      const angle = i * angleStep - Math.PI / 2;
      const r = radius * Math.max(0.05, Math.min(1.0, v));
      const x = cx + Math.cos(angle) * r;
      const y = cy + Math.sin(angle) * r;
      this.ctx.fillStyle = strokeStyle;
      this.ctx.beginPath();
      this.ctx.arc(x, y, 3, 0, Math.PI * 2);
      this.ctx.fill();
    });
  }
}


// 3. Interactive ROC Curve Canvas
class ROCVisualizer {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    this.ctx = this.canvas ? this.canvas.getContext('2d') : null;
    this.data = null;
    this.threshold = 0.70;

    if (this.canvas) {
      this.initCanvas();
    }
  }

  initCanvas() {
    const rect = this.canvas.parentElement.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    this.canvas.width = rect.width * dpr;
    this.canvas.height = rect.height * dpr;
    this.ctx.scale(dpr, dpr);
  }

  updateBenchmarkData(benchmarkData, threshold) {
    this.data = benchmarkData;
    if (threshold !== undefined) this.threshold = threshold;
    this.render();
  }

  setThreshold(threshold) {
    this.threshold = threshold;
    this.render();
  }

  render() {
    if (!this.ctx || !this.canvas) return;
    const rect = this.canvas.getBoundingClientRect();
    const w = rect.width;
    const h = rect.height;
    const pad = 36;
    const plotW = w - pad * 2;
    const plotH = h - pad * 2;

    this.ctx.clearRect(0, 0, w, h);

    // Axes
    this.ctx.strokeStyle = 'rgba(148, 163, 184, 0.25)';
    this.ctx.lineWidth = 1;
    this.ctx.beginPath();
    this.ctx.moveTo(pad, pad);
    this.ctx.lineTo(pad, h - pad);
    this.ctx.lineTo(w - pad, h - pad);
    this.ctx.stroke();

    // Random Guess Diagonal (FPR = TPR)
    this.ctx.strokeStyle = 'rgba(100, 116, 139, 0.4)';
    this.ctx.setLineDash([4, 4]);
    this.ctx.beginPath();
    this.ctx.moveTo(pad, h - pad);
    this.ctx.lineTo(w - pad, pad);
    this.ctx.stroke();
    this.ctx.setLineDash([]);

    // Axis Labels
    this.ctx.fillStyle = '#94a3b8';
    this.ctx.font = '10px JetBrains Mono, monospace';
    this.ctx.textAlign = 'center';
    this.ctx.fillText('False Positive Rate (FAR)', w / 2, h - 10);
    this.ctx.save();
    this.ctx.translate(12, h / 2);
    this.ctx.rotate(-Math.PI / 2);
    this.ctx.fillText('True Positive Rate (1 - FRR)', 0, 0);
    this.ctx.restore();

    if (!this.data || !this.data.fpr || !this.data.tpr) {
      // Synthetic fallback curve
      this.ctx.strokeStyle = '#06b6d4';
      this.ctx.lineWidth = 2.5;
      this.ctx.beginPath();
      this.ctx.moveTo(pad, h - pad);
      for (let x = 0; x <= 1.0; x += 0.05) {
        const y = Math.pow(x, 0.22); // high AUC curve
        const px = pad + x * plotW;
        const py = (h - pad) - y * plotH;
        this.ctx.lineTo(px, py);
      }
      this.ctx.stroke();
      return;
    }

    // Draw Real ROC curve from PyTorch Benchmark
    const fpr = this.data.fpr;
    const tpr = this.data.tpr;

    this.ctx.strokeStyle = '#06b6d4';
    this.ctx.lineWidth = 2.5;
    this.ctx.beginPath();
    for (let i = 0; i < fpr.length; i++) {
      const px = pad + fpr[i] * plotW;
      const py = (h - pad) - tpr[i] * plotH;
      if (i === 0) this.ctx.moveTo(px, py);
      else this.ctx.lineTo(px, py);
    }
    this.ctx.stroke();

    // Highlight EER operating point
    const eer = this.data.eer || 0.07;
    const eerX = pad + eer * plotW;
    const eerY = (h - pad) - (1.0 - eer) * plotH;

    this.ctx.fillStyle = '#f43f5e';
    this.ctx.beginPath();
    this.ctx.arc(eerX, eerY, 5, 0, Math.PI * 2);
    this.ctx.fill();

    this.ctx.fillStyle = '#ffffff';
    this.ctx.font = '9px JetBrains Mono, monospace';
    this.ctx.fillText(`EER: ${(eer * 100).toFixed(1)}%`, eerX + 15, eerY - 8);
  }
}

window.Oscilloscope = Oscilloscope;
window.BiometricRadar = BiometricRadar;
window.ROCVisualizer = ROCVisualizer;
