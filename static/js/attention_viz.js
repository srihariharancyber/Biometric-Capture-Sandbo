/**
 * AURA: Self-Attention Deep Network Heatmap & Architecture Visualizer
 * Renders per-head self-attention weight matrices (32x32) on high-performance Canvas.
 */
class AttentionVisualizer {
  constructor(canvasId, tooltipId) {
    this.canvas = document.getElementById(canvasId);
    this.tooltip = document.getElementById(tooltipId);
    this.ctx = this.canvas ? this.canvas.getContext('2d') : null;
    this.currentHead = 'mean'; // 'head_0', 'head_1', 'head_2', 'head_3', 'mean'
    this.matrices = null;
    this.gridSize = 32;
    this.hoverCell = null;
    
    if (this.canvas) {
      this.initCanvas();
      this.initEvents();
    }
  }

  initCanvas() {
    const rect = this.canvas.parentElement.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    this.canvas.width = rect.width * dpr;
    this.canvas.height = rect.height * dpr;
    this.ctx.scale(dpr, dpr);
    this.renderFallback();
  }

  initEvents() {
    this.canvas.addEventListener('mousemove', (e) => {
      const rect = this.canvas.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;

      const cellSizeX = rect.width / this.gridSize;
      const cellSizeY = rect.height / this.gridSize;

      const col = Math.floor(x / cellSizeX);
      const row = Math.floor(y / cellSizeY);

      if (row >= 0 && row < this.gridSize && col >= 0 && col < this.gridSize) {
        this.hoverCell = { row, col };
        this.showTooltip(e.clientX, e.clientY, row, col);
      } else {
        this.hoverCell = null;
        this.hideTooltip();
      }
      this.render();
    });

    this.canvas.addEventListener('mouseleave', () => {
      this.hoverCell = null;
      this.hideTooltip();
      this.render();
    });

    window.addEventListener('resize', () => {
      if (this.canvas && this.canvas.parentElement) {
        const rect = this.canvas.parentElement.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;
        this.canvas.width = rect.width * dpr;
        this.canvas.height = rect.height * dpr;
        this.ctx.scale(dpr, dpr);
        this.render();
      }
    });
  }

  setHead(headKey) {
    this.currentHead = headKey;
    this.render();
  }

  updateAttention(attentionMatrices) {
    this.matrices = attentionMatrices;
    this.render();
  }

  getColorForWeight(w) {
    // w is normalized [0.0 to 1.0]
    // Interpolate through Cyber Palette: Navy (0) -> Cyan (0.25) -> Emerald (0.5) -> Amber (0.75) -> Crimson (1.0)
    const val = Math.min(1.0, Math.max(0.0, w * 5.0)); // amplify contrast for visual clarity
    
    if (val < 0.25) {
      const t = val / 0.25;
      return `rgb(${Math.round(10 + t * 4)}, ${Math.round(17 + t * 90)}, ${Math.round(40 + t * 140)})`;
    } else if (val < 0.5) {
      const t = (val - 0.25) / 0.25;
      return `rgb(${Math.round(14 + t * 2)}, ${Math.round(107 + t * 78)}, ${Math.round(180 - t * 51)})`;
    } else if (val < 0.75) {
      const t = (val - 0.5) / 0.25;
      return `rgb(${Math.round(16 + t * 229)}, ${Math.round(185 - t * 27)}, ${Math.round(129 - t * 118)})`;
    } else {
      const t = (val - 0.75) / 0.25;
      return `rgb(${Math.round(245 - t * 1)}, ${Math.round(158 - t * 95)}, ${Math.round(11 + t * 83)})`;
    }
  }

  render() {
    if (!this.ctx || !this.canvas) return;

    const rect = this.canvas.getBoundingClientRect();
    const width = rect.width;
    const height = rect.height;

    this.ctx.clearRect(0, 0, width, height);

    const matrix = this.matrices ? this.matrices[this.currentHead] : null;

    if (!matrix || !matrix.length) {
      this.renderFallback();
      return;
    }

    const rows = matrix.length;
    const cols = matrix[0].length;
    this.gridSize = rows;

    const cellW = width / cols;
    const cellH = height / rows;

    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const weight = matrix[r][c];
        this.ctx.fillStyle = this.getColorForWeight(weight);
        this.ctx.fillRect(c * cellW, r * cellH, cellW - 0.5, cellH - 0.5);
      }
    }

    // Highlight hovered cell and crosshair
    if (this.hoverCell) {
      const { row, col } = this.hoverCell;
      this.ctx.strokeStyle = '#38bdf8';
      this.ctx.lineWidth = 1.5;
      this.ctx.strokeRect(col * cellW, row * cellH, cellW, cellH);

      // Faint crosshair
      this.ctx.fillStyle = 'rgba(56, 189, 248, 0.15)';
      this.ctx.fillRect(0, row * cellH, width, cellH);
      this.ctx.fillRect(col * cellW, 0, cellW, height);
    }
  }

  renderFallback() {
    if (!this.ctx || !this.canvas) return;
    const rect = this.canvas.getBoundingClientRect();
    const w = rect.width;
    const h = rect.height;

    this.ctx.fillStyle = '#0a0f1d';
    this.ctx.fillRect(0, 0, w, h);

    // Subtle synthetic placeholder grid
    const cols = 32;
    const cellW = w / cols;
    const cellH = h / cols;

    for (let r = 0; r < cols; r++) {
      for (let c = 0; c < cols; c++) {
        const dist = Math.abs(r - c);
        const synthW = Math.max(0.01, (1.0 / (dist + 1)) * 0.4);
        this.ctx.fillStyle = this.getColorForWeight(synthW);
        this.ctx.fillRect(c * cellW, r * cellH, cellW - 0.5, cellH - 0.5);
      }
    }
  }

  showTooltip(clientX, clientY, row, col) {
    if (!this.tooltip || !this.matrices) return;
    const matrix = this.matrices[this.currentHead];
    if (!matrix || !matrix[row]) return;

    const weight = matrix[row][col];
    this.tooltip.innerHTML = `
      <div style="font-weight:700; color:#38bdf8;">Query Token [${row}] &rarr; Key Token [${col}]</div>
      <div>Attention Weight: <span style="color:#10b981; font-weight:700;">${weight.toFixed(4)}</span></div>
      <div style="font-size:0.65rem; color:#94a3b8; margin-top:2px;">Head: ${this.currentHead.toUpperCase()}</div>
    `;

    const containerRect = this.canvas.parentElement.getBoundingClientRect();
    const left = clientX - containerRect.left + 15;
    const top = clientY - containerRect.top + 10;

    this.tooltip.style.left = `${Math.min(containerRect.width - 180, Math.max(10, left))}px`;
    this.tooltip.style.top = `${Math.min(containerRect.height - 70, Math.max(10, top))}px`;
    this.tooltip.style.display = 'block';
  }

  hideTooltip() {
    if (this.tooltip) {
      this.tooltip.style.display = 'none';
    }
  }
}

window.AttentionVisualizer = AttentionVisualizer;
