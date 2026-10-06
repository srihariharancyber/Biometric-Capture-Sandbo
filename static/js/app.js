/**
 * AURA: Application Controller
 * Orchestrates real-time telemetry, continuous authentication scoring,
 * interactive typing/mouse arenas, adversarial simulations, and model training.
 */

document.addEventListener('DOMContentLoaded', () => {
  // Initialize Subsystems
  const audio = window.cyberAudio;
  const oscilloscope = new Oscilloscope('oscilloscope-canvas');
  const radar = new BiometricRadar('radar-canvas');
  const rocViz = new ROCVisualizer('roc-canvas');
  const attentionViz = new AttentionVisualizer('heatmap-canvas', 'heatmap-tooltip');

  let currentThreshold = 0.70;
  let activeProfileId = 0;
  let sessionTrust = 95.0;
  let isLocked = false;
  let keystrokeCount = 0;
  let startTime = Date.now();

  // 1. Initialize Biometric Tracker
  const tracker = new BiometricTracker(async (batch) => {
    if (isLocked) return;
    try {
      const res = await fetch('/api/infer', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          events: batch,
          threshold: currentThreshold
        })
      });
      const data = await res.json();
      handleInferenceResult(data);
    } catch (err) {
      console.warn('Inference request failed:', err);
    }
  });

  // Local hook for immediate responsive telemetry
  window.onLocalBiometricEvent = (event) => {
    keystrokeCount++;
    if (event.type === 'keystroke') {
      const holdEl = document.getElementById('stat-hold-time');
      const flightEl = document.getElementById('stat-flight-time');
      if (holdEl) holdEl.textContent = `${event.hold_time} ms`;
      if (flightEl) flightEl.textContent = `${event.flight_time} ms`;
      
      // Update oscilloscope
      oscilloscope.pushSample(event.hold_time, event.flight_time, 200, sessionTrust);
    } else if (event.type === 'mouse') {
      const speedEl = document.getElementById('stat-mouse-speed');
      const curvEl = document.getElementById('stat-mouse-curv');
      if (speedEl) speedEl.textContent = `${Math.round(event.velocity)} px/s`;
      if (curvEl) curvEl.textContent = `${event.curvature.toFixed(2)} rad`;
      
      oscilloscope.pushSample(70, 85, event.velocity, sessionTrust);
    }
  };

  // 2. Handle Backend Inference Response
  function handleInferenceResult(data) {
    if (!data) return;

    if (data.trust_score !== undefined) {
      sessionTrust = data.trust_score;
      updateTrustGauge(sessionTrust, data.threat_state);
    }

    if (data.is_locked) {
      triggerLockout(true);
      return;
    }

    if (data.attention) {
      attentionViz.updateAttention(data.attention);
    }

    if (data.telemetry) {
      radar.updateMetrics(data.telemetry.current, data.telemetry.baseline);
    }

    // Sound feedback on state transitions
    if (data.threat_state === 'SECURE') {
      // Audio chime on high confidence
    } else if (data.threat_state === 'MONITORING' || data.threat_state === 'ELEVATED') {
      audio.playWarningChime();
    } else if (data.threat_state === 'LOCKED') {
      audio.playLockAlarm();
    }

    // Refresh audit trail
    fetchAuditLog();
  }

  // 3. Trust Gauge & Threat Level Controller
  function updateTrustGauge(score, threatState) {
    const valCircle = document.getElementById('gauge-val-circle');
    const scoreVal = document.getElementById('gauge-score-value');
    const stateBanner = document.getElementById('security-state-banner');
    const stateText = document.getElementById('security-state-text');
    const navPulse = document.getElementById('nav-pulse-indicator');
    const navThreat = document.getElementById('nav-threat-label');

    if (!valCircle || !scoreVal) return;

    // SVG Circumference = 2 * PI * 90 = 565.48
    const circumference = 565.48;
    const offset = circumference - (score / 100.0) * circumference;
    valCircle.style.strokeDashoffset = offset;
    scoreVal.textContent = `${Math.round(score)}%`;

    valCircle.classList.remove('warning', 'danger');
    stateBanner.className = 'security-state-banner';
    navPulse.className = 'pulse-indicator';

    if (score >= currentThreshold * 100) {
      stateBanner.classList.add('secure');
      stateText.textContent = 'SECURE / AUTHENTICATED';
      navThreat.textContent = 'SECURE';
    } else if (score >= 52) {
      valCircle.classList.add('warning');
      stateBanner.classList.add('monitoring');
      stateText.textContent = 'MONITORING / ELEVATED SCRUTINY';
      navPulse.classList.add('warning');
      navThreat.textContent = 'MONITORING';
    } else if (score >= 35) {
      valCircle.classList.add('warning');
      stateBanner.classList.add('elevated');
      stateText.textContent = 'ANOMALY DETECTED / STEP-UP CHALLENGE';
      navPulse.classList.add('warning');
      navThreat.textContent = 'ELEVATED';
    } else {
      valCircle.classList.add('danger');
      stateBanner.classList.add('locked');
      stateText.textContent = 'CRITICAL / SESSION LOCKED';
      navPulse.classList.add('danger');
      navThreat.textContent = 'LOCKED';
      triggerLockout(true);
    }
  }

  // 4. Session Lockout Modal Controller
  function triggerLockout(locked) {
    isLocked = locked;
    const modal = document.getElementById('lockout-modal');
    if (!modal) return;

    if (locked) {
      modal.classList.add('active');
      audio.playLockAlarm();
      const pinInputs = document.querySelectorAll('.pin-input');
      pinInputs.forEach(i => i.value = '');
      if (pinInputs[0]) pinInputs[0].focus();
    } else {
      modal.classList.remove('active');
    }
  }

  // PIN Input Handling
  const pinInputs = document.querySelectorAll('.pin-input');
  pinInputs.forEach((input, idx) => {
    input.addEventListener('input', (e) => {
      if (e.target.value.length === 1 && idx < pinInputs.length - 1) {
        pinInputs[idx + 1].focus();
      }
      checkPin();
    });
    input.addEventListener('keydown', (e) => {
      if (e.key === 'Backspace' && !e.target.value && idx > 0) {
        pinInputs[idx - 1].focus();
      }
    });
  });

  async function checkPin() {
    let pin = '';
    pinInputs.forEach(i => pin += i.value);
    if (pin.length === 4) {
      if (pin === '1234') {
        const res = await fetch('/api/unlock', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ pin })
        });
        const data = await res.json();
        if (data.success) {
          triggerLockout(false);
          sessionTrust = data.trust_score;
          updateTrustGauge(sessionTrust, 'SECURE');
          audio.playAuthChime();
        }
      } else {
        audio.playWarningChime();
        pinInputs.forEach(i => i.value = '');
        pinInputs[0].focus();
      }
    }
  }

  // 5. Interactive Keystroke Challenge Arena & Typing Speed Comparator
  const challengeTexts = {
    standard: "Security is not a product, but a process. Continuous behavioral authentication dynamically safeguards user identities through distinctive neuromuscular dynamics and self-attention transformers.",
    tech: "import torch\nfrom model import BiometricSelfAttentionNet\nmodel = BiometricSelfAttentionNet(input_dim=8, d_model=64, nhead=4)\nemb, attn = model.extract_embedding(x, return_attention=True)",
    freeform: "Type freely in your natural everyday typing rhythm. Notice how your hold times, inter-key flight latencies, and cadence are captured passively."
  };

  const typingInput = document.getElementById('typing-input-area');
  const typingPromptBox = document.getElementById('typing-prompt-box');
  const typingPromptTabs = document.querySelectorAll('.capture-tab-btn');
  const btnCheckTyping = document.getElementById('btn-check-typing');
  const btnRestartTyping = document.getElementById('btn-restart-typing');
  const inlineVerdict = document.getElementById('inline-typing-verdict');
  const inlineVerdictText = document.getElementById('inline-verdict-text');
  const btnViewReport = document.getElementById('btn-view-report');
  const typingReportModal = document.getElementById('typing-report-modal');
  const btnCloseReport = document.getElementById('btn-close-report');
  const btnModalClose = document.getElementById('btn-modal-close');
  const btnEnrollFromReport = document.getElementById('btn-enroll-from-report');

  let currentPrompt = challengeTexts.standard;
  let currentTypingEvents = [];
  let typingTestStartTime = null;
  let lastTypingReportData = null;

  let firstUserBaseline = {
    id: 0,
    name: 'Alice Vance (Touch Typist)',
    wpm: 84.0,
    hold: 72.0,
    flight: 85.0
  };

  function setPrompt(key) {
    currentPrompt = challengeTexts[key] || challengeTexts.standard;
    if (typingPromptBox) {
      typingPromptBox.textContent = currentPrompt;
    }
    resetTypingTest();
  }

  function resetTypingTest() {
    if (typingInput) {
      typingInput.value = '';
      typingInput.focus();
    }
    typingTestStartTime = null;
    currentTypingEvents = [];
    
    // Reset UI indicators
    const wpmEl = document.getElementById('stat-wpm');
    const liveWpmEl = document.getElementById('my-live-wpm');
    const liveCpsEl = document.getElementById('my-live-cps');
    const diffEl = document.getElementById('speed-diff-val');
    const matchEl = document.getElementById('speed-match-val');

    if (wpmEl) wpmEl.textContent = '0';
    if (liveWpmEl) liveWpmEl.textContent = '0.0 WPM';
    if (liveCpsEl) liveCpsEl.textContent = '0.0 chars/sec | 0 keys';
    if (diffEl) diffEl.textContent = '-- WPM';
    if (matchEl) matchEl.textContent = 'Speed Match: --%';
    if (inlineVerdict) inlineVerdict.style.display = 'none';

    // Reset prompt highlighting
    if (typingPromptBox) {
      typingPromptBox.textContent = currentPrompt;
    }
  }

  typingPromptTabs.forEach(btn => {
    btn.addEventListener('click', (e) => {
      typingPromptTabs.forEach(b => b.classList.remove('active'));
      e.target.classList.add('active');
      const mode = e.target.getAttribute('data-mode');
      setPrompt(mode);
    });
  });

  if (typingInput) {
    typingInput.addEventListener('input', (e) => {
      const typed = e.target.value;
      const prompt = currentPrompt;
      const now = Date.now();

      if (!typingTestStartTime && typed.length > 0) {
        typingTestStartTime = now;
      }

      // Calculate My Live Typing Speed (WPM & CPS)
      const elapsedMinutes = Math.max(0.015, (now - (typingTestStartTime || now)) / 60000.0);
      const elapsedSeconds = Math.max(0.9, (now - (typingTestStartTime || now)) / 1000.0);
      const words = typed.trim().split(/\s+/).filter(Boolean).length;
      const charCount = typed.length;
      const wpm = Math.round((charCount / 5.0) / elapsedMinutes);
      const cps = (charCount / elapsedSeconds).toFixed(1);

      // Speed Differential & Match Ratio vs First User (Alice Vance: 84 WPM)
      const diffWpm = Math.round(wpm - firstUserBaseline.wpm);
      const matchRatio = Math.round((wpm / firstUserBaseline.wpm) * 100.0);

      // Update UI Counters
      const wpmEl = document.getElementById('stat-wpm');
      const liveWpmEl = document.getElementById('my-live-wpm');
      const liveCpsEl = document.getElementById('my-live-cps');
      const diffEl = document.getElementById('speed-diff-val');
      const matchEl = document.getElementById('speed-match-val');

      if (wpmEl) wpmEl.textContent = `${wpm}`;
      if (liveWpmEl) liveWpmEl.textContent = `${wpm} WPM`;
      if (liveCpsEl) liveCpsEl.textContent = `${cps} chars/sec | ${charCount} keys`;
      if (diffEl) {
        diffEl.textContent = `${diffWpm >= 0 ? '+' : ''}${diffWpm} WPM`;
        diffEl.style.color = Math.abs(diffWpm) <= 15 ? 'var(--accent-emerald)' : 'var(--accent-amber)';
      }
      if (matchEl) {
        matchEl.textContent = `Speed Match: ${matchRatio}%`;
      }

      // Highlight matched characters in prompt box
      if (typingPromptBox) {
        let html = '';
        for (let i = 0; i < prompt.length; i++) {
          if (i < typed.length) {
            if (typed[i] === prompt[i]) {
              html += `<span class="typing-prompt-char matched">${escapeHtml(prompt[i])}</span>`;
            } else {
              html += `<span class="typing-prompt-char mismatch">${escapeHtml(prompt[i])}</span>`;
            }
          } else if (i === typed.length) {
            html += `<span class="typing-prompt-char current">${escapeHtml(prompt[i])}</span>`;
          } else {
            html += `<span class="typing-prompt-char">${escapeHtml(prompt[i])}</span>`;
          }
        }
        typingPromptBox.innerHTML = html;
      }
    });
  }

  // Hook into local keystroke capture to record typing events for the test
  const originalOnLocalBiometric = window.onLocalBiometricEvent;
  window.onLocalBiometricEvent = (event) => {
    if (originalOnLocalBiometric) originalOnLocalBiometric(event);
    if (event.type === 'keystroke') {
      currentTypingEvents.push(event);
    }
  };

  // Check Typing Speed & Verify Biometrics Action
  if (btnCheckTyping) {
    btnCheckTyping.addEventListener('click', async () => {
      const typed = typingInput ? typingInput.value : '';
      if (!typed || typed.length < 4) {
        alert('Please type a few words in the box first to measure your typing speed!');
        return;
      }

      btnCheckTyping.disabled = true;
      btnCheckTyping.innerHTML = `<span>Checking Typing Speed...</span>`;

      const now = Date.now();
      const durMs = typingTestStartTime ? (now - typingTestStartTime) : 5000;

      try {
        const res = await fetch('/api/check_typing', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            text: typed,
            events: currentTypingEvents,
            duration_ms: durMs,
            char_count: typed.length,
            error_count: 0
          })
        });

        const data = await res.json();
        lastTypingReportData = data;
        displayTypingReport(data);

        // If attention weights available, update heatmap
        if (data.attention) {
          attentionViz.updateAttention(data.attention);
        }
        audio.playAuthChime();
      } catch (err) {
        console.warn('Check typing error:', err);
      } finally {
        btnCheckTyping.disabled = false;
        btnCheckTyping.innerHTML = `
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <polyline points="20 6 9 17 4 12"/>
          </svg>
          <span>Check Typing Speed &amp; Verify Biometrics</span>
        `;
      }
    });
  }

  // Display Typing Check Results & Open Report
  function displayTypingReport(data) {
    if (!data || !data.success) return;

    const my = data.my_typing;
    const first = data.first_user;
    const comp = data.comparison;
    const verif = data.verification;

    // 1. Update Inline Banner
    if (inlineVerdict && inlineVerdictText) {
      inlineVerdict.className = `inline-typing-verdict ${verif.verdict_class === 'MATCH' ? 'match' : 'mismatch'}`;
      inlineVerdictText.innerHTML = `<strong>${verif.verdict_class === 'MATCH' ? 'VERIFIED MATCH' : 'BEHAVIORAL DRIFT'}:</strong> My Speed: <strong>${my.wpm} WPM</strong> vs First User: <strong>${first.wpm} WPM</strong> (Delta: ${comp.speed_diff_wpm >= 0 ? '+' : ''}${comp.speed_diff_wpm} WPM | Match: ${comp.rhythm_match_pct}%)`;
      inlineVerdict.style.display = 'flex';
    }

    // 2. Populate Modal Fields
    const modalVerdict = document.getElementById('modal-report-verdict');
    const modalFirstWpm = document.getElementById('modal-first-user-wpm');
    const modalFirstSub = document.getElementById('modal-first-user-sub');
    const modalMyWpm = document.getElementById('modal-my-wpm');
    const modalMySub = document.getElementById('modal-my-sub');
    const modalDiffWpm = document.getElementById('modal-diff-wpm');
    const modalRatioSub = document.getElementById('modal-ratio-sub');

    if (modalVerdict) {
      modalVerdict.className = `report-verdict-box ${verif.verdict_class === 'MATCH' ? 'match' : 'mismatch'}`;
      modalVerdict.innerHTML = `<strong>${verif.verdict_class === 'MATCH' ? '&#10004; AUTHENTIC USER VERIFICATION SUCCESS' : '&#9888; ANOMALY / DIVERGENCE DETECTED'}</strong><br>${verif.verdict}`;
    }

    if (modalFirstWpm) modalFirstWpm.textContent = `${first.wpm} WPM`;
    if (modalFirstSub) modalFirstSub.textContent = `Hold: ${first.mean_hold_ms}ms | Flight: ${first.mean_flight_ms}ms`;
    if (modalMyWpm) modalMyWpm.textContent = `${my.wpm} WPM`;
    if (modalMySub) modalMySub.textContent = `Hold: ${my.mean_hold_ms}ms | Flight: ${my.mean_flight_ms}ms`;
    if (modalDiffWpm) {
      modalDiffWpm.textContent = `${comp.speed_diff_wpm >= 0 ? '+' : ''}${comp.speed_diff_wpm} WPM`;
      modalDiffWpm.style.color = Math.abs(comp.speed_diff_wpm) <= 12 ? 'var(--accent-emerald)' : 'var(--accent-rose)';
    }
    if (modalRatioSub) modalRatioSub.textContent = `Rhythm Match: ${comp.rhythm_match_pct}% (Ratio: ${comp.speed_ratio_pct}%)`;

    // Table rows
    const rowMyWpm = document.getElementById('row-my-wpm');
    const rowDeltaWpm = document.getElementById('row-delta-wpm');
    const rowMyHold = document.getElementById('row-my-hold');
    const rowDeltaHold = document.getElementById('row-delta-hold');
    const rowMyFlight = document.getElementById('row-my-flight');
    const rowDeltaFlight = document.getElementById('row-delta-flight');
    const rowMyJitter = document.getElementById('row-my-jitter');
    const rowDeltaJitter = document.getElementById('row-delta-jitter');
    const rowMyAcc = document.getElementById('row-my-acc');

    if (rowMyWpm) rowMyWpm.textContent = `${my.wpm} WPM (${my.cps} cps)`;
    if (rowDeltaWpm) rowDeltaWpm.textContent = `${comp.speed_diff_wpm >= 0 ? '+' : ''}${comp.speed_diff_wpm} WPM`;
    if (rowMyHold) rowMyHold.textContent = `${my.mean_hold_ms} ms`;
    if (rowDeltaHold) rowDeltaHold.textContent = `${comp.hold_diff_ms >= 0 ? '+' : ''}${comp.hold_diff_ms} ms`;
    if (rowMyFlight) rowMyFlight.textContent = `${my.mean_flight_ms} ms`;
    if (rowDeltaFlight) rowDeltaFlight.textContent = `${comp.flight_diff_ms >= 0 ? '+' : ''}${comp.flight_diff_ms} ms`;
    if (rowMyJitter) rowMyJitter.textContent = `${my.jitter_ms} ms`;
    if (rowDeltaJitter) rowDeltaJitter.textContent = `${(my.jitter_ms - 22.0).toFixed(1)} ms`;
    if (rowMyAcc) rowMyAcc.textContent = `${my.accuracy_pct}%`;

    // Open Modal
    if (typingReportModal) {
      typingReportModal.classList.add('active');
    }
  }

  // Modal Handlers
  if (btnViewReport) {
    btnViewReport.addEventListener('click', () => {
      if (typingReportModal) typingReportModal.classList.add('active');
    });
  }

  if (btnCloseReport) {
    btnCloseReport.addEventListener('click', () => {
      if (typingReportModal) typingReportModal.classList.remove('active');
    });
  }

  if (btnModalClose) {
    btnModalClose.addEventListener('click', () => {
      if (typingReportModal) typingReportModal.classList.remove('active');
    });
  }

  if (btnRestartTyping) {
    btnRestartTyping.addEventListener('click', resetTypingTest);
  }

  // Save / Enroll Custom Profile from Report
  if (btnEnrollFromReport) {
    btnEnrollFromReport.addEventListener('click', async () => {
      if (!currentTypingEvents || currentTypingEvents.length < 10) {
        alert('Need at least 10 keystrokes to enroll a custom data model profile!');
        return;
      }
      const profileName = prompt('Enter your name for the new Biometric Profile in Data Model:', 'My Custom Profile');
      if (!profileName) return;

      try {
        const res = await fetch('/api/enroll', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            name: profileName,
            events: currentTypingEvents
          })
        });
        const data = await res.json();
        if (data.success) {
          alert(`Enrolled profile '${profileName}' with typing speed ${data.profile.typing_speed_wpm} WPM successfully into datamodel!`);
          if (typingReportModal) typingReportModal.classList.remove('active');
          fetchStatus();
          fetchProfiles();
        }
      } catch (e) {
        console.warn('Enrollment error:', e);
      }
    });
  }

  function escapeHtml(text) {
    return text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/\n/g, '<br/>');
  }

  // 6. Interactive Mouse Canvas Arena
  const mouseCanvas = document.getElementById('mouse-canvas');
  if (mouseCanvas) {
    const mctx = mouseCanvas.getContext('2d');
    let mouseTrail = [];
    let targets = [
      { x: 60, y: 50, r: 16, active: true },
      { x: 220, y: 110, r: 16, active: true },
      { x: 380, y: 60, r: 16, active: true }
    ];

    function resizeMouseCanvas() {
      const rect = mouseCanvas.parentElement.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      mouseCanvas.width = rect.width * dpr;
      mouseCanvas.height = rect.height * dpr;
      mctx.scale(dpr, dpr);
    }
    resizeMouseCanvas();
    window.addEventListener('resize', resizeMouseCanvas);

    mouseCanvas.addEventListener('mousemove', (e) => {
      const rect = mouseCanvas.getBoundingClientRect();
      mouseTrail.push({
        x: e.clientX - rect.left,
        y: e.clientY - rect.top,
        alpha: 1.0
      });
      if (mouseTrail.length > 25) mouseTrail.shift();
    });

    mouseCanvas.addEventListener('mousedown', (e) => {
      const rect = mouseCanvas.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;

      targets.forEach(t => {
        const dist = Math.sqrt((mx - t.x) ** 2 + (my - t.y) ** 2);
        if (dist <= t.r + 10) {
          audio.playAuthChime();
          t.x = Math.random() * (rect.width - 60) + 30;
          t.y = Math.random() * (rect.height - 60) + 30;
        }
      });
    });

    function drawMouseArena() {
      const rect = mouseCanvas.getBoundingClientRect();
      mctx.clearRect(0, 0, rect.width, rect.height);

      // Draw interactive targets
      targets.forEach((t, i) => {
        mctx.fillStyle = 'rgba(6, 182, 212, 0.2)';
        mctx.strokeStyle = '#06b6d4';
        mctx.lineWidth = 1.5;
        mctx.beginPath();
        mctx.arc(t.x, t.y, t.r, 0, Math.PI * 2);
        mctx.fill();
        mctx.stroke();

        mctx.fillStyle = '#ffffff';
        mctx.font = '10px Outfit, sans-serif';
        mctx.textAlign = 'center';
        mctx.textBaseline = 'middle';
        mctx.fillText(`Target ${i + 1}`, t.x, t.y);
      });

      // Draw cursor trailing sparks
      for (let i = 0; i < mouseTrail.length; i++) {
        const pt = mouseTrail[i];
        pt.alpha *= 0.92;
        mctx.fillStyle = `rgba(56, 189, 248, ${pt.alpha * 0.7})`;
        mctx.beginPath();
        mctx.arc(pt.x, pt.y, (i / mouseTrail.length) * 4 + 1, 0, Math.PI * 2);
        mctx.fill();
      }

      requestAnimationFrame(drawMouseArena);
    }
    requestAnimationFrame(drawMouseArena);
  }

  // 7. Attack Vector Simulation Buttons
  const simButtons = document.querySelectorAll('.sim-btn');
  simButtons.forEach(btn => {
    btn.addEventListener('click', async (e) => {
      const target = e.currentTarget;
      const mode = target.getAttribute('data-sim');
      target.style.opacity = '0.6';
      
      try {
        const res = await fetch('/api/simulate', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ mode, steps: 32 })
        });
        const data = await res.json();
        handleInferenceResult(data);
      } catch (err) {
        console.warn('Simulation call failed:', err);
      } finally {
        target.style.opacity = '1';
      }
    });
  });

  // 8. Self-Attention Head Switcher
  const headTabs = document.querySelectorAll('.attn-tab-btn');
  headTabs.forEach(btn => {
    btn.addEventListener('click', (e) => {
      headTabs.forEach(b => b.classList.remove('active'));
      e.target.classList.add('active');
      const headKey = e.target.getAttribute('data-head');
      attentionViz.setHead(headKey);
    });
  });

  // 9. User Profile Selector
  const userSelect = document.getElementById('user-profile-select');
  if (userSelect) {
    userSelect.addEventListener('change', async (e) => {
      const newId = parseInt(e.target.value);
      const res = await fetch('/api/switch_profile', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ profile_id: newId })
      });
      const data = await res.json();
      if (data.success) {
        activeProfileId = newId;
        sessionTrust = 92.0;
        updateTrustGauge(sessionTrust, 'SECURE');
        fetchStatus();
      }
    });
  }

  // 10. Threshold Slider
  const threshSlider = document.getElementById('threshold-slider');
  const threshValLabel = document.getElementById('threshold-value-label');
  if (threshSlider) {
    threshSlider.addEventListener('input', (e) => {
      currentThreshold = parseFloat(e.target.value);
      if (threshValLabel) {
        threshValLabel.textContent = `${Math.round(currentThreshold * 100)}%`;
      }
      rocViz.setThreshold(currentThreshold);
      updateTrustGauge(sessionTrust, sessionTrust >= currentThreshold * 100 ? 'SECURE' : 'MONITORING');
    });
  }

  // 11. Model Training Studio
  const trainBtn = document.getElementById('btn-train-model');
  const trainProgress = document.getElementById('train-progress-bar');
  const trainProgressFill = document.getElementById('train-progress-fill');
  const trainLossText = document.getElementById('train-loss-text');

  if (trainBtn) {
    trainBtn.addEventListener('click', async () => {
      trainBtn.disabled = true;
      trainBtn.textContent = 'Training in PyTorch...';
      if (trainProgress) trainProgress.style.display = 'block';
      if (trainProgressFill) trainProgressFill.style.width = '30%';

      try {
        const res = await fetch('/api/train', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ epochs: 10, lr: 0.001 })
        });
        const data = await res.json();

        if (trainProgressFill) trainProgressFill.style.width = '100%';
        if (trainLossText) {
          const finalLoss = data.losses[data.losses.length - 1];
          trainLossText.textContent = `Completed 10 epochs. Final Loss: ${finalLoss} | EER: ${(data.eer * 100).toFixed(1)}%`;
        }
        audio.playAuthChime();
        fetchBenchmark();
      } catch (err) {
        console.warn('Training failed:', err);
      } finally {
        setTimeout(() => {
          trainBtn.disabled = false;
          trainBtn.textContent = 'Retrain Deep Network';
          if (trainProgress) trainProgress.style.display = 'none';
        }, 1200);
      }
    });
  }

  // 12. Reset Session & Sound Toggle
  const btnReset = document.getElementById('btn-reset-session');
  if (btnReset) {
    btnReset.addEventListener('click', async () => {
      await fetch('/api/reset_session', { method: 'POST' });
      sessionTrust = 95.0;
      updateTrustGauge(sessionTrust, 'SECURE');
      triggerLockout(false);
      audio.playAuthChime();
    });
  }

  const btnLogout = document.getElementById('btn-logout');
  if (btnLogout) {
    btnLogout.addEventListener('click', async () => {
      if (confirm('Terminate session and log out of Bio-Sentry?')) {
        await fetch('/api/logout', { method: 'POST' });
        window.location.href = '/login';
      }
    });
  }

  const btnSound = document.getElementById('btn-toggle-sound');
  if (btnSound) {
    btnSound.addEventListener('click', () => {
      const muted = audio.toggleMute();
      btnSound.style.opacity = muted ? '0.4' : '1.0';
    });
  }

  // 13. Audit Log Fetcher
  async function fetchAuditLog() {
    try {
      const res = await fetch('/api/audit_log');
      const data = await res.json();
      const tbody = document.getElementById('audit-table-body');
      if (!tbody || !data.events) return;

      tbody.innerHTML = data.events.map(ev => `
        <tr>
          <td>${ev.timestamp}</td>
          <td><span class="card-badge ${ev.threat.toLowerCase()}">${ev.type}</span></td>
          <td>${ev.details}</td>
          <td style="font-weight:700;">${ev.score}%</td>
        </tr>
      `).join('');
    } catch (e) {}
  }

  // 14. Initial Data Sync
  async function fetchStatus() {
    try {
      const res = await fetch('/api/status');
      const data = await res.json();
      if (data.active_profile) {
        const profBadge = document.getElementById('nav-profile-name');
        if (profBadge) profBadge.textContent = data.active_profile.name;
      }
      if (data.first_user) {
        firstUserBaseline = {
          id: data.first_user.id,
          name: data.first_user.name,
          wpm: data.first_user.typing_speed_wpm || 84.0,
          hold: data.first_user.mean_hold || 72.0,
          flight: data.first_user.mean_flight || 85.0
        };

        const navSpeedEl = document.getElementById('nav-first-user-speed');
        const firstWpmValEl = document.getElementById('first-user-wpm-val');
        if (navSpeedEl) navSpeedEl.textContent = `${firstUserBaseline.wpm} WPM`;
        if (firstWpmValEl) firstWpmValEl.textContent = `${firstUserBaseline.wpm} WPM`;
      }
      if (data.trust_score) {
        sessionTrust = data.trust_score;
        updateTrustGauge(sessionTrust, data.threat_state);
      }
      if (data.model) {
        const modelDesc = document.getElementById('model-specs-desc');
        if (modelDesc) {
          modelDesc.textContent = `${data.model.architecture} (${data.model.parameters.toLocaleString()} params, d_model=${data.model.d_model}, heads=${data.model.heads})`;
        }
      }
    } catch (e) {}
  }

  async function fetchProfiles() {
    try {
      const res = await fetch('/api/profiles');
      const data = await res.json();
      if (!data.profiles) return;
      const select = document.getElementById('user-profile-select');
      if (select) {
        select.innerHTML = data.profiles.map(p => `
          <option value="${p.id}" ${p.is_active ? 'selected' : ''}>${p.name} (${p.typing_speed_wpm} WPM)</option>
        `).join('');
      }
    } catch (e) {}
  }

  async function fetchBenchmark() {
    try {
      const res = await fetch('/api/benchmark');
      const data = await res.json();
      rocViz.updateBenchmarkData(data, currentThreshold);

      const eerEl = document.getElementById('stat-eer-val');
      const aucEl = document.getElementById('stat-auc-val');
      if (eerEl) eerEl.textContent = `${(data.eer * 100).toFixed(1)}%`;
      if (aucEl) aucEl.textContent = `${data.auc.toFixed(3)}`;
    } catch (e) {}
  }

  // Initial runs
  fetchStatus();
  fetchProfiles();
  fetchBenchmark();
  fetchAuditLog();
  setPrompt('standard');
});
