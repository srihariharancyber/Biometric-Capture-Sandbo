/**
 * AURA: Real-Time Behavioral Biometric Tracker
 * Captures keystroke dynamics and mouse movement kinematics with sub-millisecond precision.
 */
class BiometricTracker {
  constructor(onBatchReady) {
    this.onBatchReady = onBatchReady;
    this.activeKeyDowns = new Map(); // key -> timestamp
    this.lastKeyDownTime = null;
    this.lastMouseSample = null;
    this.mouseHistory = []; // [ {x, y, t} ]
    this.eventBuffer = []; // buffered events to send
    this.isTracking = true;
    this.batchInterval = 1200; // ms
    this.timer = null;
    
    this.initListeners();
    this.startDispatchLoop();
  }

  initListeners() {
    // 1. Keystroke Dynamics Listeners
    window.addEventListener('keydown', (e) => {
      if (!this.isTracking) return;
      const now = performance.now();
      
      // Calculate Flight Time (inter-key latency)
      let flightTime = 0;
      if (this.lastKeyDownTime !== null) {
        flightTime = Math.max(5, now - this.lastKeyDownTime);
      }
      this.lastKeyDownTime = now;

      // Store keydown start time if not repeating
      if (!this.activeKeyDowns.has(e.code)) {
        this.activeKeyDowns.set(e.code, {
          startTime: now,
          flightTime: flightTime,
          key: e.key
        });
      }

      window.cyberAudio?.playKeyTick();
    }, true);

    window.addEventListener('keyup', (e) => {
      if (!this.isTracking) return;
      const now = performance.now();
      
      if (this.activeKeyDowns.has(e.code)) {
        const info = this.activeKeyDowns.get(e.code);
        this.activeKeyDowns.delete(e.code);
        
        const holdTime = Math.max(10, now - info.startTime);
        
        const event = {
          type: 'keystroke',
          key: info.key,
          hold_time: Math.round(holdTime),
          flight_time: Math.round(info.flightTime),
          velocity: 0,
          accel: 0,
          curvature: 0,
          jerk: 0,
          timestamp: Date.now()
        };

        this.recordEvent(event);
      }
    }, true);

    // 2. Mouse Dynamics Listeners (sampled at max 60Hz)
    let lastMouseTime = 0;
    window.addEventListener('mousemove', (e) => {
      if (!this.isTracking) return;
      const now = performance.now();
      if (now - lastMouseTime < 24) return; // limit sample rate
      lastMouseTime = now;

      this.processMouseSample(e.clientX, e.clientY, now);
    }, { passive: true });

    let clickStartTime = 0;
    window.addEventListener('mousedown', (e) => {
      if (!this.isTracking) return;
      clickStartTime = performance.now();
    });

    window.addEventListener('mouseup', (e) => {
      if (!this.isTracking) return;
      const now = performance.now();
      const clickDuration = now - clickStartTime;
      
      this.recordEvent({
        type: 'mouse',
        key: 'click',
        hold_time: Math.round(clickDuration),
        flight_time: 25,
        velocity: 150,
        accel: 50,
        curvature: 0.1,
        jerk: 10,
        timestamp: Date.now()
      });
    });
  }

  processMouseSample(x, y, t) {
    this.mouseHistory.push({ x, y, t });
    if (this.mouseHistory.length > 5) {
      this.mouseHistory.shift();
    }

    if (this.mouseHistory.length < 3) return;

    const p0 = this.mouseHistory[this.mouseHistory.length - 3];
    const p1 = this.mouseHistory[this.mouseHistory.length - 2];
    const p2 = this.mouseHistory[this.mouseHistory.length - 1];

    const dt1 = Math.max(1, (p1.t - p0.t) / 1000); // seconds
    const dt2 = Math.max(1, (p2.t - p1.t) / 1000); // seconds

    const dx1 = p1.x - p0.x;
    const dy1 = p1.y - p0.y;
    const dist1 = Math.sqrt(dx1 * dx1 + dy1 * dy1);
    const v1 = dist1 / dt1; // px/s

    const dx2 = p2.x - p1.x;
    const dy2 = p2.y - p1.y;
    const dist2 = Math.sqrt(dx2 * dx2 + dy2 * dy2);
    const v2 = dist2 / dt2; // px/s

    // Acceleration & Jerk
    const dtAvg = (dt1 + dt2) / 2;
    const accel = Math.abs(v2 - v1) / dtAvg;
    const jerk = accel / dtAvg;

    // Trajectory Curvature (angle change)
    const angle1 = Math.atan2(dy1, dx1);
    const angle2 = Math.atan2(dy2, dx2);
    let diffAngle = Math.abs(angle2 - angle1);
    if (diffAngle > Math.PI) diffAngle = 2 * Math.PI - diffAngle;

    // Only record if actual motion occurred
    if (dist2 > 3) {
      this.recordEvent({
        type: 'mouse',
        key: '',
        hold_time: 0,
        flight_time: Math.round(dt2 * 1000),
        velocity: Math.round(v2),
        accel: Math.round(accel),
        curvature: parseFloat(diffAngle.toFixed(3)),
        jerk: Math.round(jerk),
        timestamp: Date.now()
      });
    }
  }

  recordEvent(event) {
    this.eventBuffer.push(event);
    if (window.onLocalBiometricEvent) {
      window.onLocalBiometricEvent(event);
    }

    // Auto-dispatch if buffer gets large enough
    if (this.eventBuffer.length >= 12) {
      this.dispatchBatch();
    }
  }

  startDispatchLoop() {
    this.timer = setInterval(() => {
      if (this.eventBuffer.length >= 4) {
        this.dispatchBatch();
      }
    }, this.batchInterval);
  }

  dispatchBatch() {
    if (this.eventBuffer.length === 0) return;
    const batch = [...this.eventBuffer];
    this.eventBuffer = [];
    if (this.onBatchReady) {
      this.onBatchReady(batch);
    }
  }
}

window.BiometricTracker = BiometricTracker;
