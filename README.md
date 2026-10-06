# AURA: Continuous User Authentication via Behavioral Biometrics and Self-Attention Deep Networks

**AURA (Adaptive User Recognition & Authentication)** is an AI-powered, real-time continuous behavioral biometrics authentication platform. Unlike conventional static authentication systems (passwords, OTPs, or initial biometric scans) that only authenticate a user at login, AURA operates passively and continuously in the background, verifying the legitimate user's identity through micro-behavioral neuromuscular patterns: keystroke dynamics and mouse kinematics.

---

## 🌟 Key Capabilities & Features

1. **Continuous Passive Identity Verification**
   - **Zero Friction**: Monitors legitimate human-device interaction in real-time without requiring active user interruptions.
   - **Multi-Modal Biometrics**: Jointly analyzes keystroke dynamics (hold times, inter-key flight latencies, digram cadences) and mouse kinematics (instantaneous velocity, acceleration, trajectory curvature, jerk).
   - **Exponential Moving Average (EMA) Trust Engine**: Smooths temporal trust scores $\text{Trust}_t = \lambda \cdot \text{Trust}_{t-1} + (1 - \lambda) \cdot \text{Score}_t$ to prevent jitter while reacting swiftly to intrusion attempts.

2. **Self-Attention Deep Network (Transformer Encoder)**
   - **Architecture**: Multi-Head Self-Attention (4 specialized attention heads), Sinusoidal Positional Encoding, Temporal Attention Pooling, and Deep Metric Learning projection into a 32-dimensional biometric latent space $\mathbb{R}^{32}$.
   - **Explainable AI (XAI) Heatmap**: Real-time $32 \times 32$ interactive attention matrix rendering token-to-token cross-attention weights across individual attention heads:
     - **Head 0**: Key Dwell & Hold Dynamics (Muscle memory chords)
     - **Head 1**: Inter-Key Flight Rhythms (Transition latencies)
     - **Head 2**: Mouse Trajectory Curvature & Jerk (Fine motor control arcs)
     - **Head 3**: Cadence Boundaries & Cognitive Pauses
     - **Mean**: Averaged Multi-Head Representation

3. **User Data Model & Typing Speed Comparator (`/api/check_typing`)**
   - **Data Model Integration**: Each user profile in the data model explicitly tracks `typing_speed_wpm` (Words Per Minute), `mean_hold` (ms), `mean_flight` (ms), and neuromuscular chord profiles.
     - **First User Baseline (Alice Vance)**: `84.0 WPM`, Hold `72 ms`, Flight `85 ms`.
     - **Secondary Users / Personas**: Bob (`36 WPM`), Charlie (`68 WPM`), Eve Imposter (`42 WPM`), ReplayBot (`150 WPM`).
     - **Custom Profiles**: Automatically enrolled with exact user-measured WPM and biometric centroid.
   - **Real-Time Speed Comparison**: Live telemetry strip displaying:
     - First User Baseline Speed vs My Current Measured Typing Speed (WPM & CPS).
     - Speed Differential ($\Delta$ WPM) and Cadence Match Ratio (%).
   - **Post-Typing Biometric Verification Report**:
     - Evaluates typing sequences against the First User baseline.
     - Computes Hold and Flight Latency deltas, jitter variance, and keystroke accuracy.
     - Feeds the sequence through the PyTorch Transformer to generate an authentic vs imposter verification verdict with full Explainable AI attention matrices.

4. **Interactive Biometric Capture & Testing Arena**
   - **Typing Sandbox**: Guided security challenge phrases, code snippets, or freeform typing with real-time character matching, WPM, hold time, and flight latency readouts.
   - **Mouse Kinematics Canvas**: Dynamic tracking arena with glowing particle cursor trails, interactive target click points, and real-time curvature/velocity calculations.

4. **Adversarial Attack Simulation Suite**
   - **Authentic User**: Natural cadence conforming to the enrolled baseline profile.
   - **Human Intruder (Eve)**: Unfamiliar neuromuscular cadence causing behavioral drift.
   - **Replay Bot Macro**: Synthetic attack with robotic zero-jitter timings and linear mouse vectors.
   - **Fatigue Drift**: Gradual decline in typing speed and motor coordination.

5. **Security State Machine & Automated Lockdown**
   - **SECURE / AUTHENTICATED** ($\ge 70\%$ Trust): Green status, seamless operation.
   - **MONITORING / ELEVATED SCRUTINY** ($52\% - 69\%$ Trust): Amber warning, heightened surveillance.
   - **ANOMALY DETECTED / STEP-UP** ($35\% - 51\%$ Trust): High alert, warning chime.
   - **BREACH / SESSION LOCKED** ($< 35\%$ Trust): Fullscreen lockdown modal, auditory alarm siren, operator PIN challenge (default `1234`).

6. **Benchmark & Retraining Studio**
   - Evaluates False Acceptance Rate (FAR), False Rejection Rate (FRR), Equal Error Rate (EER), and Area Under the ROC Curve (AUC).
   - Interactive ROC Curve canvas with live threshold sensitivity adjustment.
   - In-app PyTorch fine-tuning and model retraining with live progress feedback.

---

## 🔬 Mathematical Formulation

### 1. Behavioral Biometric Token Representation
Each interaction event $e_t$ is transformed into an 8-dimensional normalized feature vector:
$$\mathbf{x}_t = \big[ \text{is\_mouse}, \hat{t}_{\text{hold}}, \hat{t}_{\text{flight}}, \hat{v}_{\text{mouse}}, \hat{a}_{\text{mouse}}, \hat{\kappa}_{\text{curv}}, \hat{j}_{\text{jerk}}, \text{cat}_{\text{key}} \big]^T \in \mathbb{R}^8$$

Where:
- $\hat{t}_{\text{hold}} = \min(1.0, (t_{\text{keyup}} - t_{\text{keydown}}) / 400\,\text{ms})$
- $\hat{t}_{\text{flight}} = \min(1.0, (t_{\text{keydown}, t} - t_{\text{keydown}, t-1}) / 800\,\text{ms})$
- $\hat{\kappa}_{\text{curv}} = |\theta_t - \theta_{t-1}| / \pi$ (Trajectory angular deviation)
- $\hat{j}_{\text{jerk}} = \frac{da}{dt}$ (Rate of change of acceleration)

### 2. Multi-Head Self-Attention
Given a sequence of $T=32$ biometric tokens $\mathbf{X} \in \mathbb{R}^{T \times d_{\text{model}}}$ after linear projection and positional encoding:
$$\mathbf{Q} = \mathbf{X}\mathbf{W}^Q, \quad \mathbf{K} = \mathbf{X}\mathbf{W}^K, \quad \mathbf{V} = \mathbf{X}\mathbf{W}^V$$
$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}$$
$$\text{MultiHead}(\mathbf{X}) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)\mathbf{W}^O$$

### 3. Deep Metric Learning & Identity Verification
The network extracts an $L_2$-normalized identity embedding $\mathbf{z}_t \in \mathbb{R}^{32}$ via temporal attention pooling:
$$\|\mathbf{z}_t\|_2 = 1$$
Continuous identity verification score against the enrolled user centroid $\boldsymbol{\mu}_{\text{user}}$:
$$S_t = \cos(\mathbf{z}_t, \boldsymbol{\mu}_{\text{user}}) = \mathbf{z}_t^T \boldsymbol{\mu}_{\text{user}}$$
The Triplet Loss optimization objective:
$$\mathcal{L}_{\text{triplet}} = \max\left(0, \|\mathbf{z}_a - \mathbf{z}_p\|_2^2 - \|\mathbf{z}_a - \mathbf{z}_n\|_2^2 + \alpha\right)$$

---

## 🚀 Running the Web Application

### Prerequisites
- Python 3.10+ (PyTorch, Flask, NumPy, Scipy, Scikit-Learn installed)

### Start the Server
```powershell
python app.py
```
The server will start listening at:
```
http://127.0.0.1:5000
```

### Accessing the Dashboard
Open your web browser (Chrome, Edge, Firefox, Brave) and navigate to:
[http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 🛠️ Project Structure

```
c:\Users\SRIHARIHARAN\Desktop\ML BIO\
├── app.py                      # Flask REST API, PyTorch inference, session state manager
├── model.py                    # BiometricSelfAttentionNet (Multi-Head Attention, Metric Learning)
├── dataset.py                  # Tokenizer, synthetic human/imposter generators, ROC/EER metrics
├── pretrained_model.pt         # Saved PyTorch checkpoint weights
├── templates/
│   └── index.html              # Cyber-defense command center web UI
├── static/
│   ├── css/
│   │   └── style.css           # Glassmorphic dark design system with neon accents
│   └── js/
│       ├── app.js              # Master application controller and state orchestrator
│       ├── audio.js            # Synthesized Web Audio API sound effects & alarms
│       ├── biometric_tracker.js# Millisecond-precision keystroke & mouse dynamics logger
│       ├── attention_viz.js    # 32x32 Self-Attention matrix heatmap canvas renderer
│       └── charts.js           # Multi-channel oscilloscope, radar chart & ROC visualizer
└── README.md                   # System documentation & mathematical formulas
```

---

## 🛡️ Default Credentials & Recovery
- **Default Operator Step-Up PIN**: `1234`
- **Reset Session**: Click the circular reset button in the top navigation bar or invoke `POST /api/reset_session`.
