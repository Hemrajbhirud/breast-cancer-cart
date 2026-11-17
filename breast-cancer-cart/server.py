"""
server.py - Web Interface for Breast Cancer Classification using CART.
Runs a local HTTP server providing a modern web UI and API.
"""

import http.server
import socketserver
import json
import io
import base64
import urllib.parse
from model import CARTModel, INPUT_FEATURES, MAX_DEPTH
import visualizations as viz

PORT = 8080

model = CARTModel()
model.train()  # Pre-train so all metrics and visualizations are immediately ready

def fig_to_base64(fig, dpi=120):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=dpi)
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode("utf-8")
    return f"data:image/png;base64,{data}"

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Breast Cancer Classification Using CART</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  :root {
    --navy: #1f2d3d;
    --navy-dark: #15202b;
    --accent: #2980b9;
    --accent-hover: #1f618d;
    --bg: #f4f6f8;
    --card: #ffffff;
    --text: #2c3e50;
    --text-muted: #7f8c8d;
    --border: #e2e8f0;
    --red: #c0392b;
    --green: #27ae60;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }
  body { background: var(--bg); color: var(--text); min-height: 100vh; display: flex; flex-direction: column; }
  
  .app-container { display: flex; flex: 1; min-height: 100vh; }
  
  /* Sidebar */
  .sidebar {
    width: 260px;
    background: var(--navy);
    color: white;
    display: flex;
    flex-direction: column;
    flex-shrink: 0;
  }
  .sidebar-header {
    padding: 32px 24px;
    text-align: center;
    border-bottom: 1px solid rgba(255,255,255,0.08);
  }
  .sidebar-header h1 {
    font-size: 1.4rem;
    font-weight: 700;
    letter-spacing: 0.5px;
    color: #ecf0f1;
    line-height: 1.3;
  }
  .sidebar-header .subtitle {
    font-size: 0.8rem;
    color: #95a5a6;
    margin-top: 6px;
    text-transform: uppercase;
    letter-spacing: 1px;
  }
  .nav-menu { display: flex; flex-direction: column; margin-top: 20px; }
  .nav-item {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 16px 24px;
    color: #bdc3c7;
    background: none;
    border: none;
    font-size: 0.95rem;
    font-weight: 500;
    cursor: pointer;
    text-align: left;
    transition: all 0.2s;
  }
  .nav-item:hover { background: rgba(255,255,255,0.05); color: #fff; }
  .nav-item.active { background: var(--accent); color: #fff; font-weight: 600; }
  
  /* Main Content */
  .main-content {
    flex: 1;
    display: flex;
    flex-direction: column;
    overflow-y: auto;
  }
  .disclaimer-banner {
    background: #fdecea;
    color: var(--red);
    padding: 10px 24px;
    font-size: 0.85rem;
    font-style: italic;
    border-bottom: 1px solid #fadbd8;
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .content-body {
    padding: 32px;
    max-width: 1200px;
    width: 100%;
    margin: 0 auto;
    flex: 1;
  }
  
  .page-title {
    font-size: 1.8rem;
    font-weight: 700;
    color: var(--navy);
    margin-bottom: 8px;
  }
  .page-subtitle {
    color: var(--text-muted);
    font-size: 0.95rem;
    margin-bottom: 24px;
    line-height: 1.5;
  }
  
  /* Cards */
  .stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 16px;
    margin-bottom: 28px;
  }
  .stat-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    text-align: center;
  }
  .stat-card .label { font-size: 0.85rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px; font-weight: 600; }
  .stat-card .value { font-size: 1.9rem; font-weight: 700; color: var(--accent); }
  .stat-card.benign .value { color: var(--green); }
  .stat-card.malignant .value { color: var(--red); }
  
  .card-box {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 24px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.03);
  }
  .card-box h3 { font-size: 1.15rem; font-weight: 600; margin-bottom: 16px; color: var(--navy); }
  
  /* Tables */
  .table-responsive { overflow-x: auto; }
  table { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
  th, td { padding: 12px 14px; text-align: left; border-bottom: 1px solid var(--border); }
  th { background: #f8fafc; color: var(--navy); font-weight: 600; }
  tr:hover td { background: #f8fafc; }
  
  /* Buttons */
  .btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    padding: 12px 24px;
    border-radius: 8px;
    font-size: 0.95rem;
    font-weight: 600;
    cursor: pointer;
    border: none;
    transition: all 0.2s;
  }
  .btn-primary { background: var(--accent); color: white; }
  .btn-primary:hover { background: var(--accent-hover); }
  .btn-secondary { background: #e2e8f0; color: var(--navy); }
  .btn-secondary:hover { background: #cbd5e1; }
  .btn-outline-green { background: #e8f8f0; color: var(--green); border: 1px solid #a3e4d7; }
  .btn-outline-green:hover { background: #d4efdf; }
  .btn-outline-red { background: #fdedec; color: var(--red); border: 1px solid #f5b7b1; }
  .btn-outline-red:hover { background: #fadbd8; }
  
  /* Forms */
  .form-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 18px;
    margin-bottom: 24px;
  }
  .form-group { display: flex; flex-direction: column; gap: 6px; }
  .form-group label { font-size: 0.85rem; font-weight: 600; color: var(--text); }
  .form-group .hint { font-size: 0.75rem; color: var(--text-muted); }
  .form-group input {
    padding: 10px 14px;
    border: 1px solid var(--border);
    border-radius: 8px;
    font-size: 0.95rem;
    transition: border-color 0.2s;
  }
  .form-group input:focus { outline: none; border-color: var(--accent); }
  
  /* Prediction Result Card */
  .result-banner {
    padding: 24px;
    border-radius: 12px;
    margin-top: 24px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    animation: fadeIn 0.3s ease-in-out;
  }
  .result-banner.benign { background: #e8f8f0; border: 2px solid var(--green); }
  .result-banner.malignant { background: #fdedec; border: 2px solid var(--red); }
  .result-badge {
    font-size: 2rem;
    font-weight: 800;
    letter-spacing: 1px;
    margin-bottom: 8px;
  }
  .benign .result-badge { color: var(--green); }
  .malignant .result-badge { color: var(--red); }
  .result-prob { font-size: 1.1rem; color: var(--text); font-weight: 500; }
  
  /* Images / Visualizations */
  .viz-container { text-align: center; margin-bottom: 24px; }
  .viz-img {
    max-width: 100%;
    height: auto;
    border-radius: 8px;
    border: 1px solid var(--border);
    box-shadow: 0 4px 6px rgba(0,0,0,0.03);
  }
  
  .tab-content { display: none; }
  .tab-content.active { display: block; }

  @keyframes fadeIn { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: translateY(0); } }
</style>
</head>
<body>

<div class="app-container">
  <!-- Sidebar -->
  <aside class="sidebar">
    <div class="sidebar-header">
      <h1>CART Classifier</h1>
      <div class="subtitle">Decision Tree ML Project</div>
    </div>
    <nav class="nav-menu">
      <button class="nav-item active" onclick="switchTab('dashboard')">
        <span>📊</span> Dashboard
      </button>
      <button class="nav-item" onclick="switchTab('training')">
        <span>⚙️</span> Model Training
      </button>
      <button class="nav-item" onclick="switchTab('visualizations')">
        <span>📈</span> Visualizations
      </button>
      <button class="nav-item" onclick="switchTab('prediction')">
        <span>🩺</span> Tumour Prediction
      </button>
    </nav>
  </aside>

  <!-- Main View -->
  <main class="main-content">
    <div class="disclaimer-banner">
      ⚠️ <strong>Disclaimer:</strong> Educational project only — predictions are NOT a medical diagnosis. Consult a qualified physician for clinical care.
    </div>
    
    <div class="content-body">
      
      <!-- DASHBOARD TAB -->
      <section id="tab-dashboard" class="tab-content active">
        <h2 class="page-title">Dataset Dashboard</h2>
        <p class="page-subtitle">
          Classifies breast tumour samples as <strong>Benign</strong> (non-cancerous) or <strong>Malignant</strong> (cancerous) using the CART (Classification and Regression Trees) algorithm based on Wisconsin FNA measurements.
        </p>

        <div class="stats-grid">
          <div class="stat-card">
            <div class="label">Total Records</div>
            <div class="value" id="val-records">569</div>
          </div>
          <div class="stat-card">
            <div class="label">Total Features</div>
            <div class="value" id="val-features">30</div>
          </div>
          <div class="stat-card">
            <div class="label">Classes</div>
            <div class="value" id="val-classes">2</div>
          </div>
          <div class="stat-card benign">
            <div class="label">Benign (0)</div>
            <div class="value" id="val-benign">357</div>
          </div>
          <div class="stat-card malignant">
            <div class="label">Malignant (1)</div>
            <div class="value" id="val-malignant">212</div>
          </div>
        </div>

        <div class="card-box">
          <h3>Sample Records (First 5 Rows)</h3>
          <p class="page-subtitle" style="margin-bottom: 12px;">Target: <code>0 = Malignant</code>, <code>1 = Benign</code></p>
          <div class="table-responsive">
            <table id="sample-table">
              <thead><tr><th>Loading...</th></tr></thead>
              <tbody></tbody>
            </table>
          </div>
        </div>

        <div class="card-box">
          <h3>About the CART Algorithm</h3>
          <p style="line-height: 1.6; color: #475569;">
            CART builds a binary tree. At each node, it tests every candidate feature and split threshold to find the split that maximizes the reduction in <strong>Gini impurity</strong>:
            <br>
            <code>Gini = 1 - &Sigma; (p_i)&sup2;</code>
            <br><br>
            A Gini score of <code>0</code> signifies a pure node (all samples belong to a single class). In our implementation, <code>max_depth=4</code> is used to prevent overfitting and ensure interpretability.
          </p>
        </div>
      </section>

      <!-- MODEL TRAINING TAB -->
      <section id="tab-training" class="tab-content">
        <h2 class="page-title">Model Training & Evaluation</h2>
        <p class="page-subtitle">
          Split: 80% Train, 20% Test (Stratified, Random State = 42). Criterion: Gini Impurity (CART).
        </p>

        <div style="margin-bottom: 24px; display: flex; gap: 12px; align-items: center;">
          <button class="btn btn-primary" onclick="trainModel()" id="btn-train">
            🔄 Retrain CART Model
          </button>
          <span id="train-status" style="font-weight: 600; color: var(--green);">Model is trained and evaluated!</span>
        </div>

        <div class="stats-grid">
          <div class="stat-card">
            <div class="label">Test Accuracy</div>
            <div class="value" id="metric-acc">-</div>
          </div>
          <div class="stat-card">
            <div class="label">Precision (Malignant)</div>
            <div class="value" id="metric-prec">-</div>
          </div>
          <div class="stat-card">
            <div class="label">Recall (Malignant)</div>
            <div class="value" id="metric-rec">-</div>
          </div>
          <div class="stat-card">
            <div class="label">F1-Score</div>
            <div class="value" id="metric-f1">-</div>
          </div>
          <div class="stat-card">
            <div class="label">Train Accuracy</div>
            <div class="value" id="metric-train">-</div>
          </div>
        </div>

        <div class="card-box">
          <h3>Why Recall is the Critical Metric</h3>
          <p style="line-height: 1.6; color: #475569;">
            In medical diagnosis, <strong>Recall</strong> measures the fraction of malignant tumours correctly flagged. Missing a malignant tumour (False Negative) has severe health consequences, so maximizing recall while retaining high precision is essential.
          </p>
        </div>
      </section>

      <!-- VISUALIZATIONS TAB -->
      <section id="tab-visualizations" class="tab-content">
        <h2 class="page-title">Model Visualizations</h2>
        <p class="page-subtitle">Inspect the class distribution, confusion matrix, and full CART decision tree splits.</p>

        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 24px; margin-bottom: 24px;">
          <div class="card-box">
            <h3>Class Distribution</h3>
            <div class="viz-container">
              <img id="img-distribution" class="viz-img" src="" alt="Class Distribution">
            </div>
          </div>
          <div class="card-box">
            <h3>Confusion Matrix (Test Set)</h3>
            <div class="viz-container">
              <img id="img-confusion" class="viz-img" src="" alt="Confusion Matrix">
            </div>
          </div>
        </div>

        <div class="card-box">
          <h3>CART Decision Tree Structure</h3>
          <p style="margin-bottom: 16px; color: var(--text-muted); font-size: 0.9rem;">
            Shows the exact thresholds, Gini impurity, sample distribution, and dominant class at each node.
          </p>
          <div class="viz-container" style="overflow-x: auto;">
            <img id="img-tree" class="viz-img" style="min-width: 900px;" src="" alt="Decision Tree">
          </div>
        </div>
      </section>

      <!-- PREDICTION TAB -->
      <section id="tab-prediction" class="tab-content">
        <h2 class="page-title">Predict Tumour Classification</h2>
        <p class="page-subtitle">
          Input 8 core cell nuclei measurements from the fine-needle aspirate (FNA). The remaining 22 correlated features are automatically estimated.
        </p>

        <div style="margin-bottom: 20px; display: flex; gap: 12px; flex-wrap: wrap;">
          <button class="btn btn-outline-green" onclick="loadSample('benign')">
            🟢 Load Typical Benign Sample
          </button>
          <button class="btn btn-outline-red" onclick="loadSample('malignant')">
            🔴 Load Typical Malignant Sample
          </button>
        </div>

        <form id="predict-form" onsubmit="event.preventDefault(); submitPrediction();" class="card-box">
          <h3>Input Measurements</h3>
          <div class="form-grid" id="features-inputs">
            <!-- Inputs populated dynamically -->
          </div>
          
          <button type="submit" class="btn btn-primary" style="width: 100%; padding: 14px; font-size: 1.05rem;">
            🔍 Classify Tumour Sample
          </button>
        </form>

        <div id="prediction-result" style="display: none;"></div>
      </section>

    </div>
  </main>
</div>

<script>
let ranges = {};

function switchTab(name) {
  document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(el => el.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
  event.currentTarget.classList.add('active');
}

async function loadData() {
  const res = await fetch('/api/info');
  const data = await res.json();
  
  document.getElementById('val-records').innerText = data.info.records;
  document.getElementById('val-features').innerText = data.info.features;
  document.getElementById('val-classes').innerText = data.info.classes;
  document.getElementById('val-benign').innerText = data.info.benign;
  document.getElementById('val-malignant').innerText = data.info.malignant;

  // Build Sample table
  const thead = '<tr>' + data.sample_cols.map(c => `<th>${c}</th>`).join('') + '</tr>';
  const tbody = data.sample_rows.map(row => 
    '<tr>' + row.map(v => `<td>${typeof v === 'number' ? v.toFixed(3) : v}</td>`).join('') + '</tr>'
  ).join('');
  document.getElementById('sample-table').innerHTML = `<thead>${thead}</thead><tbody>${tbody}</tbody>`;

  // Load metrics
  updateMetrics(data.metrics);

  // Load visualizations
  document.getElementById('img-distribution').src = data.viz.distribution;
  document.getElementById('img-confusion').src = data.viz.confusion;
  document.getElementById('img-tree').src = data.viz.tree;

  // Load ranges for prediction
  ranges = data.ranges;
  buildInputs();
}

function updateMetrics(metrics) {
  document.getElementById('metric-acc').innerText = (metrics['Accuracy'] * 100).toFixed(1) + '%';
  document.getElementById('metric-prec').innerText = (metrics['Precision'] * 100).toFixed(1) + '%';
  document.getElementById('metric-rec').innerText = (metrics['Recall'] * 100).toFixed(1) + '%';
  document.getElementById('metric-f1').innerText = (metrics['F1-score'] * 100).toFixed(1) + '%';
  document.getElementById('metric-train').innerText = (metrics['Train accuracy'] * 100).toFixed(1) + '%';
}

async function trainModel() {
  const btn = document.getElementById('btn-train');
  const status = document.getElementById('train-status');
  btn.innerText = "⏳ Training...";
  btn.disabled = true;

  const res = await fetch('/api/train', { method: 'POST' });
  const data = await res.json();
  updateMetrics(data.metrics);
  document.getElementById('img-confusion').src = data.viz.confusion;
  document.getElementById('img-tree').src = data.viz.tree;

  btn.innerText = "🔄 Retrain CART Model";
  btn.disabled = false;
  status.innerText = "Model retrained successfully!";
}

function buildInputs() {
  const container = document.getElementById('features-inputs');
  container.innerHTML = '';
  for (const [key, [min, max, mean]] of Object.entries(ranges)) {
    const div = document.createElement('div');
    div.className = 'form-group';
    div.innerHTML = `
      <label for="inp-${key}">${key}</label>
      <input type="number" step="any" id="inp-${key}" name="${key}" value="${mean.toFixed(3)}" required>
      <div class="hint">Range: ${min.toFixed(2)} - ${max.toFixed(2)} (mean: ${mean.toFixed(2)})</div>
    `;
    container.appendChild(div);
  }
}

function loadSample(type) {
  fetch('/api/samples?type=' + type)
    .then(r => r.json())
    .then(vals => {
      for (const [key, val] of Object.entries(vals)) {
        const inp = document.getElementById('inp-' + key);
        if (inp) inp.value = Number(val).toFixed(4);
      }
      submitPrediction();
    });
}

async function submitPrediction() {
  const inputs = {};
  for (const key of Object.keys(ranges)) {
    const el = document.getElementById('inp-' + key);
    inputs[key] = parseFloat(el.value);
  }

  const res = await fetch('/api/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(inputs)
  });
  const data = await res.json();

  const isBenign = data.prediction.toLowerCase() === 'benign';
  const resultDiv = document.getElementById('prediction-result');
  resultDiv.className = `result-banner ${isBenign ? 'benign' : 'malignant'}`;
  resultDiv.style.display = 'block';
  resultDiv.innerHTML = `
    <div class="result-badge">${isBenign ? '🟢 BENIGN' : '🔴 MALIGNANT'}</div>
    <div class="result-prob">Confidence: <strong>${(data.probability * 100).toFixed(1)}%</strong></div>
    <p style="margin-top: 10px; font-size: 0.9rem; color: #475569;">
      ${isBenign ? 'Non-cancerous cells detected based on fine-needle aspirate cell morphology.' : 'Cancerous characteristics detected. Recommended for immediate clinical evaluation.'}
    </p>
  `;
  resultDiv.scrollIntoView({ behavior: 'smooth' });
}

window.onload = loadData;
</script>

</body>
</html>
"""

class RequestHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        # Concise logging
        pass

    def do_GET(self):
        url = urllib.parse.urlparse(self.path)
        path = url.path

        if path == "/" or path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))
            return

        elif path == "/api/info":
            info = model.dataset_info()
            sample = model.sample_records()
            sample_cols = list(sample.columns)
            sample_rows = sample.values.tolist()

            # Render charts
            fig_dist = viz.class_distribution_figure(model.df, model.class_names)
            dist_b64 = fig_to_base64(fig_dist)

            fig_cm = viz.confusion_matrix_figure(model.cm, model.class_names)
            cm_b64 = fig_to_base64(fig_cm)

            fig_tree = viz.decision_tree_figure(model.clf, model.feature_names, model.class_names)
            tree_b64 = fig_to_base64(fig_tree, dpi=130)

            payload = {
                "info": info,
                "sample_cols": sample_cols,
                "sample_rows": sample_rows,
                "metrics": model.metrics,
                "ranges": model.feature_ranges(),
                "viz": {
                    "distribution": dist_b64,
                    "confusion": cm_b64,
                    "tree": tree_b64
                }
            }
            self._send_json(payload)
            return

        elif path == "/api/samples":
            query = urllib.parse.parse_qs(url.query)
            stype = query.get("type", ["benign"])[0]
            label = 1 if stype == "benign" else 0
            series = model.class_example(label)
            self._send_json(series.to_dict())
            return

        self.send_error(404, "Not Found")

    def do_POST(self):
        url = urllib.parse.urlparse(self.path)
        path = url.path

        if path == "/api/train":
            metrics = model.train()
            fig_cm = viz.confusion_matrix_figure(model.cm, model.class_names)
            cm_b64 = fig_to_base64(fig_cm)

            fig_tree = viz.decision_tree_figure(model.clf, model.feature_names, model.class_names)
            tree_b64 = fig_to_base64(fig_tree, dpi=130)

            self._send_json({
                "metrics": metrics,
                "viz": {
                    "confusion": cm_b64,
                    "tree": tree_b64
                }
            })
            return

        elif path == "/api/predict":
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length)
            data = json.loads(raw.decode("utf-8"))
            pred_class, prob = model.predict(data)
            self._send_json({
                "prediction": pred_class,
                "probability": prob
            })
            return

        self.send_error(404, "Not Found")

    def _send_json(self, data):
        out = json.dumps(data).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)


def run_server():
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", PORT), RequestHandler) as httpd:
        print(f"Serving Breast Cancer CART application at http://localhost:{PORT}")
        httpd.serve_forever()


if __name__ == "__main__":
    run_server()
