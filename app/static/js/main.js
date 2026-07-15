/* ── main.js ── Chart.js + Upload UX + Flash dismiss ── */

// ─── Flash auto-dismiss ───────────────────────────────────────
document.querySelectorAll('.flash').forEach(el => {
  setTimeout(() => el.style.opacity = '0', 4000);
  setTimeout(() => el.remove(), 4300);
});

// ─── Image Upload & Preview ───────────────────────────────────
(function initUpload() {
  const zone    = document.getElementById('uploadZone');
  const input   = document.getElementById('xrayInput');
  const preview = document.getElementById('previewContainer');
  const img     = document.getElementById('previewImg');
  const form    = document.getElementById('predictForm');
  const spinner = document.getElementById('spinnerOverlay');

  if (!zone || !input) return;

  // Click zone → trigger file picker
  zone.addEventListener('click', () => input.click());

  // Drag-and-drop
  zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('dragover'); });
  zone.addEventListener('dragleave', ()  => zone.classList.remove('dragover'));
  zone.addEventListener('drop', e => {
    e.preventDefault();
    zone.classList.remove('dragover');
    const file = e.dataTransfer.files[0];
    if (file) showPreview(file);
  });

  input.addEventListener('change', () => {
    if (input.files[0]) showPreview(input.files[0]);
  });

  function showPreview(file) {
    const allowed = ['image/jpeg', 'image/jpg', 'image/png'];
    if (!allowed.includes(file.type)) {
      showAlert('Only JPG / JPEG / PNG files are allowed.', 'danger'); return;
    }
    if (file.size > 16 * 1024 * 1024) {
      showAlert('File size must be under 16 MB.', 'danger'); return;
    }
    const reader = new FileReader();
    reader.onload = e => {
      img.src = e.target.result;
      preview.style.display = 'block';
      zone.style.display = 'none';
    };
    reader.readAsDataURL(file);
    // Sync to real input if drag-dropped
    const dt = new DataTransfer();
    dt.items.add(file);
    input.files = dt.files;
  }

  // Show spinner on form submit
  if (form && spinner) {
    form.addEventListener('submit', () => {
      spinner.classList.add('show');
    });
  }
})();

// ─── Animate probability bars on load ────────────────────────
(function animateBars() {
  document.querySelectorAll('.prob-bar-fill[data-width]').forEach(bar => {
    const w = bar.dataset.width;
    setTimeout(() => { bar.style.width = w + '%'; }, 200);
  });
})();

// ─── SVG Confidence Ring ──────────────────────────────────────
(function drawRing() {
  const ring = document.getElementById('confRing');
  if (!ring) return;
  const val  = parseFloat(ring.dataset.conf || 0);
  const r    = 54;
  const circ = 2 * Math.PI * r;
  const dash = (val / 100) * circ;
  const color = ring.dataset.type === 'Pneumonia' ? '#e63946' : '#06d6a0';
  ring.innerHTML = `
    <svg width="130" height="130" viewBox="0 0 130 130">
      <circle cx="65" cy="65" r="${r}" fill="none" stroke="rgba(255,255,255,0.07)" stroke-width="10"/>
      <circle cx="65" cy="65" r="${r}" fill="none" stroke="${color}" stroke-width="10"
        stroke-dasharray="${dash.toFixed(1)} ${circ.toFixed(1)}"
        stroke-linecap="round"
        style="transition: stroke-dasharray 1.4s cubic-bezier(0.4,0,0.2,1); transform:rotate(-90deg); transform-origin:center"/>
    </svg>
    <div style="position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center">
      <span class="conf-text" style="color:${color}">${val.toFixed(1)}%</span>
      <span class="conf-label">Confidence</span>
    </div>`;
  ring.style.position = 'relative';
  ring.style.display  = 'inline-block';
})();

// ─── Dashboard Charts (Chart.js) ──────────────────────────────
(function initCharts() {
  if (typeof Chart === 'undefined') return;

  Chart.defaults.color = '#8da9c4';
  Chart.defaults.font.family = 'Inter, sans-serif';

  // Fetch stats from API
  fetch('/api/stats')
    .then(r => r.json())
    .then(data => {
      buildLineChart(data);
      buildDonutChart(data.donut);
    })
    .catch(() => {});

  function buildLineChart(data) {
    const ctx = document.getElementById('lineChart');
    if (!ctx) return;
    new Chart(ctx, {
      type: 'line',
      data: {
        labels: data.labels,
        datasets: [
          {
            label: 'Normal',
            data: data.normal,
            borderColor: '#06d6a0',
            backgroundColor: 'rgba(6,214,160,0.08)',
            tension: 0.4, fill: true, pointRadius: 4,
            pointBackgroundColor: '#06d6a0',
          },
          {
            label: 'Pneumonia',
            data: data.pneumonia,
            borderColor: '#e63946',
            backgroundColor: 'rgba(230,57,70,0.08)',
            tension: 0.4, fill: true, pointRadius: 4,
            pointBackgroundColor: '#e63946',
          },
        ],
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: {
          legend: { position: 'top', labels: { boxWidth: 10, padding: 16 } },
          tooltip: { mode: 'index', intersect: false },
        },
        scales: {
          x: { grid: { color: 'rgba(255,255,255,0.04)' } },
          y: { grid: { color: 'rgba(255,255,255,0.04)' }, beginAtZero: true, ticks: { precision: 0 } },
        },
      },
    });
  }

  function buildDonutChart(donut) {
    const ctx = document.getElementById('donutChart');
    if (!ctx) return;
    const total = (donut.normal || 0) + (donut.pneumonia || 0);
    if (total === 0) { ctx.parentElement.innerHTML = '<p style="color:var(--text-muted);text-align:center;padding:40px">No data yet</p>'; return; }
    new Chart(ctx, {
      type: 'doughnut',
      data: {
        labels: ['Normal', 'Pneumonia'],
        datasets: [{
          data: [donut.normal, donut.pneumonia],
          backgroundColor: ['rgba(6,214,160,0.75)', 'rgba(230,57,70,0.75)'],
          borderColor: ['#06d6a0', '#e63946'],
          borderWidth: 2, hoverOffset: 8,
        }],
      },
      options: {
        responsive: true, maintainAspectRatio: false, cutout: '72%',
        plugins: {
          legend: { position: 'bottom', labels: { boxWidth: 12, padding: 16 } },
          tooltip: {
            callbacks: {
              label: ctx => `${ctx.label}: ${ctx.parsed} (${((ctx.parsed/total)*100).toFixed(1)}%)`
            }
          }
        },
      },
    });
  }
})();

// ─── History table search ─────────────────────────────────────
(function initSearch() {
  const input = document.getElementById('tableSearch');
  if (!input) return;
  input.addEventListener('input', () => {
    const q = input.value.toLowerCase();
    document.querySelectorAll('.ai-table tbody tr').forEach(row => {
      row.style.display = row.textContent.toLowerCase().includes(q) ? '' : 'none';
    });
  });
})();

// ─── Helper: show inline alert ────────────────────────────────
function showAlert(msg, type) {
  const div = document.createElement('div');
  div.className = `flash flash-${type}`;
  div.innerHTML = `<i class="fas fa-exclamation-circle"></i> ${msg}`;
  const container = document.querySelector('.flash-container');
  if (container) { container.prepend(div); setTimeout(() => div.remove(), 4000); }
}
