document.addEventListener('DOMContentLoaded', () => {
    // ═══════════════════════════════════════
    // Navigation & Routing
    // ═══════════════════════════════════════
    const navItems = document.querySelectorAll('.nav-item');
    const panels = document.querySelectorAll('.panel');
    const pageTitle = document.getElementById('page-title');
    const pageSubtitle = document.getElementById('page-subtitle');
    const langSelect = document.getElementById('lang');
    const sidebar = document.getElementById('sidebar');
    const menuToggle = document.getElementById('menu-toggle');

    const routes = {
        'disease-panel': { title: 'Disease Detection', sub: 'AI-Powered Crop Diagnostics' },
        'price-panel':   { title: 'Price Analytics', sub: 'Market Forecasting & Trends' },
        'scheme-panel':  { title: 'Scheme Advisor', sub: 'Government Scheme RAG Search' },
        'pesticide-panel': { title: 'Pesticide Guide', sub: 'Crop Protection & Fertilizers' },
        'chat-panel':    { title: 'Ask KrishiMitra', sub: 'Intelligent Farming Assistant' }
    };

    navItems.forEach(item => {
        item.addEventListener('click', () => {
            navItems.forEach(n => n.classList.remove('active'));
            item.classList.add('active');

            const targetId = item.getAttribute('data-target');
            panels.forEach(p => p.classList.remove('active'));
            document.getElementById(targetId).classList.add('active');

            pageTitle.textContent = routes[targetId].title;
            pageSubtitle.textContent = routes[targetId].sub;

            // Close mobile sidebar
            sidebar.classList.remove('open');
        });
    });

    // ═══════════════════════════════════════
    // Safe Fetch Helper (with timeout + error handling)
    // ═══════════════════════════════════════
    async function safeFetch(url, options = {}) {
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), 15000);
        try {
            const res = await fetch(url, { ...options, signal: controller.signal });
            clearTimeout(timeout);
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            return await res.json();
        } catch (err) {
            clearTimeout(timeout);
            console.warn(`[safeFetch] ${url} failed:`, err.message);
            return null;
        }
    }

    // ═══════════════════════════════════════
    // Fallback data when server is unreachable
    // ═══════════════════════════════════════
    const FALLBACK_COMMODITIES = ["Wheat", "Rice", "Tomato", "Onion", "Potato", "Soyabean", "Gram (Chana)", "Maize", "Green Chilli", "Banana", "Apple", "Cotton"];
    const FALLBACK_CROPS = ["Rice", "Wheat", "Tomato", "Potato", "Cotton", "Soybean", "Groundnut", "Maize"];

    // Auto-load dropdowns on page load
    loadCommodities();
    loadCrops();

    // Mobile menu toggle
    if (menuToggle) {
        menuToggle.addEventListener('click', () => sidebar.classList.toggle('open'));
    }

    // Close sidebar on outside click (mobile)
    document.addEventListener('click', (e) => {
        if (window.innerWidth <= 768 && sidebar.classList.contains('open') && !sidebar.contains(e.target) && !menuToggle.contains(e.target)) {
            sidebar.classList.remove('open');
        }
    });

    // ═══════════════════════════════════════
    // Loader
    // ═══════════════════════════════════════
    const showLoader = () => document.getElementById('loader').classList.remove('hidden');
    const hideLoader = () => document.getElementById('loader').classList.add('hidden');

    // ═══════════════════════════════════════
    // Module 1: Disease Prediction
    // ═══════════════════════════════════════
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('leaf-image');
    const imagePreview = document.getElementById('image-preview');
    const btnPredict = document.getElementById('btn-predict-disease');
    let selectedImage = null;

    dropZone.addEventListener('click', () => fileInput.click());

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) handleFile(e.target.files[0]);
    });

    ['dragover', 'dragenter'].forEach(evt => dropZone.addEventListener(evt, e => {
        e.preventDefault();
        e.stopPropagation();
        dropZone.style.borderColor = 'var(--accent-green)';
    }));

    ['dragleave', 'drop'].forEach(evt => dropZone.addEventListener(evt, e => {
        e.preventDefault();
        e.stopPropagation();
        dropZone.style.borderColor = '';
    }));

    dropZone.addEventListener('drop', (e) => {
        if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
    });

    function handleFile(file) {
        selectedImage = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            imagePreview.classList.remove('hidden');
            dropZone.classList.add('hidden');
            btnPredict.disabled = false;
        };
        reader.readAsDataURL(file);
    }

    btnPredict.addEventListener('click', async () => {
        if (!selectedImage) return;
        showLoader();
        const formData = new FormData();
        formData.append('image', selectedImage);
        formData.append('language', langSelect.value);

        try {
            const res = await fetch('/api/disease/predict', { method: 'POST', body: formData });
            const data = await res.json();

            document.getElementById('disease-results').classList.remove('hidden');
            document.getElementById('res-disease').textContent = data.disease || '—';
            document.getElementById('res-crop').textContent = data.crop || '—';
            document.getElementById('res-confidence').textContent = data.confidence || '—';

            const statusEl = document.getElementById('res-status');
            statusEl.textContent = data.status || '—';
            statusEl.style.color = data.status === 'Healthy' ? 'var(--accent-green)' : 'var(--accent-rose)';

            document.getElementById('res-treatment').innerHTML = marked.parse(data.treatment_md || 'No treatment data available.');
        } catch (err) {
            console.error('Disease prediction error:', err);
            alert('Error connecting to disease prediction service.');
        }
        hideLoader();
    });

    // ═══════════════════════════════════════
    // Module 2: Price Analytics
    // ═══════════════════════════════════════
    const comSelect = document.getElementById('price-commodity');
    const stateSelect = document.getElementById('price-state');
    const marketSelect = document.getElementById('price-market');

    async function loadCommodities() {
        const data = await safeFetch('/api/price/commodities');
        const commodities = (data && data.commodities && data.commodities.length > 0)
            ? data.commodities
            : FALLBACK_COMMODITIES;
        comSelect.innerHTML = '<option value="">Select commodity...</option>' +
            commodities.map(c => `<option value="${c}">${c}</option>`).join('');
    }

    comSelect.addEventListener('change', async () => {
        const val = comSelect.value;
        if (!val) return;
        try {
            const res = await fetch(`/api/price/states?commodity=${encodeURIComponent(val)}`);
            const data = await res.json();
            stateSelect.innerHTML = '<option value="">All States</option>' +
                data.states.map(s => `<option value="${s}">${s}</option>`).join('');
            marketSelect.innerHTML = '<option value="">All Markets</option>';
        } catch (e) {}
    });

    stateSelect.addEventListener('change', async () => {
        const com = comSelect.value, st = stateSelect.value;
        if (!com || !st) return;
        try {
            const res = await fetch(`/api/price/markets?commodity=${encodeURIComponent(com)}&state=${encodeURIComponent(st)}`);
            const data = await res.json();
            marketSelect.innerHTML = '<option value="">All Markets</option>' +
                data.markets.map(m => `<option value="${m}">${m}</option>`).join('');
        } catch (e) {}
    });

    document.getElementById('btn-analyze-price').addEventListener('click', async () => {
        if (!comSelect.value) return alert('Please select a commodity first.');
        showLoader();
        try {
            const req = {
                commodity: comSelect.value,
                state: stateSelect.value,
                market: marketSelect.value,
                days: 90
            };
            const res = await fetch('/api/price/analyze', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(req)
            });
            const data = await res.json();
            if (data.error) throw new Error(data.error);

            document.getElementById('price-results').classList.remove('hidden');
            document.getElementById('res-price-summary').innerHTML = marked.parse(data.prediction_summary || '');

            // Render Plotly chart with dark theme
            if (data.trend_chart) {
                const fig = JSON.parse(data.trend_chart);
                const darkLayout = {
                    ...fig.layout,
                    margin: { l: 50, r: 20, t: 20, b: 40 },
                    paper_bgcolor: 'rgba(0,0,0,0)',
                    plot_bgcolor: 'rgba(0,0,0,0)',
                    font: { color: '#94a3b8', family: 'Inter' },
                    xaxis: { ...fig.layout?.xaxis, gridcolor: 'rgba(255,255,255,0.05)', linecolor: 'rgba(255,255,255,0.1)' },
                    yaxis: { ...fig.layout?.yaxis, gridcolor: 'rgba(255,255,255,0.05)', linecolor: 'rgba(255,255,255,0.1)' },
                    legend: { ...fig.layout?.legend, font: { color: '#94a3b8' } }
                };
                Plotly.newPlot('plot-trend', fig.data, darkLayout, { responsive: true, displayModeBar: false });
            }

            // Table
            const tbody = document.querySelector('#price-table tbody');
            tbody.innerHTML = '';
            (data.current_prices || []).forEach(row => {
                const tr = document.createElement('tr');
                tr.innerHTML = `<td>${row.arrival_date || '—'}</td><td>${row.state || '—'}</td><td>${row.market || '—'}</td><td>₹${row.modal_price?.toLocaleString() || '—'}</td>`;
                tbody.appendChild(tr);
            });
        } catch (e) {
            alert('Error: ' + e.message);
        }
        hideLoader();
    });

    // ═══════════════════════════════════════
    // Module 3: Scheme Advisory
    // ═══════════════════════════════════════
    document.getElementById('btn-ask-scheme').addEventListener('click', async () => {
        const q = document.getElementById('scheme-query').value.trim();
        if (!q) return;
        showLoader();
        try {
            const res = await fetch('/api/scheme/ask', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ question: q, language: langSelect.value })
            });
            const data = await res.json();
            document.getElementById('scheme-result-container').classList.remove('hidden');
            document.getElementById('res-scheme').innerHTML = marked.parse(data.answer || 'No answer received.');
        } catch (e) { console.error('Scheme query error:', e); }
        hideLoader();
    });

    document.getElementById('btn-list-schemes').addEventListener('click', async () => {
        showLoader();
        try {
            const res = await fetch('/api/scheme/list');
            const data = await res.json();
            document.getElementById('scheme-result-container').classList.remove('hidden');
            document.getElementById('res-scheme').innerHTML = marked.parse(data.markdown || 'No scheme data.');
        } catch (e) { console.error('List schemes error:', e); }
        hideLoader();
    });

    // ═══════════════════════════════════════
    // Module 4: Pesticide Advisor
    // ═══════════════════════════════════════
    const pestCrop = document.getElementById('pest-crop');
    const pestStage = document.getElementById('pest-stage');

    async function loadCrops() {
        const data = await safeFetch('/api/pesticide/crops');
        const crops = (data && data.crops && data.crops.length > 0)
            ? data.crops
            : FALLBACK_CROPS;
        pestCrop.innerHTML = '<option value="">Select crop...</option>' +
            crops.map(c => `<option value="${c}">${c}</option>`).join('');
    }

    pestCrop.addEventListener('change', async () => {
        const val = pestCrop.value;
        if (!val) return;
        try {
            const res = await fetch(`/api/pesticide/stages?crop=${encodeURIComponent(val)}`);
            const data = await res.json();
            pestStage.innerHTML = '<option value="">Any Stage</option>' +
                data.stages.map(s => `<option value="${s}">${s}</option>`).join('');
        } catch (e) {}
    });

    document.getElementById('btn-get-pest-rec').addEventListener('click', async () => {
        if (!pestCrop.value) return alert('Please select a crop first.');
        showLoader();
        try {
            const req = {
                crop: pestCrop.value,
                stage: pestStage.value,
                problem: document.getElementById('pest-problem').value,
                prefer_organic: document.getElementById('pest-organic').checked,
                language: langSelect.value
            };
            const res = await fetch('/api/pesticide/recommend', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(req)
            });
            const data = await res.json();
            document.getElementById('pest-result-container').classList.remove('hidden');
            document.getElementById('res-pest').innerHTML = marked.parse(data.recommendation || 'No recommendation available.');
        } catch (e) { console.error('Pesticide recommendation error:', e); }
        hideLoader();
    });

    // ═══════════════════════════════════════
    // Module 5: Chat
    // ═══════════════════════════════════════
    const chatInput = document.getElementById('chat-input');
    const chatMessages = document.getElementById('chat-messages');

    async function sendChat() {
        const val = chatInput.value.trim();
        if (!val) return;

        chatInput.value = '';
        addChatMessage(val, 'user');

        try {
            const res = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: val, language: langSelect.value })
            });
            const data = await res.json();
            addChatMessage(marked.parse(data.response || 'Sorry, I could not process that.'), 'bot', true);
        } catch (e) {
            addChatMessage('Connection error. Please try again.', 'bot');
        }
    }

    document.getElementById('btn-send-chat').addEventListener('click', sendChat);
    chatInput.addEventListener('keypress', (e) => { if (e.key === 'Enter') sendChat(); });

    function addChatMessage(text, role, isHtml = false) {
        const div = document.createElement('div');
        div.className = `message ${role}`;

        // Avatar
        const avatar = document.createElement('div');
        avatar.className = 'msg-avatar';
        avatar.innerHTML = role === 'bot'
            ? '<i class="fa-solid fa-seedling"></i>'
            : '<i class="fa-solid fa-user"></i>';

        // Bubble
        const bubble = document.createElement('div');
        bubble.className = 'msg-bubble markdown-body';

        if (isHtml) bubble.innerHTML = text;
        else bubble.textContent = text;

        div.appendChild(avatar);
        div.appendChild(bubble);
        chatMessages.appendChild(div);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    // ═══════════════════════════════════════
    // Keyboard shortcuts
    // ═══════════════════════════════════════
    document.addEventListener('keydown', (e) => {
        // Ctrl+1-5 for quick tab switching
        if (e.ctrlKey && e.key >= '1' && e.key <= '5') {
            e.preventDefault();
            const idx = parseInt(e.key) - 1;
            if (navItems[idx]) navItems[idx].click();
        }
    });
});
