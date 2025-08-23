// Movie Vector Database Benchmark JavaScript

class MovieBenchmarkUI {
    constructor() {
        this.isInitialized = false;
        this.benchmarkRunning = false;
        this.searchResults = [];
        this.benchmarkResults = [];
        this.charts = {};
        
        this.init();
    }

    init() {
        this.checkSystemStatus();
        this.updateDatabaseAvailability();
        
        // Start status polling
        this.startStatusPolling();
    }

    async checkSystemStatus() {
        try {
            const response = await fetch('/api/status');
            const status = await response.json();
            this.updateStatusUI(status);
            
            if (status.status === 'ready') {
                this.isInitialized = true;
                this.hideInitializeSection();
            }
        } catch (error) {
            console.error('Failed to check system status:', error);
            this.updateStatusUI({
                status: 'error',
                progress: 0,
                message: 'Failed to connect to server'
            });
        }
    }

    async updateDatabaseAvailability() {
        try {
            const response = await fetch('/api/databases');
            const databases = await response.json();
            
            // Update search dropdown
            const searchSelect = document.getElementById('searchDatabase');
            searchSelect.innerHTML = '';
            
            // Update benchmark checkboxes
            const checkboxContainer = document.getElementById('databaseCheckboxes');
            checkboxContainer.innerHTML = '';
            
            Object.entries(databases).forEach(([key, db]) => {
                // Search dropdown
                const option = document.createElement('option');
                option.value = key;
                option.textContent = `${db.name} ${db.available ? '' : '(Unavailable)'}`;
                option.disabled = !db.available;
                searchSelect.appendChild(option);
                
                // Benchmark checkboxes
                const label = document.createElement('label');
                const checkbox = document.createElement('input');
                checkbox.type = 'checkbox';
                checkbox.value = key;
                checkbox.disabled = !db.available;
                checkbox.checked = db.available; // Check all available databases
                
                label.appendChild(checkbox);
                label.appendChild(document.createTextNode(`${db.name} ${db.available ? '' : '(Unavailable)'}`));
                checkboxContainer.appendChild(label);
            });
            
        } catch (error) {
            console.error('Failed to update database availability:', error);
        }
    }

    updateStatusUI(status) {
        const indicator = document.getElementById('statusIndicator');
        const statusText = document.getElementById('statusText');
        const progressBar = document.getElementById('progressBar');
        const progressFill = document.getElementById('progressFill');
        const statusDot = indicator.querySelector('.status-dot');
        
        // Update status dot
        statusDot.className = `status-dot ${status.status}`;
        statusText.textContent = status.message;
        
        // Update progress bar
        if (status.progress > 0 && status.status !== 'ready') {
            progressBar.style.display = 'block';
            progressFill.style.width = `${status.progress * 100}%`;
        } else {
            progressBar.style.display = 'none';
        }
    }

    startStatusPolling() {
        setInterval(async () => {
            if (this.benchmarkRunning || !this.isInitialized) {
                await this.checkSystemStatus();
            }
        }, 2000);
    }

    hideInitializeSection() {
        const initSection = document.getElementById('initializeSection');
        initSection.style.display = 'none';
    }

    showInitializeSection() {
        const initSection = document.getElementById('initializeSection');
        initSection.style.display = 'block';
    }
}

// Initialize the UI
const ui = new MovieBenchmarkUI();

// Tab switching
function showTab(tabName) {
    // Hide all tabs
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });
    
    // Remove active class from all buttons
    document.querySelectorAll('.tab-button').forEach(button => {
        button.classList.remove('active');
    });
    
    // Show selected tab
    document.getElementById(tabName + 'Tab').classList.add('active');
    
    // Add active class to clicked button
    event.target.classList.add('active');
}

// Initialize system
async function initializeSystem() {
    try {
        const response = await fetch('/api/initialize', { method: 'POST' });
        const result = await response.json();
        
        if (response.ok) {
            ui.updateStatusUI({
                status: 'loading',
                progress: 0.1,
                message: 'Initializing system...'
            });
            ui.benchmarkRunning = true;
            
            // Start polling for initialization status
            const pollInit = setInterval(async () => {
                const statusResponse = await fetch('/api/status');
                const status = await statusResponse.json();
                ui.updateStatusUI(status);
                
                if (status.status === 'ready') {
                    clearInterval(pollInit);
                    ui.isInitialized = true;
                    ui.benchmarkRunning = false;
                    ui.hideInitializeSection();
                } else if (status.status === 'error') {
                    clearInterval(pollInit);
                    ui.benchmarkRunning = false;
                }
            }, 1000);
            
        } else {
            throw new Error(result.detail || 'Initialization failed');
        }
    } catch (error) {
        console.error('Initialization failed:', error);
        alert('Failed to initialize system: ' + error.message);
    }
}

// Search movies
async function searchMovies() {
    if (!ui.isInitialized) {
        alert('System not initialized. Please click Initialize System first.');
        return;
    }
    
    const query = document.getElementById('searchQuery').value.trim();
    const database = document.getElementById('searchDatabase').value;
    const topK = parseInt(document.getElementById('topK').value);
    
    if (!query) {
        alert('Please enter a search query');
        return;
    }
    
    try {
        const response = await fetch('/api/search', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query, database, top_k: topK })
        });
        
        const result = await response.json();
        
        if (response.ok) {
            displaySearchResults(result, query, database);
        } else {
            throw new Error(result.detail || 'Search failed');
        }
        
    } catch (error) {
        console.error('Search failed:', error);
        alert('Search failed: ' + error.message);
    }
}

function displaySearchResults(result, query, database) {
    const resultsContainer = document.getElementById('searchResults');
    const searchTime = document.getElementById('searchTime');
    const resultCount = document.getElementById('resultCount');
    const resultsList = document.getElementById('resultsList');
    
    // Update stats
    searchTime.textContent = `${(result.query_time * 1000).toFixed(2)}ms`;
    resultCount.textContent = `${result.total_results} results`;
    
    // Clear previous results
    resultsList.innerHTML = '';
    
    // Display results
    result.movies.forEach((movie, index) => {
        const card = document.createElement('div');
        card.className = 'result-card';
        
        card.innerHTML = `
            <div class="result-title">${movie.title || 'Unknown Title'}</div>
            <div class="result-details">
                <div class="result-detail">
                    <span class="label">Score:</span>
                    <span class="value">${(movie.score || 0).toFixed(3)}</span>
                </div>
                <div class="result-detail">
                    <span class="label">Genres:</span>
                    <span class="value">${movie.genres || 'Unknown'}</span>
                </div>
                <div class="result-detail">
                    <span class="label">Movie ID:</span>
                    <span class="value">${movie.movieId || movie.id || 'N/A'}</span>
                </div>
                <div class="result-detail">
                    <span class="label">Rating:</span>
                    <span class="value">${movie.avg_rating ? movie.avg_rating.toFixed(1) : 'N/A'}</span>
                </div>
            </div>
        `;
        
        resultsList.appendChild(card);
    });
    
    resultsContainer.style.display = 'block';
}

// Run benchmark
async function runBenchmark() {
    if (!ui.isInitialized) {
        alert('System not initialized. Please click Initialize System first.');
        return;
    }
    
    // Get selected databases
    const checkboxes = document.querySelectorAll('#databaseCheckboxes input[type="checkbox"]:checked');
    const databases = Array.from(checkboxes).map(cb => cb.value);
    
    if (databases.length === 0) {
        alert('Please select at least one database to benchmark');
        return;
    }
    
    const sampleSize = parseInt(document.getElementById('sampleSize').value);
    const embeddingModel = document.getElementById('embeddingModel').value;
    
    try {
        const response = await fetch('/api/benchmark', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                databases,
                sample_size: sampleSize,
                embedding_model: embeddingModel
            })
        });
        
        const result = await response.json();
        
        if (response.ok) {
            ui.benchmarkRunning = true;
            ui.updateStatusUI({
                status: 'loading',
                progress: 0.1,
                message: 'Starting benchmark...'
            });
            
            // Start polling for benchmark results
            const pollBenchmark = setInterval(async () => {
                const statusResponse = await fetch('/api/benchmark/results');
                const status = await statusResponse.json();
                ui.updateStatusUI(status);
                
                if (status.status === 'completed' && status.results) {
                    clearInterval(pollBenchmark);
                    ui.benchmarkRunning = false;
                    displayBenchmarkResults(status.results);
                    // Switch to results tab
                    showTab('results');
                } else if (status.status === 'error') {
                    clearInterval(pollBenchmark);
                    ui.benchmarkRunning = false;
                }
            }, 2000);
            
        } else {
            throw new Error(result.detail || 'Benchmark failed to start');
        }
        
    } catch (error) {
        console.error('Benchmark failed:', error);
        alert('Failed to start benchmark: ' + error.message);
    }
}

function displayBenchmarkResults(results) {
    const resultsContainer = document.getElementById('benchmarkResults');
    const chartsContainer = document.getElementById('chartsContainer');
    
    // Create results table
    const table = document.createElement('table');
    table.className = 'results-table';
    
    // Table header
    const thead = document.createElement('thead');
    thead.innerHTML = `
        <tr>
            <th>Database</th>
            <th>Ingest Time (s)</th>
            <th>Throughput (vec/s)</th>
            <th>Avg Latency (ms)</th>
            <th>P95 Latency (ms)</th>
            <th>Recall@10</th>
            <th>Hit Rate</th>
            <th>Vectors</th>
        </tr>
    `;
    table.appendChild(thead);
    
    // Table body
    const tbody = document.createElement('tbody');
    results.forEach(result => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td><strong>${result.database.toUpperCase()}</strong></td>
            <td>${result.ingest_time}</td>
            <td>${result.throughput}</td>
            <td>${result.avg_latency}</td>
            <td>${result.p95_latency}</td>
            <td>${result.recall_at_10}</td>
            <td>${result.hit_rate}</td>
            <td>${result.total_vectors}</td>
        `;
        tbody.appendChild(row);
    });
    table.appendChild(tbody);
    
    // Clear previous results and add table
    resultsContainer.innerHTML = '';
    resultsContainer.appendChild(table);
    
    // Show charts
    chartsContainer.style.display = 'block';
    createCharts(results);
}

function createCharts(results) {
    const databases = results.map(r => r.database.toUpperCase());
    
    // Destroy existing charts
    Object.values(ui.charts).forEach(chart => {
        if (chart) chart.destroy();
    });
    
    // Ingest performance chart
    const ingestCtx = document.getElementById('ingestChart').getContext('2d');
    ui.charts.ingest = new Chart(ingestCtx, {
        type: 'bar',
        data: {
            labels: databases,
            datasets: [{
                label: 'Throughput (vectors/sec)',
                data: results.map(r => r.throughput),
                backgroundColor: 'rgba(37, 99, 235, 0.7)',
                borderColor: 'rgba(37, 99, 235, 1)',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'Ingestion Throughput'
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Vectors/Second'
                    }
                }
            }
        }
    });
    
    // Latency chart
    const latencyCtx = document.getElementById('latencyChart').getContext('2d');
    ui.charts.latency = new Chart(latencyCtx, {
        type: 'bar',
        data: {
            labels: databases,
            datasets: [{
                label: 'Avg Latency (ms)',
                data: results.map(r => r.avg_latency),
                backgroundColor: 'rgba(16, 185, 129, 0.7)',
                borderColor: 'rgba(16, 185, 129, 1)',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'Query Latency'
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Milliseconds'
                    }
                }
            }
        }
    });
    
    // Recall chart
    const recallCtx = document.getElementById('recallChart').getContext('2d');
    ui.charts.recall = new Chart(recallCtx, {
        type: 'bar',
        data: {
            labels: databases,
            datasets: [{
                label: 'Recall@10',
                data: results.map(r => r.recall_at_10),
                backgroundColor: 'rgba(245, 158, 11, 0.7)',
                borderColor: 'rgba(245, 158, 11, 1)',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'Recall@10'
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 1,
                    title: {
                        display: true,
                        text: 'Recall Score'
                    }
                }
            }
        }
    });
    
    // Hit rate chart
    const hitRateCtx = document.getElementById('hitRateChart').getContext('2d');
    ui.charts.hitRate = new Chart(hitRateCtx, {
        type: 'bar',
        data: {
            labels: databases,
            datasets: [{
                label: 'Hit Rate',
                data: results.map(r => r.hit_rate),
                backgroundColor: 'rgba(139, 92, 246, 0.7)',
                borderColor: 'rgba(139, 92, 246, 1)',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'Hit Rate'
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 1,
                    title: {
                        display: true,
                        text: 'Hit Rate'
                    }
                }
            }
        }
    });
    
    // Add a comprehensive comparison chart
    createComparisonChart(results);
}

function createComparisonChart(results) {
    // Add a comprehensive comparison chart container if not exists
    const chartsContainer = document.getElementById('chartsContainer');
    
    // Check if comparison chart already exists, if not create it
    let comparisonContainer = document.getElementById('comparisonChartContainer');
    if (!comparisonContainer) {
        comparisonContainer = document.createElement('div');
        comparisonContainer.id = 'comparisonChartContainer';
        comparisonContainer.className = 'chart-row';
        comparisonContainer.innerHTML = `
            <div class="chart-container full-width">
                <canvas id="comparisonChart"></canvas>
            </div>
        `;
        chartsContainer.appendChild(comparisonContainer);
    }
    
    // Normalize metrics for radar chart (0-1 scale)
    const metrics = ['throughput', 'avg_latency', 'recall_at_10', 'hit_rate'];
    const maxValues = {
        throughput: Math.max(...results.map(r => r.throughput)),
        avg_latency: Math.max(...results.map(r => r.avg_latency)),
        recall_at_10: 1, // Already normalized
        hit_rate: 1 // Already normalized
    };
    
    const datasets = results.map((result, index) => {
        const colors = [
            'rgba(37, 99, 235, 0.4)',   // Blue
            'rgba(16, 185, 129, 0.4)',  // Green  
            'rgba(245, 158, 11, 0.4)',  // Orange
            'rgba(139, 92, 246, 0.4)',  // Purple
            'rgba(239, 68, 68, 0.4)',   // Red
        ];
        
        return {
            label: result.database.toUpperCase(),
            data: [
                result.throughput / maxValues.throughput,
                1 - (result.avg_latency / maxValues.avg_latency), // Inverted (lower is better)
                result.recall_at_10,
                result.hit_rate
            ],
            backgroundColor: colors[index % colors.length],
            borderColor: colors[index % colors.length].replace('0.4', '1'),
            borderWidth: 2,
            pointRadius: 4
        };
    });
    
    const comparisonCtx = document.getElementById('comparisonChart').getContext('2d');
    if (ui.charts.comparison) ui.charts.comparison.destroy();
    
    ui.charts.comparison = new Chart(comparisonCtx, {
        type: 'radar',
        data: {
            labels: ['Throughput', 'Query Speed', 'Recall@10', 'Hit Rate'],
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'Overall Performance Comparison',
                    font: { size: 16 }
                },
                legend: {
                    display: true,
                    position: 'bottom'
                }
            },
            scales: {
                r: {
                    beginAtZero: true,
                    max: 1,
                    ticks: {
                        stepSize: 0.2
                    }
                }
            }
        }
    });
}