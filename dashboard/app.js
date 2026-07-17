// Dashboard metrics loader and interactive controller
let modelShareChartInstance = null;
let cacheChartInstance = null;

document.addEventListener("DOMContentLoaded", () => {
    // Initial fetch
    fetchMetrics();
    
    // Setup event listeners
    document.getElementById("refreshBtn").addEventListener("click", () => {
        fetchMetrics(true);
    });
    
    document.getElementById("clearCacheBtn").addEventListener("click", clearCache);

    // Auto refresh every 8 seconds
    setInterval(() => {
        fetchMetrics(false);
    }, 8000);
});

// Toast notification trigger
function showToast(message, isSuccess = true) {
    const toast = document.getElementById("toast");
    const toastMsg = document.getElementById("toastMessage");
    const toastIcon = document.getElementById("toastIcon");

    toastMsg.textContent = message;
    if (isSuccess) {
        toastIcon.className = "fa-solid fa-circle-check";
        toastIcon.style.color = "var(--emerald-light)";
    } else {
        toastIcon.className = "fa-solid fa-circle-exclamation";
        toastIcon.style.color = "var(--rose-light)";
    }

    toast.classList.remove("hidden");
    setTimeout(() => {
        toast.classList.add("hidden");
    }, 3000);
}

// Fetch metrics from API
async function fetchMetrics(manualTrigger = false) {
    try {
        const response = await fetch("/api/metrics");
        if (!response.ok) throw new Error("Failed to fetch metrics");
        const data = await response.json();

        updateStats(data);
        updateCharts(data);
        updateTable(data.history);

        if (manualTrigger) {
            showToast("Dashboard metrics refreshed");
        }
    } catch (err) {
        console.error("Error updating metrics dashboard:", err);
        if (manualTrigger) {
            showToast("Failed to refresh metrics", false);
        }
    }
}

// Update text statistics cards
function updateStats(data) {
    // 1. Cost Savings
    document.getElementById("costSavings").textContent = `$${data.cost_savings.toFixed(4)}`;
    document.getElementById("actualCost").textContent = `Total Cost: $${data.total_cost.toFixed(4)}`;

    // 2. Cache hit rate
    const hitRate = data.cache_hit_rate;
    document.getElementById("cacheHitRate").textContent = `${hitRate.toFixed(1)}%`;
    document.getElementById("cacheProgressBar").style.width = `${hitRate}%`;
    document.getElementById("cacheBreakdown").textContent = `Exact: ${data.exact_hits} | Semantic: ${data.semantic_hits}`;

    // 3. Requests & Tokens
    document.getElementById("totalRequests").textContent = data.total_requests;
    document.getElementById("totalTokens").textContent = `Tokens: ${data.total_tokens.toLocaleString()}`;

    // 4. Average Latency
    document.getElementById("avgLatency").textContent = `${data.average_latency_ms.toFixed(0)}ms`;

    // 5. Success rate
    document.getElementById("successRate").textContent = `${data.success_rate.toFixed(1)}%`;
    document.getElementById("feedbackCount").textContent = `Feedback: ${data.feedback_total} runs`;

    // History count badge
    document.getElementById("historyCount").textContent = `Last ${data.history ? data.history.length : 0} requests`;
}

// Build or update ChartJS instances
function updateCharts(data) {
    // 1. Model Share Chart
    const modelShareCtx = document.getElementById("modelShareChart").getContext("2d");
    const t1Count = data.tier1_count;
    const t2Count = data.tier2_count;

    if (modelShareChartInstance) {
        modelShareChartInstance.data.datasets[0].data = [t1Count, t2Count];
        modelShareChartInstance.update();
    } else {
        modelShareChartInstance = new Chart(modelShareCtx, {
            type: 'doughnut',
            data: {
                labels: ['Tier 1 (Low Cost)', 'Tier 2 (High Quality)'],
                datasets: [{
                    data: [t1Count, t2Count],
                    backgroundColor: ['rgba(99, 102, 241, 0.75)', 'rgba(168, 85, 247, 0.75)'],
                    borderColor: ['#6366f1', '#a855f7'],
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: { color: '#f8fafc', font: { family: 'Outfit' } }
                    }
                }
            }
        });
    }

    // 2. Cache Hit vs Miss Chart
    const cacheCtx = document.getElementById("cacheChart").getContext("2d");
    const exact = data.exact_hits;
    const semantic = data.semantic_hits;
    const misses = data.total_requests - (exact + semantic);

    if (cacheChartInstance) {
        cacheChartInstance.data.datasets[0].data = [exact, semantic, misses >= 0 ? misses : 0];
        cacheChartInstance.update();
    } else {
        cacheChartInstance = new Chart(cacheCtx, {
            type: 'bar',
            data: {
                labels: ['Exact Cache', 'Semantic Cache', 'Cache Miss'],
                datasets: [{
                    label: 'Requests',
                    data: [exact, semantic, misses >= 0 ? misses : 0],
                    backgroundColor: [
                        'rgba(16, 185, 129, 0.75)', // emerald
                        'rgba(59, 130, 246, 0.75)', // blue
                        'rgba(100, 116, 139, 0.75)' // slate
                    ],
                    borderColor: ['#10b981', '#3b82f6', '#64748b'],
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    x: { ticks: { color: '#94a3b8', font: { family: 'Outfit' } } },
                    y: { ticks: { color: '#94a3b8', font: { family: 'Outfit' } }, grid: { color: 'rgba(255,255,255,0.05)' } }
                }
            }
        });
    }
}

// Build table logs
function updateTable(logs) {
    const tbody = document.getElementById("historyTableBody");
    tbody.innerHTML = "";

    if (!logs || logs.length === 0) {
        tbody.innerHTML = `
            <tr>
                <td colspan="10" class="empty-state">No requests routed yet. Start querying the proxy at <code>http://localhost:8000/v1</code>.</td>
            </tr>
        `;
        return;
    }

    logs.forEach(log => {
        const row = document.createElement("tr");

        // Format Timestamp
        const date = new Date(log.timestamp);
        const timeStr = date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });

        // Cache Status Pill
        let cacheClass = "pill-miss";
        let cacheText = "MISS";
        if (log.cache_hit === "exact") {
            cacheClass = "pill-exact";
            cacheText = "EXACT HIT";
        } else if (log.cache_hit === "semantic") {
            cacheClass = "pill-semantic";
            cacheText = "SEMANTIC HIT";
        }
        const cachePill = `<span class="status-pill ${cacheClass}"><i class="fa-solid fa-circle"></i> ${cacheText}</span>`;

        // Complexity score pill
        const score = log.complexity_score;
        const complexityClass = log.tier_selected === 2 ? "complexity-high" : "complexity-low";
        const complexityLabel = log.tier_selected === 2 ? "High (T2)" : log.tier_selected === 1 ? "Low (T1)" : "Cache (T0)";
        const complexityPill = `<span class="complexity-pill ${complexityClass}">${complexityLabel} (${score.toFixed(1)})</span>`;

        // Pricing column
        const totalCost = log.cost;
        const costStr = totalCost === 0 ? "$0.0000" : `$${totalCost.toFixed(5)}`;

        // Success / Failure Icon
        let successIcon = "";
        if (log.success === 1) {
            successIcon = `<span style="color: var(--emerald-light); margin-right: 8px;" title="Run succeeded"><i class="fa-solid fa-circle-check"></i></span>`;
        } else if (log.success === 0) {
            successIcon = `<span style="color: var(--rose-light); margin-right: 8px;" title="Run failed or escalated"><i class="fa-solid fa-circle-xmark"></i></span>`;
        } else {
            successIcon = `<span style="color: var(--text-muted); margin-right: 8px;" title="No feedback"><i class="fa-solid fa-circle-minus"></i></span>`;
        }

        row.innerHTML = `
            <td>${timeStr}</td>
            <td><code>${log.requested_model}</code></td>
            <td><code>${log.routed_model}</code></td>
            <td>${log.provider || 'N/A'}</td>
            <td>${cachePill}</td>
            <td>${complexityPill}</td>
            <td>${log.tokens}</td>
            <td style="color: var(--emerald-light); font-weight: 500;">${costStr}</td>
            <td>${log.duration_ms} ms</td>
            <td style="max-width: 240px; overflow: hidden; text-overflow: ellipsis;" title="${log.routing_reason}">
                ${successIcon} ${log.routing_reason || '-'}
            </td>
        `;

        tbody.appendChild(row);
    });
}

// Clear cache command
async function clearCache() {
    if (!confirm("Are you sure you want to clear the prompt and semantic caches?")) {
        return;
    }

    try {
        const response = await fetch("/api/clear_cache", { method: "POST" });
        const result = await response.json();
        
        if (response.ok) {
            showToast("Agent Router cache successfully cleared!");
            fetchMetrics(false);
        } else {
            throw new Error(result.error || "Failed to clear cache");
        }
    } catch (err) {
        showToast(`Error clearing cache: ${err.message}`, false);
    }
}
