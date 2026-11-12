/**
 * AETHERIS • High-Performance Vanilla JS Frontend Engine
 * Handles Chart.js instances, real-time telemetry fetching, tab routing,
 * dynamic horizon toggling, and interactive neural weather simulation.
 */

let forecastChart = null;
let simChart = null;
let lossChart = null;
let residualChart = null;

let globalTelemetry = null;
let currentHorizon = 24;
let currentSpan = 168;

// Initialize on DOM load
document.addEventListener("DOMContentLoaded", () => {
    initTabRouting();
    initHorizonSelector();
    initSpanSelector();
    initSliderListeners();
    fetchDashboardTelemetry();
});

// ==============================================================================
// 1. DATA FETCHING & TELEMETRY
// ==============================================================================
async function fetchDashboardTelemetry() {
    try {
        const response = await fetch(`/api/dashboard-telemetry?span=${currentSpan}`);
        if (!response.ok) throw new Error("Telemetry API unavailable");
        globalTelemetry = await response.json();

        updateKPICards(globalTelemetry.kpis);
        renderForecastChart(globalTelemetry);
        renderDiagnosticsCharts(globalTelemetry);
        
        // Populate default simulator values from latest telemetry
        const kpis = globalTelemetry.kpis;
        document.getElementById("simTemp").value = kpis.current_pred_temp;
        document.getElementById("simHum").value = kpis.humidity;
        document.getElementById("simPres").value = kpis.pressure;
        document.getElementById("simWind").value = kpis.wind_speed;
        updateSliderLabels();

        // Run initial simulation preview
        executeSimulation();
    } catch (err) {
        console.error("Error fetching telemetry:", err);
    }
}

function updateKPICards(kpis) {
    document.getElementById("kpiCurrentTemp").textContent = kpis.current_pred_temp.toFixed(1);
    document.getElementById("kpiActualTemp").textContent = kpis.current_actual_temp.toFixed(1);
    
    const deltaSign = kpis.temp_24h_delta >= 0 ? "+" : "";
    document.getElementById("kpiDelta").textContent = `${deltaSign}${kpis.temp_24h_delta.toFixed(1)}`;
    document.getElementById("kpiDeltaBadge").textContent = kpis.temp_24h_delta >= 0 ? "▲ Warming Trend" : "▼ Cooling Trend";
    
    document.getElementById("kpiHumidity").textContent = Math.round(kpis.humidity);
    document.getElementById("kpiRmse").textContent = `±${kpis.rmse.toFixed(2)}`;
    document.getElementById("kpiR2").textContent = kpis.r2.toFixed(3);

    // Sub-bar telemetry
    document.getElementById("telTemp").textContent = `${kpis.current_actual_temp.toFixed(1)} °C`;
    document.getElementById("telHum").textContent = `${Math.round(kpis.humidity)} %`;
    document.getElementById("telPres").textContent = `${kpis.pressure.toFixed(1)} hPa`;
    document.getElementById("telWind").textContent = `${kpis.wind_speed.toFixed(1)} km/h`;
}

// ==============================================================================
// 2. MAIN FORECAST CHART (CHART.JS)
// ==============================================================================
function renderForecastChart(data) {
    const ctx = document.getElementById("forecastChart").getContext("2d");
    const labels = [...data.series.timestamps];
    const actualData = [...data.series.actual];
    const predData = [...data.series.predicted];

    // Append future horizon projection
    const futureTimes = data.future_forecast.hours.slice(0, currentHorizon);
    const futurePreds = data.future_forecast.predictions.slice(0, currentHorizon);
    const ciUpper = data.future_forecast.ci_upper.slice(0, currentHorizon);
    const ciLower = data.future_forecast.ci_lower.slice(0, currentHorizon);

    // Padding arrays for clean alignment
    const futureLabelList = futureTimes.map(h => `Future ${h}`);
    const extendedLabels = [...labels, ...futureLabelList];

    const extendedActual = [...actualData, ...Array(futureTimes.length).fill(null)];
    const extendedPred = [...predData, ...Array(futureTimes.length).fill(null)];

    const futureCurve = [
        ...Array(labels.length - 1).fill(null),
        predData[predData.length - 1], // continuous connection
        ...futurePreds
    ];

    const ciUpperCurve = [
        ...Array(labels.length - 1).fill(null),
        predData[predData.length - 1],
        ...ciUpper
    ];

    const ciLowerCurve = [
        ...Array(labels.length - 1).fill(null),
        predData[predData.length - 1],
        ...ciLower
    ];

    if (forecastChart) {
        forecastChart.destroy();
    }

    // Gradient fills
    const actualGradient = ctx.createLinearGradient(0, 0, 0, 400);
    actualGradient.addColorStop(0, "rgba(0, 242, 254, 0.22)");
    actualGradient.addColorStop(1, "rgba(0, 242, 254, 0.0)");

    const futureGradient = ctx.createLinearGradient(0, 0, 0, 400);
    futureGradient.addColorStop(0, "rgba(236, 72, 153, 0.25)");
    futureGradient.addColorStop(1, "rgba(236, 72, 153, 0.0)");

    forecastChart = new Chart(ctx, {
        type: "line",
        data: {
            labels: extendedLabels,
            datasets: [
                {
                    label: "Actual Weather (°C)",
                    data: extendedActual,
                    borderColor: "#00f2fe",
                    backgroundColor: actualGradient,
                    borderWidth: 2.2,
                    fill: true,
                    tension: 0.3,
                    pointRadius: 0,
                    pointHoverRadius: 6,
                    pointHoverBackgroundColor: "#00f2fe",
                    pointHoverBorderColor: "#ffffff",
                    pointHoverBorderWidth: 2
                },
                {
                    label: "LSTM Historical Projection",
                    data: extendedPred,
                    borderColor: "#fbbf24",
                    borderWidth: 2,
                    borderDash: [5, 4],
                    fill: false,
                    tension: 0.3,
                    pointRadius: 0,
                    pointHoverRadius: 5
                },
                {
                    label: `Future Horizon (+${currentHorizon}h)`,
                    data: futureCurve,
                    borderColor: "#ec4899",
                    backgroundColor: futureGradient,
                    borderWidth: 3,
                    fill: false,
                    tension: 0.35,
                    pointRadius: 3,
                    pointBackgroundColor: "#f43f5e",
                    pointBorderColor: "#ffffff",
                    pointBorderWidth: 1.5,
                    pointHoverRadius: 7
                },
                {
                    label: "95% Confidence Upper Band",
                    data: ciUpperCurve,
                    borderColor: "transparent",
                    backgroundColor: "rgba(236, 72, 153, 0.12)",
                    fill: "+1",
                    pointRadius: 0,
                    hoverRadius: 0
                },
                {
                    label: "95% Confidence Lower Band",
                    data: ciLowerCurve,
                    borderColor: "transparent",
                    backgroundColor: "transparent",
                    fill: false,
                    pointRadius: 0,
                    hoverRadius: 0
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: "index",
                intersect: false
            },
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: "rgba(10, 15, 25, 0.9)",
                    titleFont: { family: "Outfit", size: 13, weight: "bold" },
                    bodyFont: { family: "Space Grotesk", size: 12 },
                    borderColor: "rgba(255, 255, 255, 0.12)",
                    borderWidth: 1,
                    padding: 12,
                    boxPadding: 6,
                    usePointStyle: true,
                    callbacks: {
                        label: function (ctx) {
                            if (ctx.raw === null || ctx.dataset.label.includes("Band")) return null;
                            return `${ctx.dataset.label}: ${ctx.raw.toFixed(2)} °C`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    grid: { color: "rgba(255, 255, 255, 0.05)" },
                    ticks: {
                        color: "#94a3b8",
                        font: { family: "Inter", size: 11 },
                        maxTicksLimit: 12
                    }
                },
                y: {
                    grid: { color: "rgba(255, 255, 255, 0.05)" },
                    ticks: {
                        color: "#94a3b8",
                        font: { family: "Space Grotesk", size: 11 },
                        callback: val => `${val.toFixed(1)} °C`
                    }
                }
            }
        }
    });
}

// ==============================================================================
// 3. WEATHER SIMULATOR ENGINE
// ==============================================================================
async function executeSimulation() {
    const temp = parseFloat(document.getElementById("simTemp").value);
    const hum = parseFloat(document.getElementById("simHum").value);
    const pres = parseFloat(document.getElementById("simPres").value);
    const wind = parseFloat(document.getElementById("simWind").value);

    try {
        const response = await fetch("/api/simulate", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ temperature: temp, humidity: hum, pressure: pres, wind_speed: wind })
        });
        const result = await response.json();

        renderSimChart(result.forward_hours, result.predicted_trajectory, temp);
        document.getElementById("simDynamicInsight").textContent = result.trend_summary;
        document.getElementById("simInsightText").textContent = 
            `Simulation initialized at ${temp}°C, ${hum}% RH, ${pres} hPa. Net 48h thermal drift: ${result.delta_48h > 0 ? '+' : ''}${result.delta_48h} °C.`;
    } catch (err) {
        console.error("Simulation error:", err);
    }
}

function renderSimChart(hours, trajectory, baselineTemp) {
    const ctx = document.getElementById("simChart").getContext("2d");

    if (simChart) simChart.destroy();

    const simGradient = ctx.createLinearGradient(0, 0, 0, 300);
    simGradient.addColorStop(0, "rgba(0, 242, 254, 0.3)");
    simGradient.addColorStop(1, "rgba(0, 242, 254, 0.0)");

    simChart = new Chart(ctx, {
        type: "line",
        data: {
            labels: hours,
            datasets: [
                {
                    label: "Simulated Forecast (°C)",
                    data: trajectory,
                    borderColor: "#00f2fe",
                    backgroundColor: simGradient,
                    borderWidth: 3,
                    fill: true,
                    tension: 0.35,
                    pointRadius: 4,
                    pointBackgroundColor: "#22d3ee",
                    pointBorderColor: "#ffffff",
                    pointBorderWidth: 1.5,
                    pointHoverRadius: 7
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: "rgba(10, 15, 25, 0.9)",
                    titleFont: { family: "Outfit" },
                    bodyFont: { family: "Space Grotesk" },
                    borderColor: "rgba(0, 242, 254, 0.3)",
                    borderWidth: 1,
                    padding: 10,
                    callbacks: {
                        label: ctx => `Projected: ${ctx.raw.toFixed(2)} °C`
                    }
                }
            },
            scales: {
                x: {
                    grid: { color: "rgba(255, 255, 255, 0.05)" },
                    ticks: { color: "#94a3b8", font: { family: "Space Grotesk", size: 10 }, maxTicksLimit: 12 }
                },
                y: {
                    grid: { color: "rgba(255, 255, 255, 0.05)" },
                    ticks: { color: "#94a3b8", font: { family: "Space Grotesk", size: 10 }, callback: v => `${v.toFixed(1)} °C` }
                }
            }
        }
    });
}

function applyPreset(temp, hum, pres, wind) {
    document.getElementById("simTemp").value = temp;
    document.getElementById("simHum").value = hum;
    document.getElementById("simPres").value = pres;
    document.getElementById("simWind").value = wind;
    updateSliderLabels();
    executeSimulation();
}

function updateSliderLabels() {
    document.getElementById("simTempVal").textContent = `${parseFloat(document.getElementById("simTemp").value).toFixed(1)} °C`;
    document.getElementById("simHumVal").textContent = `${Math.round(document.getElementById("simHum").value)} %`;
    document.getElementById("simPresVal").textContent = `${parseFloat(document.getElementById("simPres").value).toFixed(1)} hPa`;
    document.getElementById("simWindVal").textContent = `${parseFloat(document.getElementById("simWind").value).toFixed(1)} km/h`;
}

function initSliderListeners() {
    ["simTemp", "simHum", "simPres", "simWind"].forEach(id => {
        const slider = document.getElementById(id);
        slider.addEventListener("input", () => {
            updateSliderLabels();
            executeSimulation();
        });
    });
}

// ==============================================================================
// 4. NEURAL DIAGNOSTICS CHARTS
// ==============================================================================
function renderDiagnosticsCharts(data) {
    const history = data.training_history;

    // Loss chart
    const lossCtx = document.getElementById("lossChart").getContext("2d");
    if (lossChart) lossChart.destroy();

    lossChart = new Chart(lossCtx, {
        type: "line",
        data: {
            labels: history.epochs.map(e => `Epoch ${e}`),
            datasets: [
                {
                    label: "Train Loss",
                    data: history.train_loss,
                    borderColor: "#38bdf8",
                    borderWidth: 2,
                    tension: 0.3,
                    pointRadius: 3
                },
                {
                    label: "Validation Loss",
                    data: history.val_loss,
                    borderColor: "#fbbf24",
                    borderWidth: 2,
                    tension: 0.3,
                    pointRadius: 3
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { labels: { color: "#94a3b8", font: { family: "Outfit" } } }
            },
            scales: {
                x: { grid: { color: "rgba(255, 255, 255, 0.05)" }, ticks: { color: "#94a3b8" } },
                y: { grid: { color: "rgba(255, 255, 255, 0.05)" }, ticks: { color: "#94a3b8" } }
            }
        }
    });

    // Residuals chart
    const residCtx = document.getElementById("residualChart").getContext("2d");
    if (residualChart) residualChart.destroy();

    // Bin residuals
    const residuals = data.residuals;
    const bins = 20;
    const minR = -4;
    const maxR = 4;
    const step = (maxR - minR) / bins;
    const binCounts = Array(bins).fill(0);
    const binLabels = [];

    for (let i = 0; i < bins; i++) {
        const low = minR + i * step;
        const high = low + step;
        binLabels.push(`${low.toFixed(1)}°`);
    }

    residuals.forEach(r => {
        if (r >= minR && r < maxR) {
            const idx = Math.floor((r - minR) / step);
            if (idx >= 0 && idx < bins) binCounts[idx]++;
        }
    });

    residualChart = new Chart(residCtx, {
        type: "bar",
        data: {
            labels: binLabels,
            datasets: [
                {
                    label: "Frequency",
                    data: binCounts,
                    backgroundColor: "rgba(0, 242, 254, 0.4)",
                    borderColor: "#00f2fe",
                    borderWidth: 1,
                    borderRadius: 4
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: { grid: { display: false }, ticks: { color: "#94a3b8", maxTicksLimit: 10 } },
                y: { grid: { color: "rgba(255, 255, 255, 0.05)" }, ticks: { color: "#94a3b8" } }
            }
        }
    });
}

// ==============================================================================
// 5. INTERACTIVE CONTROLS & ROUTING
// ==============================================================================
function initTabRouting() {
    const tabs = document.querySelectorAll(".tab-btn");
    tabs.forEach(tab => {
        tab.addEventListener("click", () => {
            tabs.forEach(t => t.classList.remove("active"));
            document.querySelectorAll(".tab-pane").forEach(p => p.classList.remove("active"));

            tab.classList.add("active");
            const targetPane = document.getElementById(tab.dataset.tab);
            if (targetPane) targetPane.classList.add("active");

            // Re-render chart layouts on tab switch
            if (tab.dataset.tab === "tabForecast" && forecastChart) forecastChart.resize();
            if (tab.dataset.tab === "tabSimulator" && simChart) simChart.resize();
            if (tab.dataset.tab === "tabDiagnostics") {
                if (lossChart) lossChart.resize();
                if (residualChart) residualChart.resize();
            }
        });
    });
}

function initHorizonSelector() {
    const btns = document.querySelectorAll(".horizon-btn");
    btns.forEach(btn => {
        btn.addEventListener("click", () => {
            btns.forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            currentHorizon = parseInt(btn.dataset.horizon, 10);
            if (globalTelemetry) renderForecastChart(globalTelemetry);
        });
    });
}

function initSpanSelector() {
    const btns = document.querySelectorAll(".span-btn");
    btns.forEach(btn => {
        btn.addEventListener("click", () => {
            btns.forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            currentSpan = parseInt(btn.dataset.span, 10);
            fetchDashboardTelemetry();
        });
    });
}
