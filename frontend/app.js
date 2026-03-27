const API_URL = "http://127.0.0.1:8000";

let priceChart, rsiChart, macdChart;

async function predict() {
  const stockInput = document.getElementById("stock").value;
  const stock = stockInput.trim() || "RELIANCE";

  const res = await fetch(`${API_URL}/predict?stock=${stock}`);
  const data = await res.json();

  console.log("DATA:", data);

  // ❗ Safety check
  if (!data || !data.history || !Array.isArray(data.history)) {
    alert("Invalid backend response");
    return;
  }

  const history = data.history;
  const prediction = data.prediction || [];

  const combined = [...history, ...prediction];

  // 🔥 DESTROY OLD CHARTS
  if (priceChart) priceChart.destroy();
  if (rsiChart) rsiChart.destroy();
  if (macdChart) macdChart.destroy();

  // =====================
  // PRICE CHART
  // =====================
  priceChart = new Chart(document.getElementById("priceChart"), {
    type: "line",
    data: {
      labels: combined.map((_, i) => i),
      datasets: [{
        label: "Stock Price",
        data: combined,
        borderWidth: 2
      }]
    }
  });

  // =====================
  // RSI
  // =====================
  if (data.rsi && Array.isArray(data.rsi)) {
    rsiChart = new Chart(document.getElementById("rsiChart"), {
      type: "line",
      data: {
        labels: data.rsi.map((_, i) => i),
        datasets: [{
          label: "RSI",
          data: data.rsi
        }]
      }
    });
  }

  // =====================
  // MACD
  // =====================
  if (data.macd && data.signal) {
    macdChart = new Chart(document.getElementById("macdChart"), {
      type: "line",
      data: {
        labels: data.macd.map((_, i) => i),
        datasets: [
          { label: "MACD", data: data.macd },
          { label: "Signal", data: data.signal }
        ]
      }
    });
  }
}