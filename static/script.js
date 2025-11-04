const analyzeBtn = document.getElementById("analyzeBtn");
const loadBtn = document.getElementById("loadBtn");
const textInput = document.getElementById("textInput");
const result = document.getElementById("result");
const tweetSelectContainer = document.getElementById("tweetSelectContainer");
const tweetSelect = document.getElementById("tweetSelect");
const confirmTweetBtn = document.getElementById("confirmTweetBtn");

// === Analyser le texte ===
analyzeBtn.addEventListener("click", async () => {
  const text = textInput.value.trim();
  if (!text) {
    result.innerHTML = "⚠️ Entrez ou sélectionnez un texte.";
    return;
  }
  result.innerHTML = "⏳ Analyse en cours...";
  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    const data = await res.json();
    if (data.error) {
      result.innerHTML = "❌ " + data.error;
    } else {
      const color = data.sentiment_predicted === "POSITIVE" ? "green" : "red";
      result.innerHTML = `<span style="color:${color}">${data.sentiment_predicted}</span> (confiance : ${data.confidence})`;
    }
  } catch {
    result.innerHTML = "❌ Erreur réseau.";
  }
});

// === Charger les tweets ===
loadBtn.addEventListener("click", async () => {
  result.innerHTML = "🔍 Chargement des tweets...";
  tweetSelectContainer.style.display = "none";
  try {
    const res = await fetch("/tweets");
    const data = await res.json();

    if (data.error) {
      result.innerHTML = "⚠️ " + data.error;
      return;
    }

    tweetSelect.innerHTML = '<option value="">Sélectionnez un tweet...</option>';
    data.tweets.forEach((tweet) => {
      const option = document.createElement("option");
      option.value = tweet.text;
      option.textContent = tweet.text.substring(0, 80) + "...";
      tweetSelect.appendChild(option);
    });

    tweetSelectContainer.style.display = "block";
    result.innerHTML = "✅ Tweets chargés. Sélectionnez-en un.";
  } catch {
    result.innerHTML = "❌ Erreur lors du chargement des tweets.";
  }
});

// === Valider le tweet sélectionné ===
confirmTweetBtn.addEventListener("click", () => {
  const selectedText = tweetSelect.value;
  if (selectedText) {
    textInput.value = selectedText;
    result.innerHTML = "💬 Tweet importé dans le champ de texte.";
    tweetSelectContainer.style.display = "none";
  } else {
    result.innerHTML = "⚠️ Sélectionnez un tweet avant de valider.";
  }
});
