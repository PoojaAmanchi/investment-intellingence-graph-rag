const API_BASE = "http://localhost:8000";

const input = document.getElementById("question-input");
const askButton = document.getElementById("ask-button");
const progressEl = document.getElementById("progress");
const answerContainer = document.getElementById("answer-container");
const answerText = document.getElementById("answer-text");
const citationsEl = document.getElementById("citations");
const questionTypeBadge = document.getElementById("question-type-badge");
const confidenceBadge = document.getElementById("confidence-badge");

const NODE_LABELS = {
  router: "Classifying question...",
  retriever: "Retrieving from graph + vector store...",
  reranker: "Reranking results...",
  synthesizer: "Generating answer...",
  verifier: "Verifying groundedness...",
};

async function askQuestion() {
  const question = input.value.trim();
  if (!question) return;

  askButton.disabled = true;
  answerContainer.style.display = "none";
  progressEl.textContent = "";

  // Stream progress via SSE first, so the user sees what stage the pipeline is at
  const streamUrl = `${API_BASE}/stream/query?question=${encodeURIComponent(question)}`;
  const eventSource = new EventSource(streamUrl);

  eventSource.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      if (data.node && NODE_LABELS[data.node]) {
        progressEl.textContent = NODE_LABELS[data.node];
      }
    } catch (e) {
      // ignore malformed progress events, the final POST /query result is authoritative
    }
  };

  eventSource.addEventListener("done", () => {
    eventSource.close();
  });

  eventSource.onerror = () => {
    eventSource.close();
  };

  // The actual answer comes from the standard POST endpoint — the SSE
  // stream above is progress-only, matching the FOMC project's pattern
  // of separating live progress from the final authoritative response.
  try {
    const response = await fetch(`${API_BASE}/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    const result = await response.json();
    renderAnswer(result);
  } catch (err) {
    progressEl.textContent = "Error: could not reach the API. Is the backend running?";
  } finally {
    askButton.disabled = false;
  }
}

function renderAnswer(result) {
  progressEl.textContent = "";
  answerContainer.style.display = "block";

  answerText.textContent = result.answer;
  questionTypeBadge.textContent = result.question_type;

  const confidencePct = Math.round((result.confidence || 0) * 100);
  confidenceBadge.textContent = `${confidencePct}% confidence`;
  confidenceBadge.className = "badge " + (confidencePct >= 70 ? "high-confidence" : "low-confidence");

  if (result.citations && result.citations.length > 0) {
    citationsEl.textContent = `Sources: ${result.citations.join(", ")}`;
  } else {
    citationsEl.textContent = "";
  }
}

askButton.addEventListener("click", askQuestion);
input.addEventListener("keypress", (e) => {
  if (e.key === "Enter") askQuestion();
});
