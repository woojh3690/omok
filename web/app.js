const boardEl = document.getElementById("board");
const statusEl = document.getElementById("status-text");
const newGameBtn = document.getElementById("new-game");
const modelSelect = document.getElementById("model-select");

let state = null;

async function api(path, opts = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...opts,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || res.statusText);
  }
  return res.json();
}

function renderBoard() {
  if (!state) return;
  boardEl.innerHTML = "";
  boardEl.style.gridTemplateColumns = `repeat(${state.board[0].length}, 1fr)`;
  state.board.flat().forEach((cell, idx) => {
    const div = document.createElement("div");
    div.className = "cell";
    const x = idx % state.board.length;
    const y = Math.floor(idx / state.board.length);
    div.dataset.x = x;
    div.dataset.y = y;
    if (cell === 1 || cell === 2) {
      const stone = document.createElement("div");
      stone.className = `stone ${cell === 1 ? "black" : "white"}`;
      div.appendChild(stone);
    }
    div.addEventListener("click", () => onCellClick(x, y));
    boardEl.appendChild(div);
  });
  updateStatus();
}

function updateStatus(msg) {
  if (msg) {
    statusEl.textContent = msg;
    return;
  }
  if (!state) {
    statusEl.textContent = "Loading...";
    return;
  }
  if (state.winner === 1) statusEl.textContent = "You win!";
  else if (state.winner === 2) statusEl.textContent = "AI wins.";
  else if (state.winner === 0) statusEl.textContent = "Draw.";
  else statusEl.textContent = state.current_player === 1 ? "Your move (black)" : "AI thinking...";
}

async function fetchState() {
  state = await api("/api/state");
  renderBoard();
}

async function fetchModels() {
  const data = await api("/api/models");
  modelSelect.innerHTML = "";
  data.models.forEach((m) => {
    const opt = document.createElement("option");
    opt.value = m.id;
    const lossText = Number.isFinite(m.loss) ? m.loss.toFixed(3) : "n/a";
    opt.textContent = `${m.id} (loss ${lossText})`;
    if (m.id === data.active_id) opt.selected = true;
    modelSelect.appendChild(opt);
  });
  modelSelect.disabled = true;
}

async function onCellClick(x, y) {
  if (!state || state.winner !== null || state.current_player !== 1) return;
  try {
    state = await api("/api/game/move", {
      method: "POST",
      body: JSON.stringify({ x, y }),
    });
    renderBoard();
  } catch (err) {
    updateStatus(err.message);
  }
}

async function newGame() {
  state = await api("/api/game/new", { method: "POST" });
  renderBoard();
}

modelSelect.addEventListener("change", async (e) => {
  const model_id = e.target.value;
  await api("/api/models/select", {
    method: "POST",
    body: JSON.stringify({ model_id }),
  });
  updateStatus(`Switched to ${model_id}`);
});

newGameBtn.addEventListener("click", () => newGame());

async function init() {
  await fetchModels().catch(() => {});
  await fetchState();
}

init().catch((err) => {
  console.error(err);
  updateStatus("Failed to load initial state.");
});
