const boardEl = document.getElementById("board");
const statusEl = document.getElementById("status-text");
const newGameBtn = document.getElementById("new-game");
const modelSelect = document.getElementById("model-select");
const ruleListEl = document.getElementById("rule-list");
const forbiddenListEl = document.getElementById("forbidden-list");
const swapStateEl = document.getElementById("swap-state");

let state = null;
let aiRequestInFlight = false;
let aiRequestId = 0;

async function api(path, opts = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...opts,
  });
  if (!res.ok) {
    const text = await res.text();
    let detail = text || res.statusText;
    try {
      detail = JSON.parse(text).detail || detail;
    } catch (_) {
      // JSON 오류가 아니면 원문을 그대로 보여준다.
    }
    throw new Error(detail);
  }
  return res.json();
}

function formatPoint(x, y) {
  return `(${x + 1}, ${y + 1})`;
}

function forbiddenKey(x, y) {
  return `${x},${y}`;
}

function forbiddenMap() {
  const map = new Map();
  (state?.forbidden_moves || []).forEach((move) => {
    map.set(forbiddenKey(move.x, move.y), move);
  });
  return map;
}

function renderBoard() {
  if (!state) return;
  const forbidden = forbiddenMap();
  boardEl.innerHTML = "";
  boardEl.classList.toggle("thinking", aiRequestInFlight);
  boardEl.style.gridTemplateColumns = `repeat(${state.board[0].length}, 1fr)`;
  state.board.flat().forEach((cell, idx) => {
    const div = document.createElement("div");
    div.className = "cell";
    const x = idx % state.board.length;
    const y = Math.floor(idx / state.board.length);
    div.dataset.x = x;
    div.dataset.y = y;
    const forbiddenInfo = forbidden.get(forbiddenKey(x, y));
    if (cell === 1 || cell === 2) {
      const stone = document.createElement("div");
      stone.className = `stone ${cell === 1 ? "black" : "white"}`;
      div.appendChild(stone);
    } else if (forbiddenInfo) {
      div.classList.add("forbidden");
      div.title = `${formatPoint(x, y)} ${forbiddenInfo.label}`;
      const mark = document.createElement("span");
      mark.className = "forbidden-mark";
      mark.textContent = "×";
      div.appendChild(mark);
    }
    div.addEventListener("click", () => onCellClick(x, y));
    boardEl.appendChild(div);
  });
  renderRulePanels();
  updateStatus();
}

function renderRulePanels() {
  renderSwapState();
  renderForbiddenList();
  renderRuleList();
}

function renderSwapState() {
  const swap = state?.rules?.swap;
  if (!swap) {
    swapStateEl.textContent = "스왑 상태를 불러오는 중입니다.";
    return;
  }
  if (!swap.enabled) {
    swapStateEl.textContent = "스왑 룰 꺼짐";
  } else if (swap.available) {
    swapStateEl.textContent = "스왑 가능: 백이 색을 바꿀 수 있습니다.";
  } else if (swap.used) {
    swapStateEl.textContent = "스왑 사용됨";
  } else {
    swapStateEl.textContent = "스왑 대기 아님";
  }
}

function renderForbiddenList() {
  const moves = state?.forbidden_moves || [];
  forbiddenListEl.innerHTML = "";
  if (state?.current_player !== 1 || state?.winner !== null) {
    forbiddenListEl.textContent = "흑 차례에만 금수점을 표시합니다.";
    return;
  }
  if (moves.length === 0) {
    forbiddenListEl.textContent = "현재 금수점 없음";
    return;
  }

  const counts = moves.reduce((acc, move) => {
    acc[move.label] = (acc[move.label] || 0) + 1;
    return acc;
  }, {});
  const summary = document.createElement("p");
  summary.className = "forbidden-summary";
  summary.textContent = Object.entries(counts)
    .map(([label, count]) => `${label} ${count}`)
    .join(" · ");
  forbiddenListEl.appendChild(summary);

  const list = document.createElement("div");
  list.className = "forbidden-pills";
  moves.slice(0, 8).forEach((move) => {
    const item = document.createElement("span");
    item.className = "forbidden-pill";
    item.textContent = `${formatPoint(move.x, move.y)} ${move.label}`;
    list.appendChild(item);
  });
  if (moves.length > 8) {
    const extra = document.createElement("span");
    extra.className = "forbidden-pill muted";
    extra.textContent = `+${moves.length - 8}`;
    list.appendChild(extra);
  }
  forbiddenListEl.appendChild(list);
}

function renderRuleList() {
  const items = state?.rules?.items || [];
  ruleListEl.innerHTML = "";
  items.forEach((rule) => {
    const div = document.createElement("div");
    div.className = "rule-item";
    const label = document.createElement("strong");
    label.textContent = rule.label;
    const desc = document.createElement("span");
    desc.textContent = rule.description;
    div.append(label, desc);
    ruleListEl.appendChild(div);
  });
}

function updateStatus(msg) {
  if (msg) {
    statusEl.textContent = msg;
    return;
  }
  if (!state) {
    statusEl.textContent = "불러오는 중...";
    return;
  }
  if (state.winner === 1) statusEl.textContent = "흑 승리";
  else if (state.winner === 2) statusEl.textContent = "백 승리";
  else if (state.winner === 0) statusEl.textContent = "무승부";
  else if (aiRequestInFlight) statusEl.textContent = "AI 생각 중...";
  else if (state.current_player === 1) statusEl.textContent = "흑 차례";
  else statusEl.textContent = "백 차례";
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
  if (!state || aiRequestInFlight || state.winner !== null || state.current_player !== 1) return;
  const forbiddenInfo = forbiddenMap().get(forbiddenKey(x, y));
  if (forbiddenInfo) {
    updateStatus(`${formatPoint(x, y)} ${forbiddenInfo.label}: 둘 수 없는 자리입니다.`);
    return;
  }
  try {
    state = await api("/api/game/move", {
      method: "POST",
      body: JSON.stringify({ x, y }),
    });
    renderBoard();
    if (state.winner === null && state.current_player === 2) {
      await requestAiMove();
    }
  } catch (err) {
    updateStatus(err.message);
  }
}

async function requestAiMove() {
  const requestId = ++aiRequestId;
  aiRequestInFlight = true;
  renderBoard();
  try {
    const nextState = await api("/api/game/ai-move", { method: "POST" });
    if (requestId === aiRequestId) {
      state = nextState;
    }
  } catch (err) {
    updateStatus(err.message);
  } finally {
    if (requestId === aiRequestId) {
      aiRequestInFlight = false;
      renderBoard();
    }
  }
}

async function newGame() {
  aiRequestId += 1;
  aiRequestInFlight = false;
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
  updateStatus("초기 상태를 불러오지 못했습니다.");
});
