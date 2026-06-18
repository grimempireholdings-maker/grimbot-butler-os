"use strict";

const SIGNAL_NAMES = [
  "attention", "urgency", "novelty", "confidence",
  "reward", "friction", "fatigue", "curiosity",
];

const byId = (id) => document.getElementById(id);
const escapeHtml = (value) => String(value ?? "")
  .replaceAll("&", "&amp;")
  .replaceAll("<", "&lt;")
  .replaceAll(">", "&gt;")
  .replaceAll('"', "&quot;")
  .replaceAll("'", "&#039;");

const empty = (message) => `<div class="empty-state">${escapeHtml(message)}</div>`;
const errorMarkup = (message) => `<div class="empty-state error-state">${escapeHtml(message)}</div>`;
const formatJson = (value) => JSON.stringify(value, null, 2);

function errorMessage(payload, fallback) {
  if (typeof payload?.detail === "string") return payload.detail;
  if (Array.isArray(payload?.detail)) {
    return payload.detail.map((item) => item.msg || JSON.stringify(item)).join("; ");
  }
  return fallback;
}

async function api(path, options = {}) {
  const request = {
    method: options.method || "GET",
    headers: { "Content-Type": "application/json" },
  };
  if (options.body !== undefined) request.body = JSON.stringify(options.body);
  const response = await fetch(path, request);
  const contentType = response.headers.get("content-type") || "";
  const payload = contentType.includes("application/json") ? await response.json() : await response.text();
  if (!response.ok) throw new Error(errorMessage(payload, `${response.status} ${response.statusText}`));
  return payload;
}

function showToast(message, isError = false) {
  const toast = byId("toast");
  toast.textContent = message;
  toast.classList.toggle("error", isError);
  toast.hidden = false;
  window.clearTimeout(showToast.timeout);
  showToast.timeout = window.setTimeout(() => { toast.hidden = true; }, 4200);
}

async function withButton(button, task) {
  button.disabled = true;
  try {
    return await task();
  } catch (error) {
    showToast(error.message, true);
    throw error;
  } finally {
    button.disabled = false;
  }
}

function renderJson(target, payload) {
  target.innerHTML = `<pre>${escapeHtml(formatJson(payload))}</pre>`;
}

async function loadHealth() {
  const dot = byId("health-dot");
  const label = byId("health-label");
  try {
    await api("/health");
    dot.className = "status-dot online";
    label.textContent = "Brain online";
  } catch {
    dot.className = "status-dot offline";
    label.textContent = "Brain offline";
  }
}

function addChatMessage(kind, text, machineOutput = null) {
  const log = byId("chat-log");
  if (log.querySelector(".empty-state")) log.innerHTML = "";
  const machine = machineOutput
    ? `<details><summary>Machine output</summary><pre>${escapeHtml(formatJson(machineOutput))}</pre></details>`
    : "";
  log.insertAdjacentHTML("beforeend", `<div class="message ${kind}">
    <span class="speaker">${kind === "user" ? "Julian" : "Maya"}</span>
    <div>${escapeHtml(text)}</div>${machine}
  </div>`);
  log.scrollTop = log.scrollHeight;
}

function submitButtonFor(event) {
  return event.submitter || byId("chat-form").querySelector('button[type="submit"]');
}

async function sendChat(event) {
  event.preventDefault();
  const input = byId("chat-input");
  const text = input.value.trim();
  if (!text) return;
  addChatMessage("user", text);
  input.value = "";
  try {
    await withButton(submitButtonFor(event), async () => {
      const result = await api("/voice/conversation", {
        method: "POST",
        body: {
          push_to_talk: true,
          mock_transcript: text,
          assistant_mode: byId("chat-mode").value,
          verified: false,
        },
      });
      const responseText = result.agent_response?.user_response
        || result.speech_output?.text
        || result.maya_response?.user_response
        || "No response text returned.";
      const machineOutput = result.agent_response?.machine_output
        || result.machine_output
        || result.maya_response?.machine_output;
      addChatMessage("maya", responseText, machineOutput);
    });
  } catch (error) {
    addChatMessage("maya", `Request failed: ${error.message}`);
  }
}

function briefingBlock(title, values) {
  const list = Array.isArray(values) ? values : [values];
  const content = list.length
    ? `<ul>${list.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>`
    : `<span class="empty-state">None</span>`;
  return `<div class="briefing-section"><strong>${escapeHtml(title)}</strong>${content}</div>`;
}

async function generateBriefing(button) {
  await withButton(button, async () => {
    const result = await api("/maya/briefing", {
      method: "POST",
      body: { verified: false, mode: "maya_chief_of_staff" },
    });
    byId("briefing-output").innerHTML = [
      briefingBlock("Priority items", result.priority_items),
      briefingBlock("FYI", result.fyi),
      briefingBlock("Wins", result.wins),
      briefingBlock("Hazards", result.hazards),
      briefingBlock("Active projects", result.active_projects),
      briefingBlock("Current bottlenecks", result.current_bottlenecks),
      briefingBlock("Next actions", result.next_actions),
      briefingBlock("Next best action", result.next_best_action),
    ].join("");
  });
}

function contextEntryItem(item) {
  return `<div class="list-item">
    <strong class="item-title">${escapeHtml(item.name)}</strong>
    <div>${escapeHtml(item.content)}</div>
    <div class="item-meta">
      <span>priority: ${escapeHtml(item.priority)}</span>
      <span>${escapeHtml(item.source)}</span>
      <span class="tag ${item.verified ? "approved" : "pending"}">${item.verified ? "verified" : "unverified"}</span>
    </div>
  </div>`;
}

function contextProjectItem(project, field = null) {
  const body = field ? project[field] : `${project.status} - priority ${project.priority}`;
  return `<div class="list-item">
    <strong class="item-title">${escapeHtml(project.name)}</strong>
    <div>${escapeHtml(body)}</div>
  </div>`;
}

async function loadContext() {
  const targets = {
    priorities: byId("context-priorities"),
    projects: byId("context-projects"),
    bottlenecks: byId("context-bottlenecks"),
    actions: byId("context-actions"),
  };
  try {
    const context = await api("/context");
    targets.priorities.innerHTML = context.priorities.length
      ? context.priorities.slice(0, 5).map(contextEntryItem).join("")
      : empty("No priorities recorded.");
    targets.projects.innerHTML = context.projects.length
      ? context.projects.slice(0, 6).map((item) => contextProjectItem(item)).join("")
      : empty("No active projects.");
    targets.bottlenecks.innerHTML = context.projects.length
      ? context.projects.slice(0, 5).map((item) => contextProjectItem(item, "current_bottleneck")).join("")
      : empty("No bottlenecks recorded.");
    targets.actions.innerHTML = context.projects.length
      ? context.projects.slice(0, 5).map((item) => contextProjectItem(item, "next_action")).join("")
      : empty("No next actions recorded.");
  } catch (error) {
    Object.values(targets).forEach((target) => { target.innerHTML = errorMarkup(error.message); });
  }
}

async function searchContext(event) {
  event.preventDefault();
  const query = byId("context-search").value.trim();
  if (!query) return;
  try {
    await withButton(event.submitter, async () => {
      const result = await api("/context/search", {
        method: "POST",
        body: { query, limit: 10 },
      });
      const projectResults = result.projects
        .slice(0, 3)
        .map((item) => contextProjectItem(item, "current_bottleneck"));
      const entryResults = result.entries
        .slice(0, 3)
        .map(contextEntryItem);
      const matches = [...projectResults, ...entryResults];
      if (matches.length) {
        matches.push(`<div class="list-item"><strong class="item-title">Next best action</strong><div>${escapeHtml(result.next_best_action)}</div></div>`);
        byId("context-search-result").innerHTML = matches.join("");
      } else {
        byId("context-search-result").innerHTML = empty(
          result.clarification_question || "No matching context."
        );
      }
    });
  } catch (error) {
    byId("context-search-result").innerHTML = errorMarkup(error.message);
  }
}

async function rememberContext(event) {
  event.preventDefault();
  const form = event.currentTarget;
  const name = byId("context-name").value.trim();
  const content = byId("context-remember").value.trim();
  if (!name || !content) return;
  try {
    await withButton(event.submitter, async () => {
      const result = await api("/context/remember", {
        method: "POST",
        body: {
          context_type: byId("context-type").value,
          name,
          content,
          priority: 70,
          source: "julian_prime",
          verified: byId("context-verified").checked,
        },
      });
      showToast(`Remembered: ${result.name}`);
      form.reset();
      await loadContext();
    });
  } catch (error) {
    showToast(error.message, true);
  }
}

function workspaceFact(label, value) {
  return `<div class="workspace-fact"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value || "Unavailable")}</strong></div>`;
}

async function loadWorkspace() {
  const summaryTarget = byId("workspace-summary");
  const commitsTarget = byId("workspace-commits");
  const warningsTarget = byId("workspace-warnings");
  try {
    const workspace = await api("/workspace");
    summaryTarget.innerHTML = [
      workspaceFact("Repository", workspace.repo_name),
      workspaceFact("Branch", workspace.branch || "Not in Git"),
      workspaceFact("Version", workspace.version || "Not detected"),
      workspaceFact("Working tree", workspace.status_summary.length ? `${workspace.status_summary.length} change(s)` : "Clean"),
    ].join("");
    commitsTarget.innerHTML = workspace.recent_commits.length
      ? workspace.recent_commits.map((commit) => `<div class="list-item">${escapeHtml(commit)}</div>`).join("")
      : empty("No recent commits available.");
    warningsTarget.innerHTML = workspace.warnings.length
      ? workspace.warnings.map((warning) => `<div class="list-item error-state">${escapeHtml(warning)}</div>`).join("")
      : empty("No workspace warnings.");
  } catch (error) {
    summaryTarget.innerHTML = errorMarkup(error.message);
    commitsTarget.innerHTML = errorMarkup(error.message);
    warningsTarget.innerHTML = errorMarkup(error.message);
  }
}

async function searchWorkspace(event) {
  event.preventDefault();
  const query = byId("workspace-search").value.trim();
  if (!query) return;
  const target = byId("workspace-search-result");
  try {
    await withButton(event.submitter, async () => {
      const result = await api("/workspace/search", {
        method: "POST",
        body: { query, max_results: 20 },
      });
      target.innerHTML = result.results.length
        ? result.results.map((match) => `<div class="list-item">
            <strong class="item-title">${escapeHtml(match.relative_path)}:${escapeHtml(match.line_number)}</strong>
            <div>${escapeHtml(match.snippet)}</div>
          </div>`).join("")
        : empty("No safe workspace matches.");
    });
  } catch (error) {
    target.innerHTML = errorMarkup(error.message);
  }
}

async function loadState() {
  const target = byId("state-output");
  try {
    const state = await api("/state");
    target.innerHTML = SIGNAL_NAMES.map((name) => {
      const value = Math.max(0, Math.min(1, Number(state.values?.[name] || 0)));
      return `<div class="signal">
        <span class="signal-name">${escapeHtml(name)}</span>
        <span class="signal-value">${value.toFixed(2)}</span>
        <div class="signal-track"><div class="signal-fill" style="width:${value * 100}%"></div></div>
      </div>`;
    }).join("");
  } catch (error) {
    target.innerHTML = errorMarkup(error.message);
  }
}

async function loadSkills() {
  const select = byId("skill-select");
  const output = byId("skills-output");
  try {
    const skills = await api("/skills");
    select.innerHTML = skills.length
      ? skills.map((skill) => `<option value="${escapeHtml(skill.name)}">${escapeHtml(skill.name)} - ${escapeHtml(skill.required_permission)}</option>`).join("")
      : `<option value="">No skills available</option>`;
    output.innerHTML = skills.length
      ? skills.map((skill) => `<div class="list-item">
          <strong class="item-title">${escapeHtml(skill.name)}</strong>
          <div>${escapeHtml(skill.description)}</div>
          <div class="item-meta"><span>${escapeHtml(skill.category)}</span><span>permission: ${escapeHtml(skill.required_permission)}</span></div>
        </div>`).join("")
      : empty("No skills available.");
  } catch (error) {
    select.innerHTML = `<option value="">Skills unavailable</option>`;
    output.innerHTML = errorMarkup(error.message);
  }
}

async function runSkill(event) {
  event.preventDefault();
  const name = byId("skill-select").value;
  if (!name) return;
  let inputs;
  try {
    inputs = JSON.parse(byId("skill-inputs").value || "{}");
  } catch {
    showToast("Skill inputs must be valid JSON.", true);
    return;
  }
  try {
    await withButton(event.submitter, async () => {
      const result = await api(`/skills/${encodeURIComponent(name)}/run`, {
        method: "POST",
        body: {
          inputs,
          permission: byId("skill-permission").value,
          assistant_mode: "maya_chief_of_staff",
          verified: false,
          include_state: true,
        },
      });
      renderJson(byId("skills-output"), result);
    });
  } catch (error) {
    byId("skills-output").innerHTML = errorMarkup(error.message);
  }
}

function factItem(fact, status = null) {
  const labels = [status, fact.tier].filter(Boolean);
  return `<div class="list-item">
    <strong class="item-title">${escapeHtml(fact.content)}</strong>
    <div class="item-meta">
      ${labels.map((label) => `<span class="tag ${escapeHtml(label)}">${escapeHtml(label)}</span>`).join("")}
      <span>confidence: ${Number(fact.confidence || 0).toFixed(2)}</span>
    </div>
  </div>`;
}

async function loadDream() {
  const factsTarget = byId("dream-facts");
  const promotionsTarget = byId("dream-promotions");
  try {
    const [facts, promotions] = await Promise.all([api("/dream/facts"), api("/dream/promotions")]);
    factsTarget.innerHTML = facts.length
      ? facts.map((fact) => factItem(fact, "approved")).join("")
      : empty("No approved semantic facts.");
    promotionsTarget.innerHTML = promotions.length
      ? promotions.map((item) => `<div class="list-item">
          <strong class="item-title">${escapeHtml(item.fact?.content || `Fact ${item.fact_id}`)}</strong>
          <div class="item-meta"><span class="tag ${escapeHtml(item.status)}">${escapeHtml(item.status)}</span></div>
          ${item.status === "pending" ? `<div class="item-actions">
            <button class="review-button" type="button" data-dream-action="approve" data-id="${item.id}">Approve</button>
            <button class="review-button reject" type="button" data-dream-action="reject" data-id="${item.id}">Reject</button>
          </div>` : ""}
        </div>`).join("")
      : empty("No pending fact promotions.");
  } catch (error) {
    factsTarget.innerHTML = errorMarkup(error.message);
    promotionsTarget.innerHTML = errorMarkup(error.message);
  }
}

async function reviewDream(button) {
  const action = button.dataset.dreamAction;
  await withButton(button, async () => {
    await api(`/dream/promotions/${encodeURIComponent(button.dataset.id)}/${action}`, {
      method: "POST",
      body: { note: "Reviewed in Maya Console", anchor: false },
    });
    showToast(`Dream fact ${action}d.`);
    await loadDream();
  });
}

async function runDream(button) {
  await withButton(button, async () => {
    const result = await api("/dream/run", {
      method: "POST",
      body: { provider: "rule_based", episode_limit: 500, run_forgetting: true },
    });
    const notice = byId("dream-result");
    notice.hidden = false;
    notice.textContent = `Dream ${result.cycle?.status || "completed"}: ${result.promotions_created || 0} promotion(s) created.`;
    await loadDream();
  });
}

function procedureItem(procedure, pending = false) {
  const name = procedure.name || procedure.proposal?.name || `Proposal ${procedure.pending_id}`;
  const status = procedure.status || "pending";
  const source = procedure.source || procedure.proposal?.source;
  const id = procedure.pending_id || procedure.id;
  return `<div class="list-item">
    <strong class="item-title">${escapeHtml(name)}</strong>
    <div class="item-meta">
      <span class="tag ${escapeHtml(status)}">${escapeHtml(status)}</span>
      ${source ? `<span>${escapeHtml(source)}</span>` : ""}
      ${procedure.version ? `<span>v${escapeHtml(procedure.version)}</span>` : ""}
    </div>
    ${pending && status === "pending" ? `<div class="item-actions">
      <button class="review-button" type="button" data-procedure-action="approve" data-id="${id}">Approve</button>
      <button class="review-button reject" type="button" data-procedure-action="reject" data-id="${id}">Reject</button>
    </div>` : ""}
  </div>`;
}

async function loadProcedures() {
  const listTarget = byId("procedures-list");
  const pendingTarget = byId("procedures-pending");
  try {
    const [procedures, pending] = await Promise.all([api("/procedures"), api("/procedures/pending")]);
    listTarget.innerHTML = procedures.length
      ? procedures.map((item) => procedureItem(item)).join("")
      : empty("No active procedures.");
    pendingTarget.innerHTML = pending.length
      ? pending.map((item) => procedureItem(item, true)).join("")
      : empty("No pending procedure proposals.");
  } catch (error) {
    listTarget.innerHTML = errorMarkup(error.message);
    pendingTarget.innerHTML = errorMarkup(error.message);
  }
}

async function reviewProcedure(button) {
  const action = button.dataset.procedureAction;
  await withButton(button, async () => {
    await api(`/procedures/pending/${encodeURIComponent(button.dataset.id)}/${action}`, {
      method: "POST",
      body: { note: "Reviewed in Maya Console" },
    });
    showToast(`Procedure proposal ${action}d.`);
    await loadProcedures();
  });
}

async function matchProcedure(event) {
  event.preventDefault();
  const query = byId("procedure-query").value.trim();
  if (!query) return;
  try {
    await withButton(event.submitter, async () => {
      const result = await api("/procedures/match", {
        method: "POST",
        body: { query, minimum_confidence: 0.65 },
      });
      renderJson(byId("procedure-match-result"), result);
    });
  } catch (error) {
    byId("procedure-match-result").innerHTML = errorMarkup(error.message);
  }
}

function memoryItem(item) {
  return `<div class="list-item">
    <strong class="item-title">${escapeHtml(item.name)}</strong>
    <div class="item-meta">
      ${item.room_name ? `<span>${escapeHtml(item.room_name)}</span>` : ""}
      ${item.zone_name ? `<span>${escapeHtml(item.zone_name)}</span>` : ""}
      ${item.count ? `<span>seen: ${escapeHtml(item.count)}</span>` : ""}
      ${item.confidence !== undefined ? `<span>confidence: ${Number(item.confidence).toFixed(2)}</span>` : ""}
    </div>
  </div>`;
}

async function loadMemory() {
  const roomsTarget = byId("memory-rooms");
  const hazardsTarget = byId("memory-hazards");
  const messTarget = byId("memory-mess");
  try {
    const [rooms, hazards, mess] = await Promise.all([
      api("/memory/rooms"), api("/memory/hazards"), api("/memory/mess-zones"),
    ]);
    roomsTarget.innerHTML = rooms.length ? rooms.map(memoryItem).join("") : empty("No known rooms.");
    hazardsTarget.innerHTML = hazards.length ? hazards.map(memoryItem).join("") : empty("No known hazards.");
    messTarget.innerHTML = mess.length ? mess.map(memoryItem).join("") : empty("No known mess zones.");
  } catch (error) {
    roomsTarget.innerHTML = errorMarkup(error.message);
    hazardsTarget.innerHTML = errorMarkup(error.message);
    messTarget.innerHTML = errorMarkup(error.message);
  }
}

async function recallMemory(event) {
  event.preventDefault();
  const query = byId("memory-query").value.trim();
  if (!query) return;
  try {
    await withButton(event.submitter, async () => {
      const result = await api("/memory/relevant", {
        method: "POST",
        body: { query, limit: 10 },
      });
      renderJson(byId("memory-relevant"), result);
    });
  } catch (error) {
    byId("memory-relevant").innerHTML = errorMarkup(error.message);
  }
}

const DAILY_LOADERS = {
  context: loadContext,
  workspace: loadWorkspace,
};

const DEVELOPER_LOADERS = {
  state: loadState,
  skills: loadSkills,
  dream: loadDream,
  procedures: loadProcedures,
  memory: loadMemory,
};

const READ_ONLY_LOADERS = { ...DAILY_LOADERS, ...DEVELOPER_LOADERS };
let developerPanelsLoaded = false;

async function loadAllReadOnlyPanels() {
  const loaders = [loadHealth(), ...Object.values(DAILY_LOADERS).map((loader) => loader())];
  if (byId("developer-mode").checked) {
    loaders.push(...Object.values(DEVELOPER_LOADERS).map((loader) => loader()));
    developerPanelsLoaded = true;
  }
  await Promise.allSettled(loaders);
}

async function toggleDeveloperMode(event) {
  const enabled = event.currentTarget.checked;
  document.querySelectorAll(".developer-panel").forEach((panel) => {
    panel.hidden = !enabled;
  });
  if (enabled && !developerPanelsLoaded) {
    developerPanelsLoaded = true;
    await Promise.allSettled(Object.values(DEVELOPER_LOADERS).map((loader) => loader()));
  }
}

function bindEvents() {
  byId("chat-form").addEventListener("submit", sendChat);
  byId("generate-briefing").addEventListener("click", (event) => generateBriefing(event.currentTarget));
  byId("context-search-form").addEventListener("submit", searchContext);
  byId("context-remember-form").addEventListener("submit", rememberContext);
  byId("workspace-search-form").addEventListener("submit", searchWorkspace);
  byId("developer-mode").addEventListener("change", toggleDeveloperMode);
  byId("skill-form").addEventListener("submit", runSkill);
  byId("run-dream").addEventListener("click", (event) => runDream(event.currentTarget));
  byId("procedure-match-form").addEventListener("submit", matchProcedure);
  byId("memory-query-form").addEventListener("submit", recallMemory);
  byId("refresh-all").addEventListener("click", (event) => withButton(event.currentTarget, loadAllReadOnlyPanels));

  document.querySelectorAll("[data-refresh]").forEach((button) => {
    button.addEventListener("click", (event) => {
      const loader = READ_ONLY_LOADERS[event.currentTarget.dataset.refresh];
      if (loader) withButton(event.currentTarget, loader);
    });
  });
  byId("dream-promotions").addEventListener("click", (event) => {
    const button = event.target.closest("[data-dream-action]");
    if (button) reviewDream(button);
  });
  byId("procedures-pending").addEventListener("click", (event) => {
    const button = event.target.closest("[data-procedure-action]");
    if (button) reviewProcedure(button);
  });
}

bindEvents();
loadAllReadOnlyPanels();
