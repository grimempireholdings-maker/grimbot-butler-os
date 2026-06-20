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
let latestMachineOutput = null;
let activeStandardView = "conversation";
let briefingGenerated = false;
let recognition = null;
let recognitionBaseText = "";
let recognitionFinalText = "";
let recognitionFailed = false;
let voiceResponsePending = false;

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
  if (log.querySelector(".welcome-message")) log.innerHTML = "";
  if (machineOutput) {
    latestMachineOutput = machineOutput;
    renderConversationDiagnostics();
  }
  log.insertAdjacentHTML("beforeend", `<div class="message ${kind}">
    <div class="message-heading"><span class="speaker">${kind === "user" ? "Julian" : "Maya"}</span>
      ${kind === "maya" ? '<button class="message-speak" type="button" aria-label="Hear Maya reply">&#128266;</button>' : ""}
    </div>
    <div>${escapeHtml(text)}</div>
  </div>`);
  if (kind === "maya") {
    log.lastElementChild.querySelector(".message-speak").addEventListener("click", () => speakVoiceReply(text));
  }
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
  const shouldSpeakReply = voiceResponsePending || byId("speak-replies").checked;
  voiceResponsePending = false;
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
          ambient_mode: byId("ambient-mode").checked,
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
      if (shouldSpeakReply) speakVoiceReply(responseText);
      const ambientBriefingModes = ["morning_ramp", "gentle_orientation", "approval_review"];
      if (result.agent_response?.intent === "chief_of_staff_briefing"
          && !ambientBriefingModes.includes(machineOutput?.conversation_mode)) {
        await openBriefing(true);
      }
    });
  } catch (error) {
    if (shouldSpeakReply) setVoiceState("idle", "The voice message couldn't be sent. You can try again or type.");
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
    briefingGenerated = true;
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

function setVoiceState(state, message) {
  const button = byId("voice-button");
  const label = byId("voice-label");
  button.dataset.state = state;
  button.setAttribute("aria-pressed", state === "listening" ? "true" : "false");
  button.setAttribute("aria-label", state === "listening" ? "Stop push-to-talk" : "Start push-to-talk");
  label.textContent = state === "listening" ? "Listening" : state === "sending" ? "Sending" : "Talk";
  byId("voice-status").textContent = message;
}

function friendlyRecognitionError(code) {
  if (code === "not-allowed" || code === "service-not-allowed") {
    if (!window.isSecureContext) {
      return "Chrome blocked the mic on this HTTP address. Use the keyboard mic, or open Maya over trusted HTTPS.";
    }
    return "Microphone permission is off. You can enable it or keep typing.";
  }
  if (code === "audio-capture") return "No microphone is available. You can keep typing.";
  if (code === "network") return "Voice recognition is unavailable right now. You can keep typing.";
  return "I didn't catch that. Tap Talk to try again, or type instead.";
}

function initializeBrowserVoice() {
  const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!Recognition) {
    const button = byId("voice-button");
    button.dataset.state = "unsupported";
    button.setAttribute("aria-label", "Voice recognition unavailable; open text input");
    byId("voice-label").textContent = "Type";
    byId("voice-status").textContent = "Chrome voice recognition isn't available on this page. Tap Type, then use the keyboard mic or text.";
    return;
  }

  recognition = new Recognition();
  recognition.continuous = false;
  recognition.interimResults = true;
  recognition.maxAlternatives = 1;
  recognition.lang = navigator.language || "en-US";
  recognition.onstart = () => setVoiceState("listening", "Listening now. Tap again to stop.");
  recognition.onresult = (event) => {
    let interim = "";
    for (let index = event.resultIndex; index < event.results.length; index += 1) {
      const words = event.results[index][0].transcript.trim();
      if (event.results[index].isFinal) recognitionFinalText = `${recognitionFinalText} ${words}`.trim();
      else interim = `${interim} ${words}`.trim();
    }
    byId("chat-input").value = [recognitionBaseText, recognitionFinalText, interim].filter(Boolean).join(" ");
  };
  recognition.onerror = (event) => {
    recognitionFailed = true;
    setVoiceState("idle", friendlyRecognitionError(event.error));
  };
  recognition.onend = () => {
    const transcript = recognitionFinalText.trim();
    if (transcript && !recognitionFailed) {
      voiceResponsePending = true;
      setVoiceState("sending", "Sending your voice message…");
      byId("chat-form").requestSubmit();
    } else if (!recognitionFailed) {
      setVoiceState("idle", "Voice ready");
    }
  };
}

function togglePushToTalk() {
  if (!recognition) {
    byId("chat-input").focus();
    byId("voice-status").textContent = "Use the keyboard microphone for dictation, or type your message.";
    return;
  }
  if (byId("voice-button").dataset.state === "listening") {
    setVoiceState("sending", "Finishing your voice message…");
    recognition.stop();
    return;
  }
  if ("speechSynthesis" in window) window.speechSynthesis.cancel();
  recognitionBaseText = byId("chat-input").value.trim();
  recognitionFinalText = "";
  recognitionFailed = false;
  try {
    recognition.start();
  } catch {
    const guidance = window.isSecureContext
      ? "Voice couldn't start. Check Chrome's microphone permission, or use the keyboard mic."
      : "Chrome blocked the mic on this HTTP address. Use the keyboard mic, or open Maya over trusted HTTPS.";
    setVoiceState("idle", guidance);
  }
}

function speakVoiceReply(text) {
  if (!("speechSynthesis" in window) || !("SpeechSynthesisUtterance" in window)) {
    byId("voice-status").textContent = "Chrome speech playback isn't available on this device.";
    return;
  }
  window.speechSynthesis.cancel();
  window.speechSynthesis.resume();
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.lang = navigator.language || "en-US";
  utterance.rate = 1;
  utterance.onstart = () => { byId("voice-status").textContent = "Maya is speaking…"; };
  utterance.onend = () => { byId("voice-status").textContent = "Voice reply ready"; };
  utterance.onerror = () => { byId("voice-status").textContent = "Reply received. Chrome couldn't play the voice; tap the speaker on Maya's message to retry."; };
  window.speechSynthesis.speak(utterance);
}

function setPhotoState(state, message = "") {
  const button = byId("photo-button");
  const status = byId("photo-status");
  button.dataset.state = state;
  byId("photo-label").textContent = state === "analyzing" ? "Looking" : "Photo";
  status.textContent = message;
  status.hidden = !message;
}

async function handlePhotoCapture(event) {
  const input = event.currentTarget;
  const file = input.files?.[0];
  if (!file) {
    setPhotoState("idle");
    return;
  }
  if (file.size > 10 * 1024 * 1024) {
    setPhotoState("idle", "That photo is over 10 MB. Choose a smaller image.");
    input.value = "";
    return;
  }

  const promptInput = byId("chat-input");
  const prompt = promptInput.value.trim() || "What do you notice in this photo?";
  addChatMessage("user", promptInput.value.trim() ? `Shared a photo: ${promptInput.value.trim()}` : "Shared a photo.");
  promptInput.value = "";
  setPhotoState("analyzing", "Analyzing this one photo. No live feed is active.");
  try {
    const response = await fetch(`/vision/photo?prompt=${encodeURIComponent(prompt)}`, {
      method: "POST",
      headers: { "Content-Type": file.type || "application/octet-stream" },
      body: file,
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(errorMessage(payload, "Photo analysis failed"));
    addChatMessage("maya", payload.agent_response.user_response, payload.agent_response.machine_output);
    if (byId("speak-replies").checked) speakVoiceReply(payload.agent_response.user_response);
    setPhotoState("idle", "Photo analyzed. The image bytes were not retained.");
  } catch {
    addChatMessage("maya", "I couldn't analyze that photo. It was not retained; you can try another image or keep chatting.");
    setPhotoState("idle", "Photo analysis unavailable. The image was not retained.");
  } finally {
    input.value = "";
  }
}

function markPhotoPickerOpen() {
  setPhotoState("selecting", "Camera or photo picker open. No live feed is active.");
  byId("photo-input").click();
}

function statusToken(kind, label, value) {
  return `<span class="status-token ${escapeHtml(kind)}" data-status-source="${escapeHtml(kind)}"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></span>`;
}

async function loadStatusTokens() {
  const target = byId("status-tokens");
  const [contextResult, workspaceResult, usageResult, promotionsResult, proceduresResult] = await Promise.allSettled([
    api("/context"), api("/workspace"), api("/search/usage"),
    api("/dream/promotions"), api("/procedures/pending"),
  ]);
  const tokens = [];
  if (contextResult.status === "fulfilled" && Array.isArray(contextResult.value?.priorities)) {
    tokens.push(statusToken("priorities", "Priorities", contextResult.value.priorities.length));
  }
  if (promotionsResult.status === "fulfilled" && proceduresResult.status === "fulfilled"
      && Array.isArray(promotionsResult.value) && Array.isArray(proceduresResult.value)) {
    const pendingFacts = promotionsResult.value.filter((item) => item.status === "pending").length;
    tokens.push(statusToken("approvals", "Pending review", pendingFacts + proceduresResult.value.length));
  }
  if (workspaceResult.status === "fulfilled" && Array.isArray(workspaceResult.value?.recent_commits)
      && workspaceResult.value.recent_commits.length > 0) {
    const shortHash = String(workspaceResult.value.recent_commits[0]).trim().split(/\s+/)[0];
    if (shortHash) tokens.push(statusToken("commit", "Commit", shortHash));
  }
  if (usageResult.status === "fulfilled" && Number.isFinite(usageResult.value?.count)
      && Number.isFinite(usageResult.value?.limit)) {
    tokens.push(statusToken("search", "Search", `${usageResult.value.count}/${usageResult.value.limit}`));
  }
  target.innerHTML = tokens.join("");
  target.hidden = tokens.length === 0;
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

const DEVELOPER_LOADERS = {
  context: loadContext,
  workspace: loadWorkspace,
  state: loadState,
  skills: loadSkills,
  dream: loadDream,
  procedures: loadProcedures,
  memory: loadMemory,
};

const READ_ONLY_LOADERS = { ...DEVELOPER_LOADERS };

async function loadAllReadOnlyPanels() {
  await Promise.allSettled(Object.values(DEVELOPER_LOADERS).map((loader) => loader()));
}

function renderConversationDiagnostics() {
  const target = byId("conversation-diagnostics");
  if (!target) return;
  if (latestMachineOutput) renderJson(target, latestMachineOutput);
}

function showStandardView(view) {
  activeStandardView = view;
  byId("developer-view").hidden = true;
  byId("conversation-view").hidden = view !== "conversation";
  byId("briefing-view").hidden = view !== "briefing";
  document.querySelectorAll("[data-view]").forEach((control) => {
    const active = control.dataset.view === view;
    control.classList.toggle("active", active);
    if (control.matches("button")) control.setAttribute("aria-current", active ? "page" : "false");
  });
}

async function openBriefing(generate = false) {
  if (byId("developer-mode").checked) {
    byId("developer-mode").checked = false;
    unmountDeveloperView();
  }
  showStandardView("briefing");
  if (generate && !briefingGenerated) await generateBriefing(byId("generate-briefing"));
}

function bindDeveloperEvents() {
  byId("context-search-form").addEventListener("submit", searchContext);
  byId("context-remember-form").addEventListener("submit", rememberContext);
  byId("workspace-search-form").addEventListener("submit", searchWorkspace);
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

async function mountDeveloperView() {
  byId("conversation-view").hidden = true;
  byId("briefing-view").hidden = true;
  byId("developer-view").hidden = false;
  document.querySelectorAll("[data-view]").forEach((control) => {
    control.classList.remove("active");
    if (control.matches("button")) control.setAttribute("aria-current", "false");
  });
  const root = byId("developer-root");
  root.replaceChildren(byId("developer-template").content.cloneNode(true));
  bindDeveloperEvents();
  renderConversationDiagnostics();
  await loadAllReadOnlyPanels();
}

function unmountDeveloperView() {
  byId("developer-root").replaceChildren();
  byId("developer-view").hidden = true;
}

async function toggleDeveloperMode(event) {
  if (event.currentTarget.checked) {
    await mountDeveloperView();
  } else {
    unmountDeveloperView();
    showStandardView(activeStandardView);
  }
}

function bindPersistentEvents() {
  byId("chat-form").addEventListener("submit", sendChat);
  byId("voice-button").addEventListener("click", togglePushToTalk);
  byId("photo-button").addEventListener("click", markPhotoPickerOpen);
  byId("photo-input").addEventListener("change", handlePhotoCapture);
  try {
    const savedSpeakReplies = window.localStorage.getItem("mayaSpeakReplies");
    byId("speak-replies").checked = savedSpeakReplies !== "false";
  } catch {
    byId("speak-replies").checked = true;
  }
  byId("speak-replies").addEventListener("change", (event) => {
    try { window.localStorage.setItem("mayaSpeakReplies", String(event.currentTarget.checked)); } catch {}
    if (event.currentTarget.checked) byId("voice-status").textContent = "Maya will speak her replies.";
    else window.speechSynthesis?.cancel();
  });
  window.addEventListener("focus", () => {
    window.setTimeout(() => {
      if (byId("photo-button").dataset.state === "selecting" && !byId("photo-input").files?.length) {
        setPhotoState("idle");
      }
    }, 400);
  });
  byId("generate-briefing").addEventListener("click", (event) => generateBriefing(event.currentTarget));
  byId("developer-mode").addEventListener("change", toggleDeveloperMode);
  document.querySelectorAll("[data-view]").forEach((control) => {
    control.addEventListener("click", async (event) => {
      event.preventDefault();
      const view = event.currentTarget.dataset.view;
      if (view === "briefing") await openBriefing(true);
      else {
        if (byId("developer-mode").checked) {
          byId("developer-mode").checked = false;
          unmountDeveloperView();
        }
        showStandardView("conversation");
      }
    });
  });
}

bindPersistentEvents();
initializeBrowserVoice();
showStandardView("conversation");
Promise.allSettled([loadHealth(), loadStatusTokens()]);
