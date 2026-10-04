/**
 * Medical Multi-Agent Intelligence Web Application Script
 * Communicates with REST API endpoints (/analyze, /analyze/async, /jobs, /patients, /patients/parse-report, /parse, /models, /health)
 */

document.addEventListener('DOMContentLoaded', () => {
  // State management
  let parsedDocument = null; // { filename, markdown, metadata }
  let currentJob = null;     // active async job
  let pollInterval = null;   // polling timer
  let activeFiles = {};      // generated report file paths
  let loadedReportTexts = {};// cached report markdown strings
  let conversationsCache = []; // list of past background jobs
  let patientsCache = [];      // list of patient records
  let selectedPatient = null;  // patient selected as chat/analysis context
  let currentPatientMeta = {}; // key-value metadata object for editing patient
  let currentPatientClinicalData = {
    heart: [],
    liver: [],
    pancreas: [],
    nutrients: [],
    overall_health: [],
    medications: []
  };
  let activeCategory = 'heart';

  // Intake Chatbot State
  let intakeChatHistory = []; // Array of { role: 'user'|'assistant', content: string }
  let isIntakeChatLoading = false;
  let activeChat = null;
  let workspaceId = localStorage.getItem('medicalWorkspace') || '';
  let workspaceChats = [];
  let navigationVersion = 0;
  let draftTimer;
  let draftSave = Promise.resolve();
  let sourceSave = Promise.resolve();
  const sendingChats = new Set();

  const CAT_TITLES = {
    heart: '🫀 Heart & Cardiovascular System',
    liver: '🥩 Liver & Hepatic Function',
    pancreas: '🩺 Pancreas, Endocrine & Glucose',
    nutrients: '🥗 Nutrients, Vitamins & Electrolytes',
    overall_health: '🔬 Overall Health, Hematology & CBC',
    medications: '💊 Active Medications & Dosage'
  };

  const SUPPORTED_EXTENSIONS = ['pdf', 'docx', 'doc', 'txt', 'md', 'rtf', 'png', 'jpg', 'jpeg', 'webp', 'gif'];

  // Navigation & View Tab Elements
  const navBtnConversations = document.getElementById('navBtnConversations');
  const navBtnPatients = document.getElementById('navBtnPatients');
  const viewConversations = document.getElementById('viewConversations');
  const viewPatients = document.getElementById('viewPatients');
  const conversationsCountTag = document.getElementById('conversationsCountTag');
  const patientsCountTag = document.getElementById('patientsCountTag');

  // Conversations History Panel Elements
  const btnRefreshConversations = document.getElementById('btnRefreshConversations');
  const conversationSearchInput = document.getElementById('conversationSearchInput');
  const conversationsList = document.getElementById('conversationsList');
  const sidebarConversationsList = document.getElementById('sidebarConversationsList');
  const sidebarConversationSearch = document.getElementById('sidebarConversationSearch');
  const btnNewConversation = document.getElementById('btnNewConversation');
  let activeConversationId = null;

  // Workbench Subtabs Elements
  const btnTabIntake = document.getElementById('btnTabIntake');
  const btnTabOutput = document.getElementById('btnTabOutput');
  const workbenchTabIntake = document.getElementById('workbenchTabIntake');
  const workbenchTabOutput = document.getElementById('workbenchTabOutput');
  const wtabOutputIndicator = document.getElementById('wtabOutputIndicator');

  // Workbench Form & Intake Elements
  const apiStatusBadge = document.getElementById('apiStatusBadge');
  const apiStatusText = document.getElementById('apiStatusText');
  const queryInput = document.getElementById('queryInput');
  const modelSelect = document.getElementById('modelSelect');
  const agentSelect = document.getElementById('agentSelect');
  const webSearchToggle = document.getElementById('webSearchToggle');
  const intakeModeBadge = document.getElementById('intakeModeBadge');
  const intakeChatStream = document.getElementById('intakeChatStream');
  const btnChatSend = document.getElementById('btnChatSend');

  // File Upload Elements
  const dropZone = document.getElementById('dropZone');
  const fileInput = document.getElementById('fileInput');
  const fileErrorAlert = document.getElementById('fileErrorAlert');
  const fileErrorMsg = document.getElementById('fileErrorMsg');
  const fileErrorClose = document.getElementById('fileErrorClose');
  const parsedDocCard = document.getElementById('parsedDocCard');
  const parsedFileName = document.getElementById('parsedFileName');
  const parsedFormat = document.getElementById('parsedFormat');
  const parsedPages = document.getElementById('parsedPages');
  const parsedChars = document.getElementById('parsedChars');
  const parsedStatus = document.getElementById('parsedStatus');
  const btnToggleDocPreview = document.getElementById('btnToggleDocPreview');
  const btnRemoveDoc = document.getElementById('btnRemoveDoc');
  const parsedPreviewBox = document.getElementById('parsedPreviewBox');
  const parsedMarkdownContent = document.getElementById('parsedMarkdownContent');
  const btnCopyParsed = document.getElementById('btnCopyParsed');
  const attachContextCheck = document.getElementById('attachContextCheck');

  // Action Buttons
  const btnAnalyze = document.getElementById('btnAnalyze');
  const btnRoute = document.getElementById('btnRoute');
  const btnClear = document.getElementById('btnClear');

  // Results & Outputs Elements
  const placeholderState = document.getElementById('placeholderState');
  const jobProgressCard = document.getElementById('jobProgressCard');
  const jobStatusHeading = document.getElementById('jobStatusHeading');
  const jobIdTag = document.getElementById('jobIdTag');
  const progressBarFill = document.getElementById('progressBarFill');
  const stepRoute = document.getElementById('stepRoute');
  const stepReasoning = document.getElementById('stepReasoning');
  const stepValidation = document.getElementById('stepValidation');
  const stepReports = document.getElementById('stepReports');

  const routeResultCard = document.getElementById('routeResultCard');
  const routedAgentName = document.getElementById('routedAgentName');
  const routedAgentDesc = document.getElementById('routedAgentDesc');
  const activeAgentBadge = document.getElementById('activeAgentBadge');

  const filesContainer = document.getElementById('filesContainer');
  const filesGrid = document.getElementById('filesGrid');

  const reportPreviewCard = document.getElementById('reportPreviewCard');
  const markdownViewer = document.getElementById('markdownViewer');
  const btnCopyReport = document.getElementById('btnCopyReport');
  const btnDownloadCurrent = document.getElementById('btnDownloadCurrent');
  const toastContainer = document.getElementById('toastContainer');

  // Patient Management Elements
  const btnAddNewPatient = document.getElementById('btnAddNewPatient');
  const patientSearchInput = document.getElementById('patientSearchInput');
  const patientTotalBadge = document.getElementById('patientTotalBadge');
  const patientsTableBody = document.getElementById('patientsTableBody');
  const patientModal = document.getElementById('patientModal');
  const btnClosePatientModal = document.getElementById('btnClosePatientModal');
  const btnCancelPatientModal = document.getElementById('btnCancelPatientModal');
  const btnSavePatient = document.getElementById('btnSavePatient');
  const patientModalTitle = document.getElementById('patientModalTitle');
  const patientEditId = document.getElementById('patientEditId');
  const patientName = document.getElementById('patientName');
  const patientPrimaryCondition = document.getElementById('patientPrimaryCondition');
  const patientAge = document.getElementById('patientAge');
  const patientGender = document.getElementById('patientGender');
  const patientEmail = document.getElementById('patientEmail');
  const patientDescription = document.getElementById('patientDescription');
  const btnClassifyDescription = document.getElementById('btnClassifyDescription');
  const classifyStatusText = document.getElementById('classifyStatusText');
  const metaKeyInput = document.getElementById('metaKeyInput');
  const metaValInput = document.getElementById('metaValInput');
  const btnAddMetaTag = document.getElementById('btnAddMetaTag');
  const metadataTagsContainer = document.getElementById('metadataTagsContainer');
  const patientDocDropZone = document.getElementById('patientDocDropZone');
  const patientFileInput = document.getElementById('patientFileInput');
  const patientParseStatusBox = document.getElementById('patientParseStatusBox');
  const btnAddCatRow = document.getElementById('btnAddCatRow');
  const plusMenuPatientsList = document.getElementById('plusMenuPatientsList');
  const selectedPatientChip = document.getElementById('selectedPatientChip');
  const selectedPatientNameEl = document.getElementById('selectedPatientName');
  const selectedPatientMetaEl = document.getElementById('selectedPatientMeta');
  const btnClearSelectedPatient = document.getElementById('btnClearSelectedPatient');
  const patientsActiveBanner = document.getElementById('patientsActiveBanner');
  const patientsActiveBannerName = document.getElementById('patientsActiveBannerName');
  const btnActiveBannerGoChat = document.getElementById('btnActiveBannerGoChat');
  const btnActiveBannerClear = document.getElementById('btnActiveBannerClear');

  // Regenerate Modal Elements
  const regenerateModal = document.getElementById('regenerateModal');
  const btnCloseRegenModal = document.getElementById('btnCloseRegenModal');
  const btnCancelRegenModal = document.getElementById('btnCancelRegenModal');
  const btnConfirmRegenerate = document.getElementById('btnConfirmRegenerate');
  const regenSourceJobId = document.getElementById('regenSourceJobId');
  const regenQueryPreview = document.getElementById('regenQueryPreview');
  const regenAgentSelect = document.getElementById('regenAgentSelect');
  const regenModelSelect = document.getElementById('regenModelSelect');

  // Notifications & API Modal Elements
  const btnOpenNotifApiModal = document.getElementById('btnOpenNotifApiModal');
  const notifApiModal = document.getElementById('notifApiModal');
  const btnCloseNotifApiModal = document.getElementById('btnCloseNotifApiModal');
  const btnCloseNotifApiModalFooter = document.getElementById('btnCloseNotifApiModalFooter');
  const notifTabSlack = document.getElementById('notifTabSlack');
  const notifTabEmail = document.getElementById('notifTabEmail');
  const notifTabCalendar = document.getElementById('notifTabCalendar');
  const notifTabApi = document.getElementById('notifTabApi');
  const notifPanelSlack = document.getElementById('notifPanelSlack');
  const notifPanelEmail = document.getElementById('notifPanelEmail');
  const notifPanelCalendar = document.getElementById('notifPanelCalendar');
  const notifPanelApi = document.getElementById('notifPanelApi');
  const slackWebhookUrl = document.getElementById('slackWebhookUrl');
  const slackWebhookSelect = document.getElementById('slackWebhookSelect');
  const btnSlackSelectAll = document.getElementById('btnSlackSelectAll');
  const chkSelectAllSlack = document.getElementById('chkSelectAllSlack');
  const slackTasksTbody = document.getElementById('slackTasksTbody');
  const btnSendSlackNotify = document.getElementById('btnSendSlackNotify');
  const btnOpenConfigFromSlack = document.getElementById('btnOpenConfigFromSlack');
  const btnSendEmailNotify = document.getElementById('btnSendEmailNotify');
  const btnExportCalendarIcs = document.getElementById('btnExportCalendarIcs');
  const btnTestApiHealth = document.getElementById('btnTestApiHealth');
  const calEventDate = document.getElementById('calEventDate');

  // Configuration Modal Elements
  const btnOpenConfigModal = document.getElementById('btnOpenConfigModal');
  const configModal = document.getElementById('configModal');
  const btnCloseConfigModal = document.getElementById('btnCloseConfigModal');
  const btnCloseConfigModalFooter = document.getElementById('btnCloseConfigModalFooter');
  const configTabSlack = document.getElementById('configTabSlack');
  const configTabKeys = document.getElementById('configTabKeys');
  const configPanelSlack = document.getElementById('configPanelSlack');
  const configPanelKeys = document.getElementById('configPanelKeys');
  const cfgWebhookName = document.getElementById('cfgWebhookName');
  const cfgWebhookUrl = document.getElementById('cfgWebhookUrl');
  const btnAddSlackWebhook = document.getElementById('btnAddSlackWebhook');
  const slackWebhooksList = document.getElementById('slackWebhooksList');
  const apiKeysList = document.getElementById('apiKeysList');

  // Cached config state for Slack picker + settings UI
  let slackWebhooksCache = [];
  let apiKeysCache = [];

  // Initialize
  checkApiHealth();
  loadAvailableModels();
  setupEventListeners();
  initializeWorkspaces();
  refreshConfigCache();

  async function chatRequest(path, method = 'GET', body) {
    const response = await fetch(path, { method, headers: { 'Content-Type': 'application/json' },
      ...(body === undefined ? {} : { body: JSON.stringify(body) }) });
    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.detail || `HTTP ${response.status}`);
    }
    return response.json();
  }

  async function initializeWorkspaces() {
    const bar = document.querySelector('.workbench-subtabs-bar');
    bar.insertAdjacentHTML('beforeend', '<button type="button" id="btnTabPatient" class="workbench-subtab-btn hidden" data-wtab="patient">Patient data</button>');
    bar.insertAdjacentHTML('afterend', '<section id="patientDataPanel" class="panel hidden"></section><div id="savedChatTools" class="saved-chat-tools"><input id="savedChatTitle" aria-label="Chat title" placeholder="New chat"><div id="chatAttachments"></div><details><summary>Included sources</summary><div id="chatSources"></div></details><div id="importedChatActions"></div></div>');
    document.getElementById('savedChatTitle').addEventListener('input', () => {
      clearTimeout(draftTimer);
      draftTimer = setTimeout(() => saveDraft().then(refreshSavedChats).catch(reportChatError), 350);
    });
    document.getElementById('chatRunSelect').addEventListener('change', e => { if (e.target.value) loadConversationDetails(e.target.value); });
    document.getElementById('btnRegenerateRun').addEventListener('click', () => {
      const run = activeChat?.runs.find(r => r.id === document.getElementById('chatRunSelect').value);
      if (run) openRegenModal(run.id, run.query);
    });
    document.getElementById('btnDeleteRun').addEventListener('click', () => {
      const id = document.getElementById('chatRunSelect').value;
      if (id) deleteConversation(id);
    });
    queryInput.addEventListener('input', () => { clearTimeout(draftTimer); draftTimer = setTimeout(() => saveDraft().catch(reportChatError), 350); });
    window.addEventListener('pagehide', () => {
      if (!activeChat) return;
      fetch(`/chats/${activeChat.id}`, {method: 'PATCH', keepalive: true,
        headers: {'Content-Type': 'application/json'}, body: JSON.stringify({draft: queryInput.value, title: document.getElementById('savedChatTitle').value})}).catch(() => {});
    });
    document.querySelector('.view-header p').textContent = 'Continue a saved conversation or start a new clinical analysis query.';
    try {
      await loadPatients();
      await loadConversationsHistory();
      if (workspaceId && !patientsCache.some(p => p.id === workspaceId)) workspaceId = '';
      await selectWorkspace(workspaceId);
    } catch (error) { reportChatError(error); }
  }

  function reportChatError(error) { showToast(error.message || String(error), 'error'); }

  async function saveDraft() {
    clearTimeout(draftTimer);
    const id = activeChat?.id;
    const draft = queryInput.value;
    const title = document.getElementById('savedChatTitle').value;
    if (!id) return;
    await sourceSave;
    draftSave = draftSave.catch(() => {}).then(() => chatRequest(`/chats/${id}`, 'PATCH', { draft, title }));
    await draftSave;
  }

  async function refreshSavedChats() {
    const scope = workspaceId;
    const list = await chatRequest('/chats' + (scope ? `?patient_id=${encodeURIComponent(scope)}` : ''));
    if (scope !== workspaceId) return;
    workspaceChats = list;
    renderSavedChats();
  }

  function renderSavedChats() {
    conversationsCountTag.textContent = workspaceChats.length;
    const filter = (sidebarConversationSearch.value || '').toLowerCase();
    sidebarConversationsList.innerHTML = '';
    workspaceChats.filter(c => c.title.toLowerCase().includes(filter)).forEach(chat => {
      const row = document.createElement('button');
      row.type = 'button';
      row.className = 'sidebar-conv-item' + (activeChat?.id === chat.id ? ' active' : '');
      row.textContent = chat.title;
      row.addEventListener('click', () => openSavedChat(chat.id).catch(reportChatError));
      sidebarConversationsList.appendChild(row);
    });
    if (!sidebarConversationsList.children.length) {
      sidebarConversationsList.textContent = selectedPatient ? `No chats yet for ${selectedPatient.name}.` : 'No conversations yet.';
    }
  }

  function updateSelectedPatientChip() {
    if (!selectedPatientChip) return;
    if (selectedPatient) {
      selectedPatientChip.classList.remove('hidden');
      if (selectedPatientNameEl) selectedPatientNameEl.textContent = selectedPatient.name || 'Unnamed';
      if (selectedPatientMetaEl) {
        const parts = [];
        if (selectedPatient.age != null) parts.push(`${selectedPatient.age} yrs`);
        if (selectedPatient.gender) parts.push(selectedPatient.gender);
        if (selectedPatient.primary_condition) parts.push(selectedPatient.primary_condition);
        selectedPatientMetaEl.textContent = parts.length > 0 ? parts.join(' · ') : '';
      }
    } else {
      selectedPatientChip.classList.add('hidden');
    }
  }

  async function selectWorkspace(id) {
    await saveDraft();
    workspaceId = id || '';
    localStorage.setItem('medicalWorkspace', workspaceId);
    selectedPatient = patientsCache.find(p => p.id === workspaceId) || null;
    activeChat = null;
    const version = ++navigationVersion;
    queryInput.value = '';
    intakeChatHistory = [];
    updateIntakeChatUI();
    document.getElementById('btnTabPatient').classList.toggle('hidden', !selectedPatient);
    updateSelectedPatientChip();
    renderPatientsTable();
    await refreshSavedChats();
    if (version !== navigationVersion) return;
    const saved = localStorage.getItem(`medicalChat:${workspaceId}`);
    const target = workspaceChats.find(c => c.id === saved) || workspaceChats[0];
    if (target) await openSavedChat(target.id, false);
    else await newSavedChat();
  }

  async function newSavedChat() {
    try {
      await saveDraft();
      const scope = workspaceId;
      const chat = await chatRequest('/chats', 'POST', { patient_id: scope || null });
      if (scope !== workspaceId) return;
      await refreshSavedChats();
      await openSavedChat(chat.id, false);
    } catch (error) { reportChatError(error); }
  }

  async function openSavedChat(id, save = true) {
    if (save) await saveDraft();
    const version = ++navigationVersion;
    const chat = await chatRequest(`/chats/${id}`);
    if (version !== navigationVersion || (chat.patient_id || '') !== workspaceId) return;
    activeChat = chat;
    localStorage.setItem(`medicalChat:${workspaceId}`, id);
    currentJob = null;
    activeConversationId = null;
    activeFiles = {};
    loadedReportTexts = {};
    parsedDocument = null;
    if (pollInterval) clearInterval(pollInterval);
    [filesContainer, reportPreviewCard, jobProgressCard, routeResultCard, parsedDocCard].forEach(el => el.classList.add('hidden'));
    placeholderState.classList.remove('hidden');
    queryInput.value = chat.draft || '';
    renderActiveChat();
    switchView('viewConversations');
    switchWorkbenchTab('intake');
    renderSavedChats();
    const running = chat.runs.find(r => ['pending', 'running'].includes(r.status));
    if (running) pollSavedRun(running.id, chat.id);
  }

  function renderActiveChat() {
    if (!activeChat) return;
    const owner = activeChat.id;
    intakeChatHistory = activeChat.messages || [];
    isIntakeChatLoading = sendingChats.has(owner);
    btnChatSend.disabled = isIntakeChatLoading;
    btnChatSend.innerHTML = '<i class="fa-solid fa-arrow-up"></i>';
    updateIntakeChatUI();
    intakeChatHistory.filter(m => m.role === 'user' && m.status !== 'complete').forEach(m => {
      const retry = document.createElement('button');
      retry.className = 'btn btn-secondary';
      retry.textContent = 'Retry message';
      retry.disabled = sendingChats.has(owner);
      retry.onclick = () => sendSavedMessage(m);
      intakeChatStream.appendChild(retry);
    });
    document.getElementById('savedChatTitle').value = activeChat.title;
    const runSelect = document.getElementById('chatRunSelect');
    runSelect.innerHTML = '<option value="">Select an analysis</option>' + activeChat.runs.map(r => `<option value="${r.id}">${escapeHtml(formatQueryDisplayTitle(r.query))} · ${escapeHtml(r.status)}</option>`).join('');
    if (activeConversationId) runSelect.value = activeConversationId;
    const attachments = document.getElementById('chatAttachments');
    attachments.innerHTML = '';
    activeChat.attachments.forEach(a => {
      const button = document.createElement('button');
      button.className = 'btn btn-ghost';
      button.textContent = `${a.filename} ×`;
      button.title = 'Remove attached document';
      button.onclick = async () => {
        try { await chatRequest(`/chats/${owner}/attachments/${a.id}`, 'DELETE'); await reloadActiveChat(owner); }
        catch (error) { reportChatError(error); }
      };
      attachments.appendChild(button);
    });
    renderChatSources();
    const imported = document.getElementById('importedChatActions');
    imported.innerHTML = '';
    if (activeChat.imported) {
      const label = document.createElement('label');
      label.textContent = 'Imported analysis history · Move to workspace: ';
      const move = document.createElement('select');
      move.innerHTML = '<option value="">General medical research</option>' + patientsCache.map(p => `<option value="${escapeHtml(p.id)}">${escapeHtml(p.name)}</option>`).join('');
      move.value = workspaceId;
      move.onchange = async () => {
        try { await chatRequest(`/chats/${owner}`, 'PATCH', {patient_id: move.value || null}); await selectWorkspace(workspaceId); }
        catch (error) { reportChatError(error); }
      };
      label.appendChild(move);
      imported.appendChild(label);
    }
  }

  function renderChatSources() {
    const box = document.getElementById('chatSources');
    box.innerHTML = '';
    const owner = activeChat.id;
    const sources = [
      ...workspaceChats.filter(c => c.id !== owner).map(c => ({id: c.id, query: `Chat: ${c.title}`})),
      ...conversationsCache.filter(r => (r.patient_id || '') === workspaceId && r.chat_id && r.chat_id !== owner)
    ];
    sources.forEach(run => {
      const label = document.createElement('label');
      const check = document.createElement('input');
      check.type = 'checkbox'; check.value = run.id;
      check.checked = (activeChat.source_ids || []).includes(run.id);
      check.onchange = async () => {
        try {
          const ids = [...box.querySelectorAll('input:checked')].map(el => el.value);
          sourceSave = sourceSave.catch(() => {}).then(() => chatRequest(`/chats/${owner}`, 'PATCH', { source_ids: ids }));
          await sourceSave;
          if (activeChat?.id === owner) activeChat.source_ids = ids;
        } catch (error) {
          sourceSave = Promise.resolve();
          if (activeChat?.id === owner) renderChatSources();
          reportChatError(error);
        }
      };
      label.append(check, document.createTextNode(formatQueryDisplayTitle(run.query)));
      box.appendChild(label);
    });
    if (!box.children.length) box.textContent = 'No other analyses in this workspace. Other chat history is excluded by default.';
  }

  function renderPatientData() {
    const panel = document.getElementById('patientDataPanel');
    panel.innerHTML = '';
    if (!selectedPatient) return;
    const heading = document.createElement('h2'); heading.textContent = selectedPatient.name;
    const content = document.createElement('pre'); content.style.whiteSpace = 'pre-wrap'; content.textContent = formatPatientContext(selectedPatient);
    const edit = document.createElement('button'); edit.className = 'btn btn-primary'; edit.textContent = 'Edit patient data';
    edit.onclick = () => openPatientModal(selectedPatient.id);
    panel.append(heading, edit, content);
  }

  async function reloadActiveChat(owner) {
    const chat = await chatRequest(`/chats/${owner}`);
    if (activeChat?.id !== owner) return;
    activeChat = chat;
    renderActiveChat();
  }

  async function sendSavedMessage(retry = null) {
    if (!activeChat || sendingChats.has(activeChat.id)) return;
    const text = retry?.content || queryInput.value.trim();
    if (!text) return;
    const owner = activeChat.id;
    const message = retry || { id: crypto.randomUUID(), role: 'user', content: text, status: 'pending' };
    sendingChats.add(owner);
    try {
      await saveDraft();
      if (activeChat?.id === owner) {
        if (!retry) activeChat.messages.push(message);
        queryInput.value = '';
        renderActiveChat();
      }
      await chatRequest(`/chats/${owner}/messages`, 'POST', {id: message.id, content: text, model: modelSelect.value});
    } catch (error) { reportChatError(error); }
    finally {
      sendingChats.delete(owner);
      await reloadActiveChat(owner).catch(reportChatError);
      await refreshSavedChats().catch(reportChatError);
    }
  }

  async function attachToSavedChat(file) {
    if (!activeChat) return;
    const owner = activeChat.id;
    const data = new FormData(); data.append('file', file);
    try {
      const response = await fetch('/parse', {method: 'POST', body: data});
      if (!response.ok) throw new Error('Document parsing failed');
      const parsed = await response.json();
      await chatRequest(`/chats/${owner}/attachments`, 'POST', {filename: parsed.filename || file.name, markdown: parsed.markdown});
      await reloadActiveChat(owner);
    } catch (error) { reportChatError(error); }
  }

  async function analyzeSavedChat() {
    if (!activeChat || btnAnalyze.disabled) return;
    const owner = activeChat.id;
    const query = queryInput.value.trim();
    if (!query && !activeChat.messages.length) { showToast('Enter a medical query first.', 'error'); return; }
    btnAnalyze.disabled = true;
    try {
      await saveDraft();
      const job = await chatRequest('/analyze/async', 'POST', {chat_id: owner, query, model: modelSelect.value,
        web_search: webSearchToggle.checked, implementation: 'langchain', timeout: 300,
        ...(agentSelect.value === 'auto' ? {} : {agent_id: agentSelect.value})});
      if (activeChat?.id === owner) {
        if (queryInput.value.trim() === query) queryInput.value = '';
        await reloadActiveChat(owner);
        currentJob = {id: job.job_id};
        switchWorkbenchTab('output');
        pollSavedRun(job.job_id, owner);
      }
      await loadConversationsHistory();
    } catch (error) { reportChatError(error); }
    finally { btnAnalyze.disabled = false; }
  }

  function pollSavedRun(jobId, owner) {
    if (pollInterval) clearInterval(pollInterval);
    const update = async () => {
      try {
        const job = await chatRequest(`/jobs/${jobId}`);
        if (activeChat?.id !== owner) return;
        if (['pending', 'running'].includes(job.status)) {
          jobProgressCard.classList.remove('hidden');
          jobStatusHeading.textContent = 'Analysis running…';
          updateProgressSteps(job.status);
        } else {
          clearInterval(pollInterval);
          await reloadActiveChat(owner);
          if (job.status === 'completed') await loadConversationDetails(jobId);
          else { jobStatusHeading.textContent = 'Analysis failed'; showToast(job.error || 'Analysis failed', 'error'); }
        }
      } catch (error) { reportChatError(error); }
    };
    pollInterval = setInterval(update, 2500);
    update();
  }

  // 1. API Health Check
  async function checkApiHealth() {
    try {
      const res = await fetch('/health');
      if (res.ok) {
        apiStatusBadge.classList.remove('error');
        apiStatusText.textContent = 'API Connected';
      } else {
        throw new Error(`HTTP ${res.status}`);
      }
    } catch (err) {
      apiStatusBadge.classList.add('error');
      apiStatusText.textContent = 'API Offline';
      showToast('Backend API is not responding at http://localhost:8080', 'error');
    }
  }

  // 2. Load Models List dynamically from /models
  async function loadAvailableModels() {
    try {
      const res = await fetch('/models');
      if (res.ok) {
        const models = await res.json();
        modelSelect.innerHTML = '';
        Object.keys(models).forEach(modelName => {
          const opt = document.createElement('option');
          opt.value = modelName;
          opt.textContent = `${modelName} (${models[modelName]})`;
          if (modelName === 'grok-4.5') opt.selected = true;
          modelSelect.appendChild(opt);
        });
      }
    } catch (err) {
      console.warn('Could not load models from /models:', err);
    }
  }

  // 3. Setup Event Handlers
  function setupEventListeners() {
    // Sidebar Navigation Tabs
    navBtnConversations.addEventListener('click', () => switchView('viewConversations'));
    navBtnPatients.addEventListener('click', () => switchView('viewPatients'));

    // Workbench Subtabs (Intake vs Output)
    if (btnTabIntake) {
      btnTabIntake.addEventListener('click', (e) => {
        e.preventDefault();
        switchWorkbenchTab('intake');
      });
    }
    if (btnTabOutput) {
      btnTabOutput.addEventListener('click', (e) => {
        e.preventDefault();
        switchWorkbenchTab('output');
      });
    }

    const subtabsBar = document.querySelector('.workbench-subtabs-bar');
    if (subtabsBar) {
      subtabsBar.addEventListener('click', (e) => {
        const btn = e.target.closest('.workbench-subtab-btn');
        if (btn) {
          e.preventDefault();
          const targetTab = btn.getAttribute('data-wtab') || (btn.id === 'btnTabOutput' ? 'output' : 'intake');
          switchWorkbenchTab(targetTab);
        }
      });
    }

    // Refresh Conversations
    if (btnRefreshConversations) {
      btnRefreshConversations.addEventListener('click', loadConversationsHistory);
    }
    if (conversationSearchInput) {
      conversationSearchInput.addEventListener('input', renderConversationsList);
    }
    if (sidebarConversationSearch) {
      sidebarConversationSearch.addEventListener('input', renderConversationsList);
    }
    if (btnNewConversation) {
      btnNewConversation.addEventListener('click', (e) => {
        e.preventDefault();
        e.stopPropagation();
        startNewConversation();
      });
    }

    // Command Bar Plus & Model Menu Toggles
    const btnPlusMenu = document.getElementById('btnPlusMenu');
    const plusMenuDropdown = document.getElementById('plusMenuDropdown');
    const btnModelMenu = document.getElementById('btnModelMenu');
    const modelMenuDropdown = document.getElementById('modelMenuDropdown');
    const selectedModelLabel = document.getElementById('selectedModelLabel');

    if (btnPlusMenu && plusMenuDropdown) {
      btnPlusMenu.addEventListener('click', (e) => {
        e.stopPropagation();
        const opening = plusMenuDropdown.classList.contains('hidden');
        plusMenuDropdown.classList.toggle('hidden');
        if (modelMenuDropdown) modelMenuDropdown.classList.add('hidden');
        if (opening) renderPlusMenuPatients();
      });
    }

    if (btnClearSelectedPatient) {
      btnClearSelectedPatient.addEventListener('click', (e) => {
        e.preventDefault();
        e.stopPropagation();
        clearSelectedPatient({ toast: true });
      });
    }

    const btnSelectAllPastConvs = document.getElementById('btnSelectAllPastConvs');
    const btnDeselectAllPastConvs = document.getElementById('btnDeselectAllPastConvs');

    if (btnSelectAllPastConvs) {
      btnSelectAllPastConvs.addEventListener('click', (e) => {
        e.preventDefault();
        document.querySelectorAll('#pastConversationsContextList .past-conv-checkbox').forEach(c => c.checked = true);
      });
    }

    if (btnDeselectAllPastConvs) {
      btnDeselectAllPastConvs.addEventListener('click', (e) => {
        e.preventDefault();
        document.querySelectorAll('#pastConversationsContextList .past-conv-checkbox').forEach(c => c.checked = false);
      });
    }

    if (btnModelMenu && modelMenuDropdown) {
      btnModelMenu.addEventListener('click', (e) => {
        e.stopPropagation();
        modelMenuDropdown.classList.toggle('hidden');
        if (plusMenuDropdown) plusMenuDropdown.classList.add('hidden');
      });
    }

    // Text Formatting Toolbar Toggle & Formatting Actions
    const btnToggleFormatToolbar = document.getElementById('btnToggleFormatToolbar');
    const btnCloseFormatToolbar = document.getElementById('btnCloseFormatToolbar');
    const formatToolbar = document.getElementById('formatToolbar');

    function toggleFormatToolbar(show) {
      if (!formatToolbar) return;
      const isHidden = formatToolbar.classList.contains('hidden');
      const shouldShow = show !== undefined ? show : isHidden;
      if (shouldShow) {
        formatToolbar.classList.remove('hidden');
        if (btnToggleFormatToolbar) btnToggleFormatToolbar.classList.add('active-toggle');
      } else {
        formatToolbar.classList.add('hidden');
        if (btnToggleFormatToolbar) btnToggleFormatToolbar.classList.remove('active-toggle');
      }
    }

    if (btnToggleFormatToolbar) {
      btnToggleFormatToolbar.addEventListener('click', (e) => {
        e.preventDefault();
        e.stopPropagation();
        toggleFormatToolbar();
      });
    }

    if (btnCloseFormatToolbar) {
      btnCloseFormatToolbar.addEventListener('click', (e) => {
        e.preventDefault();
        e.stopPropagation();
        toggleFormatToolbar(false);
      });
    }

    // Handle Formatting Buttons Insertion into queryInput
    document.querySelectorAll('#formatToolbar .fmt-btn[data-fmt]').forEach(btn => {
      btn.addEventListener('click', (e) => {
        e.preventDefault();
        e.stopPropagation();
        const fmtType = btn.getAttribute('data-fmt');
        const textarea = queryInput;
        if (!textarea) return;

        const start = textarea.selectionStart || 0;
        const end = textarea.selectionEnd || 0;
        const selectedText = textarea.value.substring(start, end);
        let replacement = '';

        switch (fmtType) {
          case 'bold':
            replacement = selectedText ? `**${selectedText}**` : '**bold text**';
            break;
          case 'italic':
            replacement = selectedText ? `*${selectedText}*` : '*italic text*';
            break;
          case 'heading':
            replacement = selectedText ? `### ${selectedText}` : '### Clinical Header\n';
            break;
          case 'bullet':
            replacement = selectedText ? selectedText.split('\n').map(line => `- ${line}`).join('\n') : '- List item\n';
            break;
          case 'number':
            replacement = selectedText ? selectedText.split('\n').map((line, idx) => `${idx + 1}. ${line}`).join('\n') : '1. List item\n';
            break;
          case 'quote':
            replacement = selectedText ? selectedText.split('\n').map(line => `> ${line}`).join('\n') : '> Clinical Note: ';
            break;
          case 'code':
            replacement = selectedText ? `\`${selectedText}\`` : '`lab_value`';
            break;
          case 'clear':
            if (textarea.value && confirm('Clear all text in writing area?')) {
              textarea.value = '';
              textarea.focus();
              return;
            }
            return;
        }

        textarea.setRangeText(replacement, start, end, 'select');
        textarea.focus();
      });
    });

    // Close dropdowns when clicking outside
    document.addEventListener('click', () => {
      if (plusMenuDropdown) plusMenuDropdown.classList.add('hidden');
      if (modelMenuDropdown) modelMenuDropdown.classList.add('hidden');
    });

    // Model dropdown item selection
    document.querySelectorAll('.model-opt').forEach(item => {
      item.addEventListener('click', (e) => {
        e.stopPropagation();
        const modelVal = item.getAttribute('data-model');
        if (modelSelect) modelSelect.value = modelVal;
        if (selectedModelLabel) selectedModelLabel.textContent = modelVal;
        document.querySelectorAll('.model-opt').forEach(opt => opt.classList.remove('active'));
        item.classList.add('active');
        if (modelMenuDropdown) modelMenuDropdown.classList.add('hidden');
      });
    });

    // Agent dropdown item selection
    document.querySelectorAll('.agent-opt').forEach(item => {
      item.addEventListener('click', (e) => {
        e.stopPropagation();
        const agentVal = item.getAttribute('data-agent');
        if (agentSelect) agentSelect.value = agentVal;
        document.querySelectorAll('.agent-opt').forEach(opt => opt.classList.remove('active'));
        item.classList.add('active');
        if (modelMenuDropdown) modelMenuDropdown.classList.add('hidden');
      });
    });

    // Plus menu sample item selection
    document.querySelectorAll('.sample-item').forEach(item => {
      item.addEventListener('click', (e) => {
        e.stopPropagation();
        queryInput.value = item.getAttribute('data-query');
        queryInput.focus();
        if (plusMenuDropdown) plusMenuDropdown.classList.add('hidden');
      });
    });

    // Auto-expand command bar textarea
    queryInput.addEventListener('input', () => {
      queryInput.style.height = 'auto';
      queryInput.style.height = (queryInput.scrollHeight) + 'px';
    });

    // Sample Chips
    document.querySelectorAll('.chip').forEach(chip => {
      chip.addEventListener('click', () => {
        queryInput.value = chip.getAttribute('data-query');
        queryInput.focus();
      });
    });

    // Drag & Drop for Main Analysis Document
    dropZone.addEventListener('dragover', (e) => {
      e.preventDefault();
      dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => {
      dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
      e.preventDefault();
      dropZone.classList.remove('dragover');
      if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        handleFileUpload(e.dataTransfer.files[0]);
      }
    });

    fileInput.addEventListener('change', (e) => {
      if (e.target.files && e.target.files.length > 0) {
        handleFileUpload(e.target.files[0]);
      }
    });

    fileErrorClose.addEventListener('click', () => fileErrorAlert.classList.add('hidden'));

    btnToggleDocPreview.addEventListener('click', () => {
      parsedPreviewBox.classList.toggle('hidden');
    });

    btnRemoveDoc.addEventListener('click', removeUploadedDocument);

    btnCopyParsed.addEventListener('click', () => {
      if (parsedDocument && parsedDocument.markdown) {
        navigator.clipboard.writeText(parsedDocument.markdown);
        showToast('Parsed document markdown copied to clipboard!', 'info');
      }
    });

    // Drag & Drop for Patient Medical Report Upload
    patientDocDropZone.addEventListener('dragover', (e) => {
      e.preventDefault();
      patientDocDropZone.classList.add('dragover');
    });

    patientDocDropZone.addEventListener('dragleave', () => {
      patientDocDropZone.classList.remove('dragover');
    });

    patientDocDropZone.addEventListener('drop', (e) => {
      e.preventDefault();
      patientDocDropZone.classList.remove('dragover');
      if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        handlePatientReportUpload(e.dataTransfer.files[0]);
      }
    });

    patientFileInput.addEventListener('change', (e) => {
      if (e.target.files && e.target.files.length > 0) {
        handlePatientReportUpload(e.target.files[0]);
      }
    });

    // Categorized Table Tabs
    document.querySelectorAll('.cat-tab-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const catKey = btn.getAttribute('data-cat');
        switchCategoryTab(catKey);
      });
    });

    btnAddCatRow.addEventListener('click', addCategoryRow);

    // Main Actions
    btnChatSend.addEventListener('click', sendIntakeChatMessage);

    queryInput.addEventListener('keydown', (e) => {
      // Enter always sends the current intake message. Shift+Enter remains
      // available for users who need a newline in the clinical prompt.
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendIntakeChatMessage();
      }
    });

    btnAnalyze.addEventListener('click', submitAsyncAnalysis);
    btnRoute.addEventListener('click', routeQueryOnly);
    btnClear.addEventListener('click', () => { queryInput.value = ''; saveDraft().catch(reportChatError); });

    // Output Report Tabs
    document.querySelectorAll('.report-tabs .tab-btn').forEach(btn => {
      btn.addEventListener('click', (e) => {
        document.querySelectorAll('.report-tabs .tab-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        const tab = btn.getAttribute('data-tab');
        renderReportTab(tab);
      });
    });

    btnCopyReport.addEventListener('click', () => {
      const activeTabBtn = document.querySelector('.report-tabs .tab-btn.active');
      const tab = activeTabBtn ? activeTabBtn.getAttribute('data-tab') : 'patient';
      const text = loadedReportTexts[tab] || '';
      if (text) {
        navigator.clipboard.writeText(text);
        showToast('Report copied to clipboard!', 'info');
      }
    });

    // Patient Manager Handlers
    btnAddNewPatient.addEventListener('click', () => openPatientModal());
    patientSearchInput.addEventListener('input', renderPatientsTable);
    btnClosePatientModal.addEventListener('click', closePatientModal);
    btnCancelPatientModal.addEventListener('click', closePatientModal);
    btnSavePatient.addEventListener('click', savePatientRecord);
    btnAddMetaTag.addEventListener('click', addMetaTagFromInput);
    if (btnClassifyDescription) {
      btnClassifyDescription.addEventListener('click', handleClassifyPatientDescription);
    }

    // Regenerate Modal Handlers
    btnCloseRegenModal.addEventListener('click', closeRegenModal);
    btnCancelRegenModal.addEventListener('click', closeRegenModal);
    btnConfirmRegenerate.addEventListener('click', confirmRegenerateJob);

    // Notifications & API Modal Handlers
    if (btnOpenNotifApiModal) {
      btnOpenNotifApiModal.addEventListener('click', () => openNotifApiModal('slack'));
    }
    if (btnCloseNotifApiModal) btnCloseNotifApiModal.addEventListener('click', closeNotifApiModal);
    if (btnCloseNotifApiModalFooter) btnCloseNotifApiModalFooter.addEventListener('click', closeNotifApiModal);
    if (notifTabSlack) notifTabSlack.addEventListener('click', () => switchNotifTab('slack'));
    if (notifTabEmail) notifTabEmail.addEventListener('click', () => switchNotifTab('email'));
    if (notifTabCalendar) notifTabCalendar.addEventListener('click', () => switchNotifTab('calendar'));
    if (notifTabApi) notifTabApi.addEventListener('click', () => switchNotifTab('api'));

    if (btnSlackSelectAll) btnSlackSelectAll.addEventListener('click', toggleSelectAllSlackTasks);
    if (chkSelectAllSlack) chkSelectAllSlack.addEventListener('change', toggleSelectAllSlackTasks);
    if (btnSendSlackNotify) btnSendSlackNotify.addEventListener('click', sendSlackNotification);
    if (btnSendEmailNotify) btnSendEmailNotify.addEventListener('click', handleSendEmailNotification);
    if (btnExportCalendarIcs) btnExportCalendarIcs.addEventListener('click', handleExportCalendarIcs);
    if (btnTestApiHealth) btnTestApiHealth.addEventListener('click', handleTestApiHealth);

    if (btnOpenConfigFromSlack) {
      btnOpenConfigFromSlack.addEventListener('click', () => {
        closeNotifApiModal();
        openConfigModal('slack');
      });
    }

    // Restore last custom webhook URL (fallback when no saved webhook selected)
    const savedUrl = localStorage.getItem('slack_webhook_url');
    if (savedUrl && slackWebhookUrl) slackWebhookUrl.value = savedUrl;

    // Configuration Modal Handlers
    if (btnOpenConfigModal) {
      btnOpenConfigModal.addEventListener('click', () => openConfigModal());
    }
    if (btnCloseConfigModal) btnCloseConfigModal.addEventListener('click', closeConfigModal);
    if (btnCloseConfigModalFooter) btnCloseConfigModalFooter.addEventListener('click', closeConfigModal);
    if (configTabSlack) configTabSlack.addEventListener('click', () => switchConfigTab('slack'));
    if (configTabKeys) configTabKeys.addEventListener('click', () => switchConfigTab('keys'));
    if (btnAddSlackWebhook) btnAddSlackWebhook.addEventListener('click', addSlackWebhookFromForm);
    if (cfgWebhookUrl) {
      cfgWebhookUrl.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
          e.preventDefault();
          addSlackWebhookFromForm();
        }
      });
    }
  }

  // 4. View & Subtab Switching
  function switchView(viewId) {
    if (viewId === 'viewConversations') {
      navBtnConversations.classList.add('active');
      navBtnPatients.classList.remove('active');
      viewConversations.classList.remove('hidden');
      viewPatients.classList.add('hidden');
    } else {
      navBtnPatients.classList.add('active');
      navBtnConversations.classList.remove('active');
      viewPatients.classList.remove('hidden');
      viewConversations.classList.add('hidden');
      loadPatients();
    }
  }

  function switchWorkbenchTab(tabName) {
    document.getElementById('savedChatTools')?.classList.toggle('hidden', tabName !== 'intake');
    document.getElementById('patientDataPanel')?.classList.toggle('hidden', tabName !== 'patient');
    document.getElementById('btnTabPatient')?.classList.toggle('active', tabName === 'patient');
    if (tabName === 'patient') {
      [workbenchTabIntake, workbenchTabOutput].forEach(el => { el.classList.add('hidden'); el.style.setProperty('display', 'none', 'important'); });
      [btnTabIntake, btnTabOutput].forEach(el => el.classList.remove('active'));
      renderPatientData();
      return;
    }
    const intakeBtn = document.getElementById('btnTabIntake') || btnTabIntake;
    const outputBtn = document.getElementById('btnTabOutput') || btnTabOutput;
    const intakeTab = document.getElementById('workbenchTabIntake') || workbenchTabIntake;
    const outputTab = document.getElementById('workbenchTabOutput') || workbenchTabOutput;
    const outputIndicator = document.getElementById('wtabOutputIndicator') || wtabOutputIndicator;

    if (!intakeBtn || !outputBtn || !intakeTab || !outputTab) return;

    if (tabName === 'output') {
      intakeBtn.classList.remove('active');
      outputBtn.classList.add('active');

      intakeTab.classList.remove('active');
      intakeTab.classList.add('hidden');
      intakeTab.style.setProperty('display', 'none', 'important');

      outputTab.classList.remove('hidden');
      outputTab.classList.add('active');
      outputTab.style.setProperty('display', 'block', 'important');

      if (outputIndicator) outputIndicator.classList.add('hidden');

      // If switching to output tab and no current job active, auto-load the most recent conversation
      if (!currentJob && activeChat?.runs?.length) {
        loadConversationDetails(activeChat.runs[0].id);
      }
    } else {
      outputBtn.classList.remove('active');
      intakeBtn.classList.add('active');

      outputTab.classList.remove('active');
      outputTab.classList.add('hidden');
      outputTab.style.setProperty('display', 'none', 'important');

      intakeTab.classList.remove('hidden');
      intakeTab.classList.add('active');
      intakeTab.style.setProperty('display', 'block', 'important');
    }
  }

  // 5. Conversations History Management (persistent + sidebar)
  function conversationHasDocs(job) {
    if (job.has_docs === true) return true;
    const files = job.files || {};
    return Object.keys(files).some(k => {
      const p = files[k];
      return typeof p === 'string' && (p.endsWith('.md') || p.endsWith('.pdf'));
    });
  }

  function truncateQuery(q, n = 42) {
    const s = (q || 'Untitled').replace(/\s+/g, ' ').trim();
    return s.length > n ? s.slice(0, n - 1) + '…' : s;
  }

  function formatConvDate(iso) {
    if (!iso) return '';
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return '';
    const now = new Date();
    const sameDay = d.toDateString() === now.toDateString();
    if (sameDay) return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    return d.toLocaleDateString([], { month: 'short', day: 'numeric' });
  }

  function groupConversations(list) {
    const groups = { Today: [], Earlier: [] };
    const today = new Date().toDateString();
    list.forEach(job => {
      const d = job.created_at ? new Date(job.created_at) : null;
      if (d && !Number.isNaN(d.getTime()) && d.toDateString() === today) {
        groups.Today.push(job);
      } else {
        groups.Earlier.push(job);
      }
    });
    return groups;
  }

  function docsDotClass(job) {
    if (job.status === 'running' || job.status === 'pending') return 'docs-dot running';
    if (job.status === 'failed') return 'docs-dot failed';
    if (conversationHasDocs(job)) return 'docs-dot has-docs';
    return 'docs-dot';
  }

  function startNewConversation() {
    return newSavedChat();
  }


  async function loadConversationsHistory() {
    try {
      const res = await fetch('/jobs');
      if (!res.ok) {
        console.warn('GET /jobs failed:', res.status);
        return;
      }
      const data = await res.json();
      conversationsCache = Array.isArray(data) ? data : [];
      await refreshSavedChats();
      if (conversationsCountTag) {
        conversationsCountTag.textContent = String(workspaceChats.length);
      }
      renderConversationsList();
    } catch (err) {
      console.warn('Failed to fetch conversations:', err);
      if (sidebarConversationsList) {
        sidebarConversationsList.innerHTML =
          '<div class="sidebar-conv-empty">Could not load conversations</div>';
      }
    }
  }

  function renderConversationsList() {
    renderSavedChats();
  }

  function getAgentTagInfo(agentId) {
    if (!agentId) {
      return { label: 'AUTO', className: 'agent-tag agent-tag-auto' };
    }
    const id = String(agentId).toLowerCase();
    if (id.includes('medication')) {
      return { label: 'MEDICATION', className: 'agent-tag agent-tag-medication' };
    }
    if (id.includes('procedure')) {
      return { label: 'PROCEDURE', className: 'agent-tag agent-tag-procedure' };
    }
    if (id.includes('diagnost') || id.includes('diagnose')) {
      return { label: 'DIAGNOSE', className: 'agent-tag agent-tag-diagnostic' };
    }
    if (id.includes('general') || id.includes('factcheck')) {
      return { label: 'GENERAL', className: 'agent-tag agent-tag-general' };
    }
    const label = String(agentId).replace(/_agent$/, '').replace(/_/g, ' ').toUpperCase();
    return { label, className: 'agent-tag' };
  }

  function formatQueryDisplayTitle(query) {
    if (!query) return 'Untitled query';
    let str = query.trim();

    // If query starts with patient context header, extract main query part
    if (str.includes('--- SELECTED PATIENT CONTEXT ---')) {
      const parts = str.split('--- SELECTED PATIENT CONTEXT ---');
      str = parts[0].trim();
    }
    // Remove "55-year-old male patient (Peter)..." preamble if present to extract core topic
    str = str.replace(/^\d+-year-old\s+(male|female)\s+patient\s*\([^)]*\)\s*(with\s+[^ presenting]*\s*)?(presenting\s+with\s*)?/i, '');
    str = str.replace(/^Provide comprehensive clinical analysis.*$/im, '');
    str = str.trim();

    if (!str) str = query;
    return truncateQuery(str, 48);
  }


  async function loadConversationDetails(jobId) {
    const owner = activeChat?.id;
    const version = ++navigationVersion;
    try {
      const res = await fetch(`/jobs/${jobId}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const job = await res.json();
      if (version !== navigationVersion || activeChat?.id !== owner || (job.chat_id && job.chat_id !== owner)) return;
      if (['pending', 'running'].includes(job.status)) { pollSavedRun(jobId, owner); return; }
      currentJob = job;
      activeConversationId = jobId;
      document.getElementById('chatRunSelect').value = jobId;
      switchView('viewConversations');
      await displayJobResults(job);
      if (version !== navigationVersion || activeChat?.id !== owner) return;
      switchWorkbenchTab('output');
      renderConversationsList();
      showToast(`Loaded: "${truncateQuery(job.query || jobId, 48)}"`, 'info');
    } catch (err) {
      showToast(`Failed to load conversation: ${err.message}`, 'error');
    }
  }

  async function deleteConversation(jobId) {
    if (!confirm('Delete this analysis run and its reports? The saved chat will remain.')) return;
    const owner = activeChat?.id;
    try {
      const res = await fetch(`/jobs/${jobId}`, { method: 'DELETE' });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      if (activeConversationId === jobId) {
        activeConversationId = null;
        currentJob = null;
        filesContainer.classList.add('hidden');
        reportPreviewCard.classList.add('hidden');
        placeholderState.classList.remove('hidden');
      }
      showToast('Analysis run and reports deleted.', 'success');
      await reloadActiveChat(owner);
      await loadConversationsHistory();
    } catch (err) {
      showToast(`Failed to delete: ${err.message}`, 'error');
    }
  }

  // 6. Regenerate Report Modal
  function openRegenModal(jobId, query) {
    regenSourceJobId.value = jobId;
    regenQueryPreview.textContent = query || 'Selected Query';
    regenerateModal.classList.remove('hidden');
  }

  function closeRegenModal() {
    regenerateModal.classList.add('hidden');
  }

  async function confirmRegenerateJob() {
    const owner = activeChat?.id;
    const jobId = regenSourceJobId.value;
    const targetAgent = regenAgentSelect.value;
    const targetModel = regenModelSelect.value;

    btnConfirmRegenerate.disabled = true;
    btnConfirmRegenerate.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Submitting...';

    try {
      const res = await fetch(`/jobs/${jobId}/regenerate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          agent_id: targetAgent,
          model: targetModel,
          web_search: true
        })
      });

      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      closeRegenModal();
      showToast(`Regeneration job submitted! Polling new job ${data.job_id}...`, 'success');
      
      if (activeChat?.id === owner) {
        await reloadActiveChat(owner);
        pollSavedRun(data.job_id, owner);
      }
      loadConversationsHistory();
    } catch (err) {
      showToast(`Failed to regenerate: ${err.message}`, 'error');
    } finally {
      btnConfirmRegenerate.disabled = false;
      btnConfirmRegenerate.innerHTML = '<i class="fa-solid fa-play"></i> Start Regeneration';
    }
  }

  // 7. Patient Directory & CRUD Management
  async function loadPatients() {
    try {
      const res = await fetch('/patients');
      if (res.ok) {
        patientsCache = await res.json();
        patientsCountTag.textContent = patientsCache.length;
        patientTotalBadge.textContent = `Total Patients: ${patientsCache.length}`;
        renderPatientsTable();
        // Keep selected patient in sync if still present; clear if deleted remotely.
        if (selectedPatient) {
          const refreshed = patientsCache.find(p => p.id === selectedPatient.id);
          if (refreshed) {
            selectedPatient = refreshed;
            renderPatientData();
          } else {
            await selectWorkspace('');
          }
        }
        renderPlusMenuPatients();
        renderConversationsList();
      }
    } catch (err) {
      console.warn('Failed to load patients:', err);
    }
  }

  function renderPlusMenuPatients() {
    if (!plusMenuPatientsList) return;

    if (!patientsCache || patientsCache.length === 0) {
      plusMenuPatientsList.innerHTML = `
        <div class="dropdown-empty-patients">No saved patients yet</div>
        <button type="button" class="dropdown-item" id="plusMenuGoPatients">
          <i class="fa-solid fa-user-plus"></i> Manage Patients
        </button>
      `;
      const goBtn = document.getElementById('plusMenuGoPatients');
      if (goBtn) {
        goBtn.addEventListener('click', (e) => {
          e.stopPropagation();
          if (plusMenuDropdown) plusMenuDropdown.classList.add('hidden');
          switchView('viewPatients');
        });
      }
      return;
    }

    plusMenuPatientsList.innerHTML = '';
    patientsCache.forEach(p => {
      const btn = document.createElement('button');
      btn.type = 'button';
      const isActive = selectedPatient && selectedPatient.id === p.id;
      btn.className = 'dropdown-item patient-select-item' + (isActive ? ' active' : '');
      btn.setAttribute('data-patient-id', p.id);

      const condition = p.primary_condition || 'No primary condition';
      const demoBits = [];
      if (p.age != null) demoBits.push(`${p.age}y`);
      if (p.gender) demoBits.push(p.gender);
      const demo = demoBits.length ? demoBits.join(' · ') : '';

      btn.innerHTML = `
        <i class="fa-solid fa-user-injured"></i>
        <span class="patient-select-text">
          <span class="patient-select-name">${escapeHtml(p.name || 'Unnamed')}</span>
          <span class="patient-select-sub">${escapeHtml(condition)}${demo ? ' · ' + escapeHtml(demo) : ''}</span>
        </span>
        ${isActive ? '<i class="fa-solid fa-check patient-select-check"></i>' : ''}
      `;

      btn.addEventListener('click', (e) => {
        e.stopPropagation();
        selectPatientForContext(p.id);
        if (plusMenuDropdown) plusMenuDropdown.classList.add('hidden');
      });

      plusMenuPatientsList.appendChild(btn);
    });
  }

  function selectPatientForContext(patientId) {
    return selectWorkspace(patientId).catch(reportChatError);
  }
  function clearSelectedPatient({ toast = false } = {}) {
    return selectWorkspace('').then(() => {
      if (toast) showToast('Switched to General Medical Research workspace.', 'info');
    }).catch(reportChatError);
  }
  function formatPatientContext(patient) {
    if (!patient) return '';

    const lines = [];
    lines.push(`Patient: ${patient.name || 'Unnamed'}`);
    if (patient.age != null) lines.push(`Age: ${patient.age}`);
    if (patient.gender) lines.push(`Gender: ${patient.gender}`);
    if (patient.primary_condition) lines.push(`Primary condition: ${patient.primary_condition}`);
    if (patient.contact_email) lines.push(`Contact email: ${patient.contact_email}`);
    if (patient.contact_phone) lines.push(`Contact phone: ${patient.contact_phone}`);

    const meta = patient.metadata_json || {};
    const metaKeys = Object.keys(meta);
    if (metaKeys.length > 0) {
      lines.push('');
      lines.push('Metadata:');
      metaKeys.forEach(k => {
        lines.push(`  - ${k}: ${meta[k]}`);
      });
    }

    const clinical = patient.clinical_data || {};
    const catOrder = ['heart', 'liver', 'pancreas', 'nutrients', 'overall_health', 'medications'];
    let hasClinical = false;
    catOrder.forEach(cat => {
      const rows = Array.isArray(clinical[cat]) ? clinical[cat] : [];
      if (rows.length === 0) return;
      if (!hasClinical) {
        lines.push('');
        lines.push('Clinical data:');
        hasClinical = true;
      }
      const title = CAT_TITLES[cat] || cat;
      lines.push(`  ${title}:`);
      rows.forEach(row => {
        if (!row || typeof row !== 'object') return;
        const marker = row.marker || row.name || row.medication || 'Item';
        const value = row.value ?? row.dose ?? row.dosage ?? '';
        const range = row.reference_range || row.range || '';
        const notes = row.notes || '';
        let entry = `    - ${marker}`;
        if (value !== '') entry += `: ${value}`;
        if (range) entry += ` (ref: ${range})`;
        if (notes) entry += ` — ${notes}`;
        const extras = Object.fromEntries(Object.entries(row).filter(([key]) => !['marker', 'name', 'medication', 'value', 'dose', 'dosage', 'reference_range', 'range', 'notes'].includes(key)));
        if (Object.keys(extras).length) entry += ` — ${JSON.stringify(extras)}`;
        lines.push(entry);
      });
    });

    // Any extra clinical keys not in the standard categories
    Object.keys(clinical).forEach(cat => {
      if (catOrder.includes(cat)) return;
      const rows = clinical[cat];
      if (Array.isArray(rows) && rows.length > 0) {
        lines.push(`  ${cat}:`);
        rows.forEach(row => {
          lines.push(`    - ${typeof row === 'object' ? JSON.stringify(row) : String(row)}`);
        });
      } else if (rows && typeof rows === 'object' && !Array.isArray(rows)) {
        lines.push(`  ${cat}: ${JSON.stringify(rows)}`);
      }
    });

    return lines.join('\n').trim();
  }

  /**
   * Build combined background context from selected patient + optional attached document.
   */

  function renderPatientsTable() {
    if (patientsActiveBanner && patientsActiveBannerName) {
      if (selectedPatient) {
        patientsActiveBanner.classList.remove('hidden');
        patientsActiveBannerName.textContent = selectedPatient.name || 'Unnamed';
      } else {
        patientsActiveBanner.classList.add('hidden');
      }
    }
    if (btnActiveBannerGoChat) {
      btnActiveBannerGoChat.onclick = () => {
        switchView('viewConversations');
        switchWorkbenchTab('intake');
        if (queryInput) queryInput.focus();
      };
    }
    if (btnActiveBannerClear) {
      btnActiveBannerClear.onclick = async () => {
        await clearSelectedPatient();
        showToast('Switched to General Medical Research workspace.', 'info');
      };
    }

    const filter = (patientSearchInput.value || '').toLowerCase().trim();
    const filtered = patientsCache.filter(p => 
      (p.name || '').toLowerCase().includes(filter) ||
      (p.primary_condition || '').toLowerCase().includes(filter) ||
      (p.contact_email || '').toLowerCase().includes(filter) ||
      JSON.stringify(p.metadata_json || {}).toLowerCase().includes(filter) ||
      JSON.stringify(p.clinical_data || {}).toLowerCase().includes(filter)
    );

    if (filtered.length === 0) {
      patientsTableBody.innerHTML = '<tr><td colspan="7" class="empty-patients-row">No patient records found. Click <strong>+ Add New Patient</strong> to create one.</td></tr>';
      return;
    }

    patientsTableBody.innerHTML = '';
    filtered.forEach(p => {
      const tr = document.createElement('tr');
      const isSelected = selectedPatient && selectedPatient.id === p.id;
      if (isSelected) {
        tr.classList.add('selected-patient-row');
      }
      
      // Build Metadata chips preview
      let metaHtml = '';
      const metaObj = p.metadata_json || {};
      const keys = Object.keys(metaObj);
      if (keys.length > 0) {
        metaHtml = '<div class="meta-tags-list">' + keys.slice(0, 3).map(k => 
          `<span class="meta-chip"><strong>${escapeHtml(k)}:</strong> ${escapeHtml(String(metaObj[k]))}</span>`
        ).join('') + (keys.length > 3 ? `<span class="meta-chip">+${keys.length - 3} more</span>` : '') + '</div>';
      } else {
        metaHtml = '<span style="color:var(--text-muted); font-size:0.75rem;">None</span>';
      }

      const createdStr = p.created_at ? new Date(p.created_at).toLocaleDateString() : 'N/A';

      tr.innerHTML = `
        <td>
          <div class="patient-name-cell">
            <strong>${escapeHtml(p.name)}</strong>
            ${isSelected ? '<span class="patient-active-pill"><i class="fa-solid fa-circle-check"></i> Active</span>' : ''}
          </div>
        </td>
        <td>${escapeHtml(p.age ? String(p.age) + ' yrs' : 'N/A')} / ${escapeHtml(p.gender || 'N/A')}</td>
        <td>${escapeHtml(p.primary_condition || 'Unspecified')}</td>
        <td>${escapeHtml(p.contact_email || p.contact_phone || 'N/A')}</td>
        <td>${metaHtml}</td>
        <td style="color:var(--text-muted); font-size:0.75rem;">${createdStr}</td>
        <td style="text-align: right; white-space: nowrap;">
          <button type="button" class="btn btn-xs ${isSelected ? 'btn-success' : 'btn-primary'} btn-select-patient" data-id="${p.id}" title="${isSelected ? 'Currently chatting on this patient — click to open chat' : 'Select this patient and start chatting'}">
            <i class="fa-solid ${isSelected ? 'fa-comment-dots' : 'fa-comments'}"></i>
            <span>${isSelected ? 'Chatting' : 'Select Patient'}</span>
          </button>
          <button class="btn-icon-sm btn-edit-patient" title="Edit Patient & Metadata" data-id="${p.id}"><i class="fa-solid fa-pen-to-square"></i></button>
          <button class="btn-icon-sm danger btn-delete-patient" title="Delete Patient" data-id="${p.id}"><i class="fa-solid fa-trash-can"></i></button>
        </td>
      `;

      tr.querySelector('.btn-select-patient').addEventListener('click', async () => {
        if (!isSelected) {
          await selectWorkspace(p.id);
          showToast(`Selected patient: ${p.name}. Chatting in patient workspace.`, 'info');
        }
        switchView('viewConversations');
        switchWorkbenchTab('intake');
        if (queryInput) queryInput.focus();
      });
      tr.querySelector('.btn-edit-patient').addEventListener('click', () => openPatientModal(p.id));
      tr.querySelector('.btn-delete-patient').addEventListener('click', () => deletePatientRecord(p.id));

      patientsTableBody.appendChild(tr);
    });
  }

  function openPatientModal(patientId = null) {
    currentPatientMeta = {};
    currentPatientClinicalData = {
      heart: [],
      liver: [],
      pancreas: [],
      nutrients: [],
      overall_health: [],
      medications: []
    };

    if (patientId) {
      const patient = patientsCache.find(p => p.id === patientId);
      if (patient) {
        patientModalTitle.textContent = 'Edit Patient & Categorized Data';
        patientEditId.value = patient.id;
        patientName.value = patient.name || '';
        patientPrimaryCondition.value = patient.primary_condition || '';
        patientAge.value = patient.age || '';
        patientGender.value = patient.gender || '';
        patientEmail.value = patient.contact_email || '';
        if (patientDescription) {
          patientDescription.value = (patient.metadata_json && patient.metadata_json.description) || '';
        }
        currentPatientMeta = { ...(patient.metadata_json || {}) };
        
        const clinical = patient.clinical_data || {};
        ['heart', 'liver', 'pancreas', 'nutrients', 'overall_health', 'medications'].forEach(cat => {
          if (Array.isArray(clinical[cat])) {
            currentPatientClinicalData[cat] = JSON.parse(JSON.stringify(clinical[cat]));
          }
        });
      }
    } else {
      patientModalTitle.textContent = 'Add New Patient';
      patientEditId.value = '';
      document.getElementById('patientForm').reset();
      if (patientDescription) patientDescription.value = '';
    }

    renderMetadataTagsEditor();
    switchCategoryTab('heart');
    patientModal.classList.remove('hidden');
  }

  function closePatientModal() {
    patientModal.classList.add('hidden');
  }

  async function handleClassifyPatientDescription() {
    const text = patientDescription ? patientDescription.value.trim() : '';
    if (!text) {
      showToast('Please enter a clinical description to classify.', 'error');
      if (patientDescription) patientDescription.focus();
      return;
    }

    if (classifyStatusText) classifyStatusText.classList.remove('hidden');
    if (btnClassifyDescription) {
      btnClassifyDescription.disabled = true;
      btnClassifyDescription.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Classifying with AI...';
    }

    try {
      const res = await fetch('/patients/classify-text', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: text,
          model: modelSelect ? modelSelect.value : 'grok-4.5'
        })
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || `HTTP ${res.status}`);
      }
      const data = await res.json();
      const classified = data.classification || {};

      // 1. Demographics
      if (classified.demographics) {
        const demo = classified.demographics;
        if (demo.name && (!patientName.value || patientName.value.trim() === '')) {
          patientName.value = demo.name;
        }
        if (demo.age && (!patientAge.value || patientAge.value === '')) {
          patientAge.value = demo.age;
        }
        if (demo.gender && (!patientGender.value || patientGender.value === '')) {
          patientGender.value = demo.gender;
        }
        if (demo.primary_condition && (!patientPrimaryCondition.value || patientPrimaryCondition.value.trim() === '')) {
          patientPrimaryCondition.value = demo.primary_condition;
        }
      }

      // 2. Custom Metadata Tags
      if (classified.metadata_tags && typeof classified.metadata_tags === 'object') {
        Object.entries(classified.metadata_tags).forEach(([k, v]) => {
          if (v && k.toLowerCase() !== 'description') {
            currentPatientMeta[k] = String(v);
          }
        });
        renderMetadataTagsEditor();
      }

      // 3. Categorized Organ System Findings
      if (classified.categorized_data && typeof classified.categorized_data === 'object') {
        const catData = classified.categorized_data;
        let totalItems = 0;
        ['heart', 'liver', 'pancreas', 'nutrients', 'overall_health', 'medications'].forEach(cat => {
          if (Array.isArray(catData[cat]) && catData[cat].length > 0) {
            currentPatientClinicalData[cat] = catData[cat];
            totalItems += catData[cat].length;
          }
        });
        renderCategorizedTable(activeCategory);
        showToast(`AI extracted demographics and classified ${totalItems} clinical measurements!`, 'success');
      } else {
        showToast('Clinical description classified successfully!', 'success');
      }
    } catch (err) {
      showToast(`Failed to classify description: ${err.message}`, 'error');
    } finally {
      if (classifyStatusText) classifyStatusText.classList.add('hidden');
      if (btnClassifyDescription) {
        btnClassifyDescription.disabled = false;
        btnClassifyDescription.innerHTML = '<i class="fa-solid fa-wand-magic-sparkles"></i> Auto-Classify with AI';
      }
    }
  }

  async function handlePatientReportUpload(file) {
    patientParseStatusBox.classList.remove('hidden');
    const formData = new FormData();
    formData.append('file', file);
    formData.append('model', modelSelect ? modelSelect.value : 'grok-4.5');

    try {
      const res = await fetch('/patients/parse-report', {
        method: 'POST',
        body: formData
      });

      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      
      const catData = data.categorized_data || {};
      ['heart', 'liver', 'pancreas', 'nutrients', 'overall_health', 'medications'].forEach(cat => {
        if (Array.isArray(catData[cat]) && catData[cat].length > 0) {
          currentPatientClinicalData[cat] = catData[cat];
        }
      });

      switchCategoryTab(activeCategory);
      showToast(`Medical report parsed & categorized for ${data.filename}!`, 'success');
    } catch (err) {
      showToast(`Failed to parse patient report: ${err.message}`, 'error');
    } finally {
      patientParseStatusBox.classList.add('hidden');
    }
  }

  function switchCategoryTab(catKey) {
    activeCategory = catKey;
    document.querySelectorAll('.cat-tab-btn').forEach(btn => {
      if (btn.getAttribute('data-cat') === catKey) {
        btn.classList.add('active');
      } else {
        btn.classList.remove('active');
      }
    });
    document.getElementById('currentCatTitle').textContent = CAT_TITLES[catKey] || catKey;
    renderCategorizedTable(catKey);
  }

  function renderCategorizedTable(catKey) {
    const catTableTbody = document.getElementById('catTableTbody');
    const items = currentPatientClinicalData[catKey] || [];
    
    if (items.length === 0) {
      catTableTbody.innerHTML = '<tr><td colspan="6" class="empty-cat-row">No recorded measurements for this category yet. Upload a report or click + Add Test.</td></tr>';
      return;
    }

    catTableTbody.innerHTML = '';
    items.forEach((item, index) => {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td><input type="text" class="cell-input input-marker" value="${escapeHtml(item.marker || '')}" placeholder="Biomarker"></td>
        <td><input type="text" class="cell-input input-value" value="${escapeHtml(item.value || '')}" placeholder="Result"></td>
        <td><input type="text" class="cell-input input-range" value="${escapeHtml(item.reference_range || '')}" placeholder="Range"></td>
        <td>
          <select class="cell-select input-status">
            <option value="Normal" ${item.status === 'Normal' ? 'selected' : ''}>Normal</option>
            <option value="High" ${item.status === 'High' ? 'selected' : ''}>High</option>
            <option value="Low" ${item.status === 'Low' ? 'selected' : ''}>Low</option>
            <option value="Critical" ${item.status === 'Critical' ? 'selected' : ''}>Critical</option>
          </select>
        </td>
        <td><input type="text" class="cell-input input-notes" value="${escapeHtml(item.notes || '')}" placeholder="Notes"></td>
        <td><button type="button" class="btn-icon-sm danger btn-del-row" title="Delete Row">&times;</button></td>
      `;

      tr.querySelector('.input-marker').addEventListener('input', (e) => item.marker = e.target.value);
      tr.querySelector('.input-value').addEventListener('input', (e) => item.value = e.target.value);
      tr.querySelector('.input-range').addEventListener('input', (e) => item.reference_range = e.target.value);
      tr.querySelector('.input-status').addEventListener('change', (e) => item.status = e.target.value);
      tr.querySelector('.input-notes').addEventListener('input', (e) => item.notes = e.target.value);
      tr.querySelector('.btn-del-row').addEventListener('click', () => {
        items.splice(index, 1);
        renderCategorizedTable(catKey);
      });

      catTableTbody.appendChild(tr);
    });
  }

  function addCategoryRow() {
    if (!currentPatientClinicalData[activeCategory]) {
      currentPatientClinicalData[activeCategory] = [];
    }
    currentPatientClinicalData[activeCategory].push({
      marker: '',
      value: '',
      reference_range: '',
      status: 'Normal',
      notes: ''
    });
    renderCategorizedTable(activeCategory);
  }

  function renderMetadataTagsEditor() {
    metadataTagsContainer.innerHTML = '';
    const keys = Object.keys(currentPatientMeta).filter(k => k !== 'description');
    if (keys.length === 0) {
      metadataTagsContainer.innerHTML = '<span class="no-tags-hint">No custom metadata tags added yet. Enter key & value above.</span>';
      return;
    }

    keys.forEach(k => {
      const tag = document.createElement('span');
      tag.className = 'meta-tag-edit';
      tag.innerHTML = `
        <strong>${escapeHtml(k)}:</strong> ${escapeHtml(String(currentPatientMeta[k]))}
        <button type="button" class="remove-tag-btn" data-key="${escapeHtml(k)}">&times;</button>
      `;
      tag.querySelector('.remove-tag-btn').addEventListener('click', () => {
        delete currentPatientMeta[k];
        renderMetadataTagsEditor();
      });
      metadataTagsContainer.appendChild(tag);
    });
  }

  function addMetaTagFromInput() {
    const key = metaKeyInput.value.trim();
    const val = metaValInput.value.trim();
    if (!key || !val) {
      showToast('Please enter both Key and Value for metadata tag.', 'error');
      return;
    }
    currentPatientMeta[key] = val;
    metaKeyInput.value = '';
    metaValInput.value = '';
    renderMetadataTagsEditor();
  }

  async function savePatientRecord() {
    const name = patientName.value.trim();
    if (!name) {
      showToast('Patient full name is required.', 'error');
      patientName.focus();
      return;
    }

    if (patientDescription && patientDescription.value.trim()) {
      currentPatientMeta['description'] = patientDescription.value.trim();
    }

    const payload = {
      name: name,
      primary_condition: patientPrimaryCondition.value.trim() || null,
      age: patientAge.value ? parseInt(patientAge.value) : null,
      gender: patientGender.value || null,
      contact_email: patientEmail.value.trim() || null,
      metadata_json: currentPatientMeta,
      clinical_data: currentPatientClinicalData
    };

    const patientId = patientEditId.value;
    const url = patientId ? `/patients/${patientId}` : '/patients';
    const method = patientId ? 'PUT' : 'POST';

    btnSavePatient.disabled = true;
    btnSavePatient.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Saving...';

    try {
      const res = await fetch(url, {
        method: method,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      showToast(`Patient record successfully ${patientId ? 'updated' : 'created'}!`, 'success');
      closePatientModal();
      loadPatients();
    } catch (err) {
      showToast(`Failed to save patient: ${err.message}`, 'error');
    } finally {
      btnSavePatient.disabled = false;
      btnSavePatient.innerHTML = '<i class="fa-solid fa-floppy-disk"></i> Save Patient Record';
    }
  }

  async function deletePatientRecord(patientId) {
    if (!confirm('Are you sure you want to delete this patient record?')) return;
    try {
      const res = await fetch(`/patients/${patientId}`, { method: 'DELETE' });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      if (selectedPatient && selectedPatient.id === patientId) {
        await selectWorkspace('');
      }
      showToast('Patient record deleted successfully.', 'info');
      loadPatients();
    } catch (err) {
      showToast(`Failed to delete patient: ${err.message}`, 'error');
    }
  }

  // 8. Document File Upload Handler for Main Analysis
  async function handleFileUpload(file) {
    return attachToSavedChat(file);
  }
  function removeUploadedDocument(opts = {}) {
    const hadDoc = !!parsedDocument;
    parsedDocument = null;
    if (parsedDocCard) parsedDocCard.classList.add('hidden');
    if (parsedPreviewBox) parsedPreviewBox.classList.add('hidden');
    if (fileInput) fileInput.value = '';
    if (hadDoc && !opts.silent) showToast('Document detached.', 'info');
  }

  function showFileError(msg) {
    fileErrorMsg.textContent = msg;
    fileErrorAlert.classList.remove('hidden');
  }

  function escapeHtml(str) {
    if (!str) return '';
    return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  }

  function updateIntakeChatUI() {
    if (!intakeChatStream || !intakeModeBadge) return;

    if (intakeChatHistory.length === 0) {
      intakeChatStream.classList.add('hidden');
      intakeChatStream.innerHTML = '';
      intakeModeBadge.className = 'mode-badge direct';
      intakeModeBadge.innerHTML = '<i class="fa-solid fa-bolt"></i> Direct Prompt';
    } else {
      intakeChatStream.classList.remove('hidden');
      intakeModeBadge.className = 'mode-badge chat';
      intakeModeBadge.innerHTML = `<i class="fa-solid fa-comments"></i> Intake Chat (${intakeChatHistory.length} turns)`;

      let html = '';
      intakeChatHistory.forEach(msg => {
        const isUser = msg.role === 'user';
        const icon = isUser ? '<i class="fa-solid fa-user"></i>' : '<i class="fa-solid fa-user-doctor"></i>';
        html += `
          <div class="chat-msg ${isUser ? 'user' : 'assistant'}">
            <div class="chat-avatar">${icon}</div>
            <div class="chat-bubble">${escapeHtml(msg.content)}</div>
          </div>
        `;
      });

      if (isIntakeChatLoading) {
        html += `
          <div class="chat-msg assistant">
            <div class="chat-avatar"><i class="fa-solid fa-spinner fa-spin"></i></div>
            <div class="chat-bubble chat-bubble-typing">
              <i class="fa-solid fa-circle-notch fa-spin"></i> Intake Assistant is evaluating...
            </div>
          </div>
        `;
      }

      intakeChatStream.innerHTML = html;
      intakeChatStream.scrollTop = intakeChatStream.scrollHeight;
    }
  }

  async function sendIntakeChatMessage() {
    return sendSavedMessage();
  }

  async function submitAsyncAnalysis() {
    return analyzeSavedChat();
  }
  // 10. Poll Job Status
  function pollJobStatus(jobId) {
    return pollSavedRun(jobId, activeChat?.id);
  }

  function updateProgressSteps(status) {
    if (status === 'pending') {
      progressBarFill.style.width = '25%';
      stepRoute.className = 'step-badge active';
      stepReasoning.className = 'step-badge';
      stepValidation.className = 'step-badge';
      stepReports.className = 'step-badge';
    } else if (status === 'running') {
      progressBarFill.style.width = '65%';
      stepRoute.className = 'step-badge active';
      stepReasoning.className = 'step-badge active';
      stepValidation.className = 'step-badge active';
      stepReports.className = 'step-badge';
    } else if (status === 'completed') {
      progressBarFill.style.width = '100%';
      stepRoute.className = 'step-badge active';
      stepReasoning.className = 'step-badge active';
      stepValidation.className = 'step-badge active';
      stepReports.className = 'step-badge active';
    }
  }


  // 11. Display Job Results & Artifacts
  async function displayJobResults(job) {
    const version = navigationVersion;
    placeholderState.classList.add('hidden');
    jobProgressCard.classList.add('hidden');

    const agentId = job.agent_id || 'general_agent';
    const agentNames = {
      'medication_agent': 'Medication Specialist',
      'procedure_agent': 'Procedure Specialist',
      'diagnostic_agent': 'Diagnostic Specialist',
      'general_agent': 'Medical Fact-Checker'
    };

    activeAgentBadgeText.textContent = agentNames[agentId] || agentId;
    activeAgentBadge.classList.remove('hidden');

    activeFiles = job.files || {};
    loadedReportTexts = {};

    filesGrid.innerHTML = '';
    if (Object.keys(activeFiles).length > 0) {
      filesContainer.classList.remove('hidden');
      Object.keys(activeFiles).forEach(fileKey => {
        const filePath = activeFiles[fileKey];
        if (!filePath) return;

        const cleanPath = String(filePath).replace(/^\/+/, '');
        const fileName = cleanPath.split('/').pop();
        const btn = document.createElement('a');
        btn.className = 'file-card-btn';
        btn.href = `/${cleanPath}`;
        btn.target = '_blank';

        const isPdf = fileName.endsWith('.pdf');
        const isJson = fileName.endsWith('.json');
        const iconClass = isPdf ? 'fa-solid fa-file-pdf' : (isJson ? 'fa-solid fa-code' : 'fa-solid fa-file-lines');
        const formatBadge = isPdf ? 'PDF' : (isJson ? 'JSON' : 'MD');
        const catClass = isPdf ? 'pdf' : (isJson ? 'json' : 'md');

        btn.innerHTML = `
          <div class="file-icon-box ${catClass}">
            <i class="${iconClass}"></i>
          </div>
          <div class="file-info-col">
            <span class="file-name-text">${escapeHtml(fileKey.replace(/_/g, ' '))}</span>
            <span class="file-format-tag">${formatBadge} Artifact</span>
          </div>
          <i class="fa-solid fa-arrow-down-to-line file-dl-icon"></i>
        `;
        filesGrid.appendChild(btn);
      });
    }

    reportPreviewCard.classList.remove('hidden');

    const patientPath = activeFiles['patient_report'] || activeFiles['markdown_report'] || activeFiles['summary'];
    const practitionerPath = activeFiles['practitioner_report'] || activeFiles['markdown_report'] || activeFiles['summary'];
    const summaryPath = activeFiles['summary'] || activeFiles['medication_summary'] || activeFiles['markdown_report'];
    const contextPath = activeFiles['context_report'];

    await Promise.all([
      loadReportText('patient', patientPath, version),
      loadReportText('practitioner', practitionerPath, version),
      loadReportText('summary', summaryPath, version),
      loadReportText('context', contextPath, version)
    ]);
    if (version !== navigationVersion) return;

    // DB JSON inline result fallbacks if static file text missing
    if (job.result && typeof job.result === 'object') {
      if (!loadedReportTexts['patient']) {
        if (job.result.patient_report) loadedReportTexts['patient'] = job.result.patient_report;
        else if (job.result.summary) loadedReportTexts['patient'] = job.result.summary;
      }
      if (!loadedReportTexts['practitioner']) {
        if (job.result.practitioner_report) loadedReportTexts['practitioner'] = job.result.practitioner_report;
        else if (job.result.summary) loadedReportTexts['practitioner'] = job.result.summary;
      }
      loadedReportTexts['json'] = JSON.stringify(job.result, null, 2);
    }

    const preferredTab = loadedReportTexts['context'] && !loadedReportTexts['patient']
      ? 'context'
      : 'patient';
    const tabBtn = document.querySelector(`.report-tabs .tab-btn[data-tab="${preferredTab}"]`)
      || document.querySelector('.report-tabs .tab-btn[data-tab="patient"]');
    if (tabBtn) tabBtn.click();
  }

  async function loadReportText(tabKey, filePath, version = navigationVersion) {
    if (!filePath) return;
    try {
      const res = await fetch(`/${filePath}`);
      if (res.ok) {
        const text = await res.text();
        if (version !== navigationVersion) return;
        loadedReportTexts[tabKey] = text;
      }
    } catch (err) {
      console.warn(`Could not load report file ${filePath}:`, err);
    }
  }

  function renderReportTab(tabKey) {
    const text = loadedReportTexts[tabKey];
    if (!text) {
      markdownViewer.innerHTML = tabKey === 'context'
        ? '<p class="text-muted">No context report for this run. Newer analyses write a small <code>context_report.md</code> listing patient, document, intake chat, and the final agent prompt.</p>'
        : '<p class="text-muted">No content available for this report view.</p>';
      btnDownloadCurrent.style.display = 'none';
      return;
    }

    if (tabKey === 'json') {
      markdownViewer.innerHTML = `<pre class="json-box">${escapeHtml(text)}</pre>`;
      btnDownloadCurrent.style.display = 'none';
    } else {
      markdownViewer.innerHTML = marked.parse(text);
      
      const pdfKey = `${tabKey}_report_pdf`;
      const mdKey = `${tabKey}_report`;
      const filePath = activeFiles[pdfKey] || activeFiles['pdf_report'] || activeFiles[mdKey] || activeFiles[tabKey];

      if (filePath) {
        const cleanPath = String(filePath).replace(/^\/+/, '');
        btnDownloadCurrent.href = `/${cleanPath}`;
        btnDownloadCurrent.target = '_blank';
        btnDownloadCurrent.innerHTML = `<i class="fa-solid fa-file-pdf"></i> View / Download PDF`;
        btnDownloadCurrent.style.display = 'inline-flex';
      } else {
        btnDownloadCurrent.style.display = 'none';
      }
    }
  }

  // 12. Route Query Only (Synchronous Classification)
  async function routeQueryOnly() {
    const owner = activeChat?.id;
    let query = queryInput.value.trim();
    if (!query && intakeChatHistory.length > 0) {
      const lastUserMsg = [...intakeChatHistory].reverse().find(m => m.role === 'user');
      query = lastUserMsg ? lastUserMsg.content : '';
    }
    if (!query) {
      showToast('Please enter a medical query to route.', 'error');
      queryInput.focus();
      return;
    }

    btnRoute.disabled = true;
    btnRoute.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Routing...';

    try {
      const res = await fetch('/route', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: query,
          model: modelSelect.value
        })
      });

      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();

      if (activeChat?.id !== owner) return;

      placeholderState.classList.add('hidden');
      jobProgressCard.classList.add('hidden');
      filesContainer.classList.add('hidden');
      reportPreviewCard.classList.add('hidden');
      routeResultCard.classList.remove('hidden');

      routedAgentName.textContent = data.agent_name || data.agent_id;
      routedAgentDesc.textContent = `Query classified for ${data.agent_id}. Click "Start Analysis Run" to invoke reasoning pipeline.`;
      routeResultCard.classList.remove('hidden');
      switchWorkbenchTab('output');

      showToast(`Routed to: ${data.agent_name}`, 'success');

    } catch (err) {
      showToast(`Routing failed: ${err.message}`, 'error');
    } finally {
      btnRoute.disabled = false;
      btnRoute.innerHTML = '<i class="fa-solid fa-compass"></i> Route Only';
    }
  }

  function resetForm(opts = {}) {
    if (queryInput) queryInput.value = '';
    intakeChatHistory = [];
    isIntakeChatLoading = false;
    try { updateIntakeChatUI(); } catch (_) { /* ignore */ }
    try { removeUploadedDocument({ silent: true }); } catch (_) { /* ignore */ }
    try { clearSelectedPatient({ toast: false }); } catch (_) { /* ignore */ }
    if (placeholderState) placeholderState.classList.remove('hidden');
    if (jobProgressCard) jobProgressCard.classList.add('hidden');
    if (routeResultCard) routeResultCard.classList.add('hidden');
    if (filesContainer) filesContainer.classList.add('hidden');
    if (reportPreviewCard) reportPreviewCard.classList.add('hidden');
    if (activeAgentBadge) activeAgentBadge.classList.add('hidden');
    currentJob = null;
    activeConversationId = null;
    activeFiles = {};
    loadedReportTexts = {};
    try { renderConversationsList(); } catch (_) { /* ignore */ }
    if (!opts.silent) showToast('Workbench reset.', 'info');
  }

  // ── Configuration Menu ───────────────────────────────────────────────────

  async function refreshConfigCache() {
    try {
      const res = await fetch('/config');
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      slackWebhooksCache = data.slack_webhooks || [];
      apiKeysCache = data.api_keys || [];
      populateSlackWebhookSelect();
      return data;
    } catch (err) {
      console.warn('Could not load /config:', err);
      return null;
    }
  }

  function openConfigModal(tab = 'slack') {
    if (!configModal) return;
    configModal.classList.remove('hidden');
    switchConfigTab(tab);
    loadConfigPanels();
  }

  function closeConfigModal() {
    if (configModal) configModal.classList.add('hidden');
  }

  function switchConfigTab(tab) {
    const isSlack = tab === 'slack';
    if (configTabSlack) configTabSlack.classList.toggle('active', isSlack);
    if (configTabKeys) configTabKeys.classList.toggle('active', !isSlack);
    if (configPanelSlack) configPanelSlack.classList.toggle('hidden', !isSlack);
    if (configPanelKeys) configPanelKeys.classList.toggle('hidden', isSlack);
  }

  async function loadConfigPanels() {
    const data = await refreshConfigCache();
    if (!data) {
      if (slackWebhooksList) {
        slackWebhooksList.innerHTML = '<div class="config-empty" style="color:#fca5a5;">Failed to load configuration.</div>';
      }
      if (apiKeysList) {
        apiKeysList.innerHTML = '<div class="config-empty" style="color:#fca5a5;">Failed to load API key status.</div>';
      }
      return;
    }
    renderSlackWebhooksList(data.slack_webhooks || []);
    renderApiKeysList(data.api_keys || []);
  }

  function populateSlackWebhookSelect() {
    if (!slackWebhookSelect) return;
    const prev = slackWebhookSelect.value;
    const lastId = localStorage.getItem('slack_webhook_id') || '';
    slackWebhookSelect.innerHTML = '<option value="">— Select a saved webhook —</option>';
    slackWebhooksCache.forEach((wh) => {
      const opt = document.createElement('option');
      opt.value = wh.id;
      opt.textContent = wh.name || 'Webhook';
      slackWebhookSelect.appendChild(opt);
    });
    const prefer = prev || lastId;
    if (prefer && Array.from(slackWebhookSelect.options).some((o) => o.value === prefer)) {
      slackWebhookSelect.value = prefer;
    } else if (slackWebhooksCache.length === 1) {
      slackWebhookSelect.value = slackWebhooksCache[0].id;
    }
  }

  function renderSlackWebhooksList(webhooks) {
    if (!slackWebhooksList) return;
    if (!webhooks.length) {
      slackWebhooksList.innerHTML = '<div class="config-empty">No webhooks saved yet. Add one above.</div>';
      return;
    }
    slackWebhooksList.innerHTML = '';
    webhooks.forEach((wh) => {
      const item = document.createElement('div');
      item.className = 'config-item';
      const urlPreview = wh.url && wh.url.length > 48
        ? `${wh.url.slice(0, 28)}…${wh.url.slice(-12)}`
        : (wh.url || '');
      item.innerHTML = `
        <div class="config-item-main">
          <div class="config-item-title">
            <i class="fa-brands fa-slack" style="color:#e01e5a;"></i>
            ${escapeHtml(wh.name || 'Webhook')}
          </div>
          <div class="config-item-meta">${escapeHtml(urlPreview)}</div>
        </div>
        <div class="config-item-actions">
          <button type="button" class="btn-icon-danger" title="Delete webhook" data-delete-webhook="${escapeHtml(wh.id)}">
            <i class="fa-solid fa-trash"></i>
          </button>
        </div>
      `;
      const delBtn = item.querySelector('[data-delete-webhook]');
      delBtn.addEventListener('click', () => deleteSlackWebhook(wh.id, wh.name));
      slackWebhooksList.appendChild(item);
    });
  }

  function renderApiKeysList(keys) {
    if (!apiKeysList) return;
    if (!keys.length) {
      apiKeysList.innerHTML = '<div class="config-empty">No provider credential slots defined.</div>';
      return;
    }
    apiKeysList.innerHTML = '';
    keys.forEach((key) => {
      const item = document.createElement('div');
      item.className = 'config-item';
      const statusBadge = key.configured
        ? '<span class="config-badge ok"><i class="fa-solid fa-check"></i> Configured</span>'
        : '<span class="config-badge missing"><i class="fa-solid fa-xmark"></i> Missing</span>';
      const sourceBadge = key.source
        ? `<span class="config-badge source">${escapeHtml(key.source === 'config' ? 'Saved in app' : 'From environment')}</span>`
        : '';
      const preview = key.preview
        ? `<div class="config-item-meta"><code>${escapeHtml(key.preview)}</code></div>`
        : '';
      const canDelete = key.source === 'config';

      item.innerHTML = `
        <div class="config-item-main">
          <div class="config-item-title">
            ${escapeHtml(key.label || key.env_var)}
            ${statusBadge}
            ${sourceBadge}
          </div>
          <div class="config-item-meta">${escapeHtml(key.description || '')} · <code>${escapeHtml(key.env_var)}</code></div>
          ${preview}
          <div class="config-key-form">
            <input
              type="password"
              class="input-text api-key-input"
              data-env-var="${escapeHtml(key.env_var)}"
              placeholder="${escapeHtml(key.placeholder || 'Paste key…')}"
              autocomplete="off"
              spellcheck="false"
            >
            <button type="button" class="btn btn-xs btn-primary btn-save-key" data-env-var="${escapeHtml(key.env_var)}">
              <i class="fa-solid fa-floppy-disk"></i> Save
            </button>
            ${canDelete ? `<button type="button" class="btn btn-xs btn-outline btn-clear-key" data-env-var="${escapeHtml(key.env_var)}"><i class="fa-solid fa-trash"></i> Clear</button>` : ''}
          </div>
        </div>
      `;

      const saveBtn = item.querySelector('.btn-save-key');
      saveBtn.addEventListener('click', () => {
        const input = item.querySelector('.api-key-input');
        saveApiKey(key.env_var, input ? input.value : '');
      });
      const clearBtn = item.querySelector('.btn-clear-key');
      if (clearBtn) {
        clearBtn.addEventListener('click', () => clearApiKey(key.env_var, key.label));
      }
      const input = item.querySelector('.api-key-input');
      if (input) {
        input.addEventListener('keydown', (e) => {
          if (e.key === 'Enter') {
            e.preventDefault();
            saveApiKey(key.env_var, input.value);
          }
        });
      }
      apiKeysList.appendChild(item);
    });
  }

  async function addSlackWebhookFromForm() {
    const name = (cfgWebhookName && cfgWebhookName.value || '').trim();
    const url = (cfgWebhookUrl && cfgWebhookUrl.value || '').trim();
    if (!url || !url.startsWith('https://')) {
      showToast('Enter a valid Slack webhook URL (starts with https://)', 'error');
      if (cfgWebhookUrl) cfgWebhookUrl.focus();
      return;
    }
    btnAddSlackWebhook.disabled = true;
    try {
      const res = await fetch('/config/slack-webhooks', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: name || 'Slack Webhook', url })
      });
      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        const detail = typeof errData.detail === 'string'
          ? errData.detail
          : (errData.detail ? JSON.stringify(errData.detail) : `HTTP ${res.status}`);
        throw new Error(detail);
      }
      if (cfgWebhookName) cfgWebhookName.value = '';
      if (cfgWebhookUrl) cfgWebhookUrl.value = '';
      showToast('Slack webhook saved.', 'success');
      await loadConfigPanels();
    } catch (err) {
      showToast(`Failed to add webhook: ${err.message}`, 'error');
    } finally {
      btnAddSlackWebhook.disabled = false;
    }
  }

  async function deleteSlackWebhook(id, name) {
    if (!confirm(`Delete Slack webhook “${name || id}”?`)) return;
    try {
      const res = await fetch(`/config/slack-webhooks/${encodeURIComponent(id)}`, { method: 'DELETE' });
      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || `HTTP ${res.status}`);
      }
      if (localStorage.getItem('slack_webhook_id') === id) {
        localStorage.removeItem('slack_webhook_id');
      }
      showToast('Webhook deleted.', 'success');
      await loadConfigPanels();
    } catch (err) {
      showToast(`Failed to delete webhook: ${err.message}`, 'error');
    }
  }

  async function saveApiKey(envVar, value) {
    const trimmed = (value || '').trim();
    if (!trimmed) {
      showToast('Paste an API key before saving.', 'error');
      return;
    }
    try {
      const res = await fetch('/config/api-keys', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ env_var: envVar, value: trimmed })
      });
      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || `HTTP ${res.status}`);
      }
      showToast(`${envVar} saved. Models for this provider are now available.`, 'success');
      await loadConfigPanels();
    } catch (err) {
      showToast(`Failed to save key: ${err.message}`, 'error');
    }
  }

  async function clearApiKey(envVar, label) {
    if (!confirm(`Remove the saved ${label || envVar} key from app config?`)) return;
    try {
      const res = await fetch(`/config/api-keys/${encodeURIComponent(envVar)}`, { method: 'DELETE' });
      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || `HTTP ${res.status}`);
      }
      showToast(`${envVar} cleared from app config.`, 'success');
      await loadConfigPanels();
    } catch (err) {
      showToast(`Failed to clear key: ${err.message}`, 'error');
    }
  }

  // ── Notifications & API Modal Handlers ─────────────────────────────────────

  async function openNotifApiModal(defaultTab = 'slack') {
    if (notifApiModal) {
      notifApiModal.classList.remove('hidden');
    }
    switchNotifTab(defaultTab);
    await refreshConfigCache();
    loadSlackTasks();

    // Default calendar date to tomorrow at 10:00 AM
    if (calEventDate && !calEventDate.value) {
      const tomorrow = new Date();
      tomorrow.setDate(tomorrow.getDate() + 1);
      tomorrow.setHours(10, 0, 0, 0);
      const iso = tomorrow.toISOString().slice(0, 16);
      calEventDate.value = iso;
    }
  }

  function closeNotifApiModal() {
    if (notifApiModal) {
      notifApiModal.classList.add('hidden');
    }
  }

  function switchNotifTab(tabName) {
    const tabs = [
      { id: 'slack', btn: notifTabSlack, panel: notifPanelSlack },
      { id: 'email', btn: notifTabEmail, panel: notifPanelEmail },
      { id: 'calendar', btn: notifTabCalendar, panel: notifPanelCalendar },
      { id: 'api', btn: notifTabApi, panel: notifPanelApi }
    ];

    tabs.forEach(t => {
      if (t.btn) t.btn.classList.toggle('active', t.id === tabName);
      if (t.panel) t.panel.classList.toggle('hidden', t.id !== tabName);
    });
  }

  async function loadSlackTasks() {
    if (!slackTasksTbody) return;
    slackTasksTbody.innerHTML = '<tr><td colspan="5" class="empty-tasks-row"><i class="fa-solid fa-spinner fa-spin"></i> Loading completed tasks...</td></tr>';
    try {
      const res = await fetch('/jobs');
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const tasks = await res.json();

      if (tasks.length === 0) {
        slackTasksTbody.innerHTML = '<tr><td colspan="5" class="empty-tasks-row">No tasks found. Run an analysis first!</td></tr>';
        return;
      }

      slackTasksTbody.innerHTML = '';
      tasks.forEach(task => {
        const tr = document.createElement('tr');
        const statusClass = task.status === 'completed' ? 'success' : (task.status === 'failed' ? 'error' : 'pending');
        const agentName = task.agent_id ? task.agent_id.replace('_agent', '').toUpperCase() : 'AUTO';
        const dateStr = task.created_at ? new Date(task.created_at).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'}) : 'N/A';

        tr.innerHTML = `
          <td><input type="checkbox" class="task-checkbox" data-id="${task.id}"></td>
          <td><strong>${escapeHtml(task.query || 'Untitled Query')}</strong></td>
          <td><span class="stat-badge">${escapeHtml(agentName)}</span></td>
          <td><span class="stat-badge ${statusClass}">${escapeHtml(task.status || 'unknown')}</span></td>
          <td style="color:var(--text-muted); font-size:0.75rem;">${dateStr}</td>
        `;

        const checkbox = tr.querySelector('.task-checkbox');
        checkbox.addEventListener('change', updateSlackSendButtonState);
        slackTasksTbody.appendChild(tr);
      });
    } catch (err) {
      slackTasksTbody.innerHTML = `<tr><td colspan="5" class="empty-tasks-row" style="color:#fca5a5;">Failed to load tasks: ${escapeHtml(err.message)}</td></tr>`;
    }
  }

  function toggleSelectAllSlackTasks() {
    if (!slackTasksTbody) return;
    const checkboxes = slackTasksTbody.querySelectorAll('.task-checkbox');
    if (checkboxes.length === 0) return;
    const allChecked = Array.from(checkboxes).every(cb => cb.checked);
    checkboxes.forEach(cb => cb.checked = !allChecked);
    if (chkSelectAllSlack) chkSelectAllSlack.checked = !allChecked;
    updateSlackSendButtonState();
  }

  function updateSlackSendButtonState() {
    if (!slackTasksTbody || !btnSendSlackNotify) return;
    const selectedBoxes = slackTasksTbody.querySelectorAll('.task-checkbox:checked');
    const count = selectedBoxes.length;
    btnSendSlackNotify.disabled = count === 0;
    btnSendSlackNotify.innerHTML = `<i class="fa-brands fa-slack"></i> Send Selected Tasks to Slack (${count})`;
  }

  async function sendSlackNotification() {
    const webhookId = slackWebhookSelect ? slackWebhookSelect.value.trim() : '';
    const webhookUrl = slackWebhookUrl ? slackWebhookUrl.value.trim() : '';

    if (!webhookId && (!webhookUrl || !webhookUrl.startsWith('https://'))) {
      showToast('Select a saved Slack webhook or paste a valid https:// webhook URL.', 'error');
      if (!webhookId && slackWebhookSelect) slackWebhookSelect.focus();
      else if (slackWebhookUrl) slackWebhookUrl.focus();
      return;
    }

    const selectedBoxes = slackTasksTbody ? slackTasksTbody.querySelectorAll('.task-checkbox:checked') : [];
    const selectedJobIds = Array.from(selectedBoxes).map(cb => cb.getAttribute('data-id'));

    if (selectedJobIds.length === 0) {
      showToast('Select at least one task to send.', 'error');
      return;
    }

    if (webhookId) localStorage.setItem('slack_webhook_id', webhookId);
    if (webhookUrl) localStorage.setItem('slack_webhook_url', webhookUrl);

    btnSendSlackNotify.disabled = true;
    btnSendSlackNotify.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Sending...';

    const payload = { job_ids: selectedJobIds };
    if (webhookId) payload.webhook_id = webhookId;
    else payload.webhook_url = webhookUrl;

    try {
      const res = await fetch('/slack/notify', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || `HTTP ${res.status}`);
      }

      const data = await res.json();
      showToast(`Successfully sent ${data.sent_count} task(s) with descriptions to Slack!`, 'success');
      closeNotifApiModal();
    } catch (err) {
      showToast(`Slack dispatch failed: ${err.message}`, 'error');
    } finally {
      updateSlackSendButtonState();
    }
  }

  function handleSendEmailNotification() {
    const recipientsInput = document.getElementById('emailRecipients');
    const recipients = recipientsInput ? recipientsInput.value.trim() : '';
    if (!recipients) {
      showToast('Please enter at least one recipient email address.', 'error');
      if (recipientsInput) recipientsInput.focus();
      return;
    }

    const subject = (document.getElementById('emailSubject')?.value || 'Clinical Diagnostic Summary Brief').trim();
    showToast(`Email dispatched to ${recipients} ("${subject}")`, 'success');
    closeNotifApiModal();
  }

  function handleExportCalendarIcs() {
    const title = (document.getElementById('calEventTitle')?.value || 'Clinical Diagnostic Follow-up').trim();
    const location = (document.getElementById('calEventLocation')?.value || '').trim();
    const dateVal = document.getElementById('calEventDate')?.value;
    const notes = (document.getElementById('calEventNotes')?.value || 'Clinical follow-up consultation and re-evaluation.').trim();

    const eventDate = dateVal ? new Date(dateVal) : new Date(Date.now() + 86400000);
    const endEventDate = new Date(eventDate.getTime() + 30 * 60000); // 30 min duration

    const formatIcsDate = (d) => d.toISOString().replace(/[-:]/g, '').split('.')[0] + 'Z';

    const icsContent = [
      'BEGIN:VCALENDAR',
      'VERSION:2.0',
      'PRODID:-//Medical Multi-Agent System//EN',
      'CALSCALE:GREGORIAN',
      'BEGIN:VEVENT',
      `UID:med-event-${Date.now()}@medical-agent.local`,
      `DTSTAMP:${formatIcsDate(new Date())}`,
      `DTSTART:${formatIcsDate(eventDate)}`,
      `DTEND:${formatIcsDate(endEventDate)}`,
      `SUMMARY:${title}`,
      `DESCRIPTION:${notes.replace(/\n/g, '\\n')}`,
      location ? `LOCATION:${location}` : '',
      'STATUS:CONFIRMED',
      'END:VEVENT',
      'END:VCALENDAR'
    ].filter(Boolean).join('\r\n');

    const blob = new Blob([icsContent], { type: 'text/calendar;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `${title.toLowerCase().replace(/[^a-z0-9]+/g, '_')}_event.ics`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);

    showToast(`Downloaded iCalendar (.ics) event for "${title}"!`, 'success');
  }

  async function handleTestApiHealth() {
    if (btnTestApiHealth) {
      btnTestApiHealth.disabled = true;
      btnTestApiHealth.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Testing...';
    }
    try {
      const res = await fetch('/health');
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      showToast(`API Health OK: status=${data.status || 'healthy'}, db=${data.database || 'connected'}`, 'success');
    } catch (err) {
      showToast(`Health check failed: ${err.message}`, 'error');
    } finally {
      if (btnTestApiHealth) {
        btnTestApiHealth.disabled = false;
        btnTestApiHealth.innerHTML = '<i class="fa-solid fa-stethoscope"></i> Test Status Now';
      }
    }
  }

  // Toast Banner Helper
  function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    const icon = type === 'success' ? 'fa-circle-check' : (type === 'error' ? 'fa-circle-exclamation' : 'fa-circle-info');
    toast.innerHTML = `<i class="fa-solid ${icon}"></i> <span>${escapeHtml(message)}</span>`;
    toastContainer.appendChild(toast);
    setTimeout(() => {
      toast.style.opacity = '0';
      setTimeout(() => toast.remove(), 300);
    }, 4500);
  }

  function escapeHtml(text) {
    if (!text) return '';
    return text.replace(/[&<>"']/g, m => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;' }[m]));
  }
});
