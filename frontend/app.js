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
  const metaKeyInput = document.getElementById('metaKeyInput');
  const metaValInput = document.getElementById('metaValInput');
  const btnAddMetaTag = document.getElementById('btnAddMetaTag');
  const metadataTagsContainer = document.getElementById('metadataTagsContainer');
  const patientDocDropZone = document.getElementById('patientDocDropZone');
  const patientFileInput = document.getElementById('patientFileInput');
  const patientParseStatusBox = document.getElementById('patientParseStatusBox');
  const btnAddCatRow = document.getElementById('btnAddCatRow');

  // Regenerate Modal Elements
  const regenerateModal = document.getElementById('regenerateModal');
  const btnCloseRegenModal = document.getElementById('btnCloseRegenModal');
  const btnCancelRegenModal = document.getElementById('btnCancelRegenModal');
  const btnConfirmRegenerate = document.getElementById('btnConfirmRegenerate');
  const regenSourceJobId = document.getElementById('regenSourceJobId');
  const regenQueryPreview = document.getElementById('regenQueryPreview');
  const regenAgentSelect = document.getElementById('regenAgentSelect');
  const regenModelSelect = document.getElementById('regenModelSelect');

  // Slack Modal Elements
  const btnOpenSlackModal = document.getElementById('btnOpenSlackModal');
  const slackModal = document.getElementById('slackModal');
  const btnCloseSlackModal = document.getElementById('btnCloseSlackModal');
  const btnCancelSlackModal = document.getElementById('btnCancelSlackModal');
  const slackWebhookUrl = document.getElementById('slackWebhookUrl');
  const btnSlackSelectAll = document.getElementById('btnSlackSelectAll');
  const chkSelectAllSlack = document.getElementById('chkSelectAllSlack');
  const slackTasksTbody = document.getElementById('slackTasksTbody');
  const btnSendSlackNotify = document.getElementById('btnSendSlackNotify');

  // Initialize
  checkApiHealth();
  loadAvailableModels();
  setupEventListeners();
  loadConversationsHistory();
  loadPatients();

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
        plusMenuDropdown.classList.toggle('hidden');
        if (modelMenuDropdown) modelMenuDropdown.classList.add('hidden');
      });
    }

    if (btnModelMenu && modelMenuDropdown) {
      btnModelMenu.addEventListener('click', (e) => {
        e.stopPropagation();
        modelMenuDropdown.classList.toggle('hidden');
        if (plusMenuDropdown) plusMenuDropdown.classList.add('hidden');
      });
    }

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
      if (e.key === 'Enter' && !e.shiftKey && (e.ctrlKey || e.metaKey || intakeChatHistory.length > 0)) {
        e.preventDefault();
        sendIntakeChatMessage();
      }
    });

    btnAnalyze.addEventListener('click', submitAsyncAnalysis);
    btnRoute.addEventListener('click', routeQueryOnly);
    btnClear.addEventListener('click', resetForm);

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

    // Regenerate Modal Handlers
    btnCloseRegenModal.addEventListener('click', closeRegenModal);
    btnCancelRegenModal.addEventListener('click', closeRegenModal);
    btnConfirmRegenerate.addEventListener('click', confirmRegenerateJob);

    // Slack Modal Handlers
    btnOpenSlackModal.addEventListener('click', openSlackModal);
    btnCloseSlackModal.addEventListener('click', closeSlackModal);
    btnCancelSlackModal.addEventListener('click', closeSlackModal);
    btnSlackSelectAll.addEventListener('click', toggleSelectAllSlackTasks);
    chkSelectAllSlack.addEventListener('change', toggleSelectAllSlackTasks);
    btnSendSlackNotify.addEventListener('click', sendSlackNotification);

    // Restore saved webhook url
    const savedUrl = localStorage.getItem('slack_webhook_url');
    if (savedUrl) slackWebhookUrl.value = savedUrl;
  }

  // 4. View Switching
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
    try {
      activeConversationId = null;
      currentJob = null;
      if (pollInterval) {
        clearInterval(pollInterval);
        pollInterval = null;
      }
      resetForm({ silent: true });
      switchView('viewConversations');
      if (sidebarConversationSearch) sidebarConversationSearch.value = '';
      if (conversationSearchInput) conversationSearchInput.value = '';
      renderConversationsList();
      if (queryInput) {
        queryInput.value = '';
        queryInput.focus();
      }
      showToast('New conversation ready.', 'info');
    } catch (err) {
      console.error('startNewConversation failed:', err);
      showToast(`Could not start new conversation: ${err.message}`, 'error');
    }
  }

  function filterJobs(list, filter) {
    const q = (filter || '').toLowerCase().trim();
    if (!q) return list.slice();
    return list.filter(j => {
      const hay = [
        j.query,
        j.agent_id,
        j.status,
        j.model,
      ].map(v => String(v || '').toLowerCase()).join(' ');
      return hay.includes(q);
    });
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
      if (conversationsCountTag) {
        conversationsCountTag.textContent = String(conversationsCache.length);
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
    const sideFilter = (sidebarConversationSearch && sidebarConversationSearch.value) || '';
    const mainFilter = (conversationSearchInput && conversationSearchInput.value) || '';

    // Sidebar and main strip use their own search boxes independently.
    const sideFiltered = filterJobs(conversationsCache, sideFilter);
    const mainFiltered = filterJobs(conversationsCache, mainFilter);

    renderSidebarConversations(sideFiltered, sideFilter);
    renderMainConversationsStrip(mainFiltered);
  }

  function renderSidebarConversations(filtered, filterText) {
    if (!sidebarConversationsList) return;

    if (!Array.isArray(conversationsCache) || conversationsCache.length === 0) {
      sidebarConversationsList.innerHTML =
        '<div class="sidebar-conv-empty">No conversations yet.<br><span style="opacity:0.7">Run an analysis to start.</span></div>';
      return;
    }

    if (filtered.length === 0) {
      sidebarConversationsList.innerHTML =
        `<div class="sidebar-conv-empty">No matches${filterText ? ` for “${escapeHtml(filterText)}”` : ''}.</div>`;
      return;
    }

    const groups = groupConversations(filtered);
    const frag = document.createDocumentFragment();

    Object.entries(groups).forEach(([label, items]) => {
      if (!items.length) return;
      const groupLabel = document.createElement('div');
      groupLabel.className = 'sidebar-conv-group-label';
      groupLabel.textContent = label;
      frag.appendChild(groupLabel);

      items.forEach(job => {
        const id = job.id || job.job_id || '';
        const row = document.createElement('div');
        row.className = 'sidebar-conv-item' + (id && id === activeConversationId ? ' active' : '');
        row.dataset.id = id;
        row.setAttribute('role', 'button');
        row.tabIndex = 0;
        row.title = job.query || 'Conversation';

        const agentLabel = job.agent_id
          ? String(job.agent_id).replace(/_agent$/, '').replace(/_/g, ' ')
          : 'auto';
        const dateStr = formatConvDate(job.created_at);
        const title = truncateQuery(job.query || 'Untitled query', 48);
        const hasDocs = conversationHasDocs(job);

        row.innerHTML = `
          <span class="${docsDotClass(job)}" title="${hasDocs ? 'Documentation ready' : escapeHtml(job.status || 'pending')}"></span>
          <div class="conv-text">
            <span class="conv-title">${escapeHtml(title)}</span>
            <span class="conv-meta"><span>${escapeHtml(agentLabel)}</span>${dateStr ? `<span>· ${dateStr}</span>` : ''}</span>
          </div>
          <button type="button" class="btn-conv-delete" title="Delete conversation &amp; reports" aria-label="Delete conversation">
            <i class="fa-solid fa-trash-can"></i>
          </button>
        `;

        const open = () => {
          if (id) loadConversationDetails(id);
        };
        row.addEventListener('click', (e) => {
          if (e.target.closest('.btn-conv-delete')) return;
          open();
        });
        row.addEventListener('keydown', (e) => {
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            open();
          }
        });
        row.querySelector('.btn-conv-delete').addEventListener('click', (e) => {
          e.preventDefault();
          e.stopPropagation();
          if (id) deleteConversation(id);
        });

        frag.appendChild(row);
      });
    });

    sidebarConversationsList.innerHTML = '';
    sidebarConversationsList.appendChild(frag);
  }

  function renderMainConversationsStrip(filtered) {
    if (!conversationsList) return;

    // Show only the most recent few in the main strip (full list is in sidebar).
    const recent = filtered.slice(0, 8);

    if (recent.length === 0) {
      conversationsList.innerHTML =
        '<div class="empty-conversations">Select a conversation from the left menu, or start a new analysis.</div>';
      return;
    }

    conversationsList.innerHTML = '';
    recent.forEach(job => {
      const id = job.id || job.job_id || '';
      const card = document.createElement('div');
      card.className = 'conversation-card';
      if (id && id === activeConversationId) card.classList.add('active');

      const agentLabel = job.agent_id ? String(job.agent_id).replace('_agent', '').toUpperCase() : 'AUTO';
      const statusColor = job.status === 'completed' ? '#34d399' : (job.status === 'failed' ? '#f87171' : '#fbbf24');
      const dateStr = formatConvDate(job.created_at);
      const hasDocs = conversationHasDocs(job);

      card.innerHTML = `
        <div class="conversation-main">
          <span class="conversation-title">
            <span class="docs-dot ${hasDocs ? 'has-docs' : ''}" title="${hasDocs ? 'Docs generated' : 'No docs yet'}"></span>
            ${escapeHtml(truncateQuery(job.query || 'Untitled', 64))}
          </span>
          <div class="conversation-meta">
            <span class="agent-tag">${escapeHtml(agentLabel)}</span>
            <span style="color: ${statusColor}; font-weight:600;">${escapeHtml(job.status || 'pending')}</span>
            <span>${dateStr}</span>
          </div>
        </div>
        <div class="conversation-actions">
          <button class="btn-icon-sm btn-view-job" title="Open" data-id="${id}">
            <i class="fa-solid fa-eye"></i>
          </button>
          <button class="btn-icon-sm danger btn-delete-job" title="Delete conversation &amp; reports" data-id="${id}">
            <i class="fa-solid fa-trash-can"></i>
          </button>
        </div>
      `;

      card.querySelector('.btn-view-job').addEventListener('click', () => id && loadConversationDetails(id));
      card.querySelector('.btn-delete-job').addEventListener('click', () => id && deleteConversation(id));
      card.addEventListener('dblclick', () => id && loadConversationDetails(id));

      conversationsList.appendChild(card);
    });
  }

  async function loadConversationDetails(jobId) {
    try {
      const res = await fetch(`/jobs/${jobId}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const job = await res.json();
      currentJob = job;
      activeConversationId = jobId;
      switchView('viewConversations');
      displayJobResults(job);
      if (job.query) queryInput.value = job.query;
      renderConversationsList();
      showToast(`Loaded: "${truncateQuery(job.query || jobId, 48)}"`, 'info');
    } catch (err) {
      showToast(`Failed to load conversation: ${err.message}`, 'error');
    }
  }

  async function deleteConversation(jobId) {
    if (!confirm('Delete this conversation and all associated reports?')) return;
    try {
      const res = await fetch(`/jobs/${jobId}`, { method: 'DELETE' });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      if (activeConversationId === jobId) {
        activeConversationId = null;
        resetForm({ silent: true });
      }
      showToast('Conversation and reports deleted.', 'success');
      loadConversationsHistory();
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
      
      pollJobStatus(data.job_id);
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
      }
    } catch (err) {
      console.warn('Failed to load patients:', err);
    }
  }

  function renderPatientsTable() {
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
        <td><strong>${escapeHtml(p.name)}</strong></td>
        <td>${escapeHtml(p.age ? String(p.age) + ' yrs' : 'N/A')} / ${escapeHtml(p.gender || 'N/A')}</td>
        <td>${escapeHtml(p.primary_condition || 'Unspecified')}</td>
        <td>${escapeHtml(p.contact_email || p.contact_phone || 'N/A')}</td>
        <td>${metaHtml}</td>
        <td style="color:var(--text-muted); font-size:0.75rem;">${createdStr}</td>
        <td style="text-align: right;">
          <button class="btn-icon-sm btn-edit-patient" title="Edit Patient & Metadata" data-id="${p.id}"><i class="fa-solid fa-pen-to-square"></i></button>
          <button class="btn-icon-sm danger btn-delete-patient" title="Delete Patient" data-id="${p.id}"><i class="fa-solid fa-trash-can"></i></button>
        </td>
      `;

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
    }

    renderMetadataTagsEditor();
    switchCategoryTab('heart');
    patientModal.classList.remove('hidden');
  }

  function closePatientModal() {
    patientModal.classList.add('hidden');
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
    const keys = Object.keys(currentPatientMeta);
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
      showToast('Patient record deleted successfully.', 'info');
      loadPatients();
    } catch (err) {
      showToast(`Failed to delete patient: ${err.message}`, 'error');
    }
  }

  // 8. Document File Upload Handler for Main Analysis
  async function handleFileUpload(file) {
    const ext = file.name.split('.').pop().toLowerCase();
    if (!SUPPORTED_EXTENSIONS.includes(ext)) {
      showFileError(`Unsupported file extension '.${ext}'. Supported: ${SUPPORTED_EXTENSIONS.join(', ')}`);
      return;
    }

    fileErrorAlert.classList.add('hidden');
    parsedStatus.textContent = 'Parsing...';
    parsedStatus.className = 'status-pill pending';

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch('/parse', {
        method: 'POST',
        body: formData
      });

      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.detail || `HTTP ${res.status}`);
      }

      const data = await res.json();
      parsedDocument = data;

      parsedFileName.textContent = data.filename;
      parsedFormat.textContent = (data.metadata?.file_format || ext).toUpperCase();
      parsedPages.textContent = `${data.metadata?.page_count || 1} page(s)`;
      parsedChars.textContent = `${data.metadata?.char_count || data.markdown.length} chars`;
      parsedMarkdownContent.textContent = data.markdown || '(No text extracted)';

      parsedStatus.textContent = data.status === 'success' ? 'Parsed Successfully' : 'Parse Warnings';
      parsedStatus.className = `status-pill ${data.status === 'success' ? 'success' : 'error'}`;

      parsedDocCard.classList.remove('hidden');
      showToast(`Successfully parsed document: ${data.filename}`, 'success');

    } catch (err) {
      console.error('Upload Error:', err);
      showFileError(`Failed to parse document: ${err.message}`);
    }
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
    const text = queryInput.value.trim();
    if (!text && intakeChatHistory.length === 0) {
      showToast('Please enter a medical query to start intake chat.', 'error');
      queryInput.focus();
      return;
    }

    if (text) {
      intakeChatHistory.push({ role: 'user', content: text });
      queryInput.value = '';
    }

    isIntakeChatLoading = true;
    updateIntakeChatUI();

    btnChatSend.disabled = true;
    btnChatSend.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Sending...';

    try {
      let docContext = null;
      if (parsedDocument && parsedDocument.markdown && attachContextCheck.checked) {
        docContext = parsedDocument.markdown;
      }

      const res = await fetch('/intake/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: intakeChatHistory,
          model: modelSelect.value,
          document_context: docContext
        })
      });

      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();

      if (data && data.content) {
        intakeChatHistory.push({ role: 'assistant', content: data.content });
      }
    } catch (err) {
      showToast(`Intake assistant error: ${err.message}`, 'error');
    } finally {
      isIntakeChatLoading = false;
      btnChatSend.disabled = false;
      btnChatSend.innerHTML = '<i class="fa-solid fa-paper-plane"></i> Send / Clarify';
      updateIntakeChatUI();
      queryInput.focus();
    }
  }

  // 9. Submit Asynchronous Analysis Run
  async function submitAsyncAnalysis() {
    let query = queryInput.value.trim();

    // If chat turns exist, summarize conversation context
    if (intakeChatHistory.length > 0) {
      if (query) {
        intakeChatHistory.push({ role: 'user', content: query });
        queryInput.value = '';
        updateIntakeChatUI();
      }

      btnAnalyze.disabled = true;
      btnAnalyze.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Summarizing Intake Chat...';

      try {
        let docContext = null;
        if (parsedDocument && parsedDocument.markdown && attachContextCheck.checked) {
          docContext = parsedDocument.markdown;
        }

        const sumRes = await fetch('/intake/summarize', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            messages: intakeChatHistory,
            model: modelSelect.value,
            document_context: docContext
          })
        });

        if (sumRes.ok) {
          const sumData = await sumRes.json();
          if (sumData && sumData.summary) {
            query = sumData.summary;
          }
        }
      } catch (err) {
        console.warn('Summary endpoint error, using transcript fallback:', err);
        query = intakeChatHistory.map(m => `${m.role.toUpperCase()}: ${m.content}`).join('\n\n');
      }
    }

    if (!query) {
      showToast('Please enter a medical query or clinical subject.', 'error');
      queryInput.focus();
      btnAnalyze.disabled = false;
      btnAnalyze.innerHTML = '<i class="fa-solid fa-play"></i> Start Analysis Run';
      return;
    }

    if (parsedDocument && parsedDocument.markdown && attachContextCheck.checked && intakeChatHistory.length === 0) {
      query += `\n\n--- ATTACHED MEDICAL DOCUMENT (${parsedDocument.filename}) ---\n` + parsedDocument.markdown;
    }

    const payload = {
      query: query,
      model: modelSelect.value,
      implementation: 'langchain',
      web_search: webSearchToggle.checked,
      timeout: 300
    };

    const targetAgentOverride = agentSelect.value;
    if (targetAgentOverride !== 'auto') {
      payload.agent_id = targetAgentOverride;
    }

    btnAnalyze.disabled = true;
    btnAnalyze.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Submitting...';

    try {
      const res = await fetch('/analyze/async', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.detail || `HTTP ${res.status}`);
      }

      const data = await res.json();
      currentJob = data;
      activeConversationId = data.job_id;
      showToast(`Analysis job submitted (ID: ${data.job_id.substring(0, 8)}...)`, 'info');

      placeholderState.classList.add('hidden');
      routeResultCard.classList.add('hidden');
      filesContainer.classList.add('hidden');
      reportPreviewCard.classList.add('hidden');
      jobProgressCard.classList.remove('hidden');

      jobIdTag.textContent = `Job ID: ${data.job_id}`;
      jobStatusHeading.textContent = 'Job Submitted — Processing...';
      updateProgressSteps('pending');

      pollJobStatus(data.job_id);
      loadConversationsHistory();

    } catch (err) {
      showToast(`Failed to submit analysis: ${err.message}`, 'error');
    } finally {
      btnAnalyze.disabled = false;
      btnAnalyze.innerHTML = '<i class="fa-solid fa-play"></i> Start Analysis Run';
    }
  }

  // 10. Poll Job Status
  function pollJobStatus(jobId) {
    if (pollInterval) clearInterval(pollInterval);

    pollInterval = setInterval(async () => {
      try {
        const res = await fetch(`/jobs/${jobId}`);
        if (!res.ok) return;

        const job = await res.json();
        currentJob = job;

        updateProgressSteps(job.status);

        if (job.status === 'completed') {
          clearInterval(pollInterval);
          showToast('Analysis completed successfully!', 'success');
          jobProgressCard.classList.add('hidden');
          displayJobResults(job);
          loadConversationsHistory();
        } else if (job.status === 'failed') {
          clearInterval(pollInterval);
          jobStatusHeading.textContent = 'Analysis Run Failed';
          showToast(`Job failed: ${job.error || 'Unknown error'}`, 'error');
          loadConversationsHistory();
        }
      } catch (err) {
        console.warn('Poll error:', err);
      }
    }, 2500);
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

        const fileName = filePath.split('/').pop();
        const btn = document.createElement('a');
        btn.className = 'file-card-btn';
        btn.href = `/${filePath}`;
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

    await loadReportText('patient', activeFiles['patient_report']);
    await loadReportText('practitioner', activeFiles['practitioner_report']);
    await loadReportText('summary', activeFiles['summary'] || activeFiles['medication_summary']);

    if (job.result) {
      loadedReportTexts['json'] = JSON.stringify(job.result, null, 2);
    }

    const patientTabBtn = document.querySelector('.report-tabs .tab-btn[data-tab="patient"]');
    if (patientTabBtn) patientTabBtn.click();
  }

  async function loadReportText(tabKey, filePath) {
    if (!filePath) return;
    try {
      const res = await fetch(`/${filePath}`);
      if (res.ok) {
        const text = await res.text();
        loadedReportTexts[tabKey] = text;
      }
    } catch (err) {
      console.warn(`Could not load report file ${filePath}:`, err);
    }
  }

  function renderReportTab(tabKey) {
    const text = loadedReportTexts[tabKey];
    if (!text) {
      markdownViewer.innerHTML = '<p class="text-muted">No content available for this report view.</p>';
      btnDownloadCurrent.style.display = 'none';
      return;
    }

    if (tabKey === 'json') {
      markdownViewer.innerHTML = `<pre class="json-box">${escapeHtml(text)}</pre>`;
      btnDownloadCurrent.style.display = 'none';
    } else {
      markdownViewer.innerHTML = marked.parse(text);
      
      const filePath = activeFiles[`${tabKey}_report`] || activeFiles[tabKey];
      if (filePath) {
        btnDownloadCurrent.href = `/${filePath}`;
        btnDownloadCurrent.style.display = 'inline-flex';
      } else {
        btnDownloadCurrent.style.display = 'none';
      }
    }
  }

  // 12. Route Query Only (Synchronous Classification)
  async function routeQueryOnly() {
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

      placeholderState.classList.add('hidden');
      jobProgressCard.classList.add('hidden');
      filesContainer.classList.add('hidden');
      reportPreviewCard.classList.add('hidden');
      routeResultCard.classList.remove('hidden');

      routedAgentName.textContent = data.agent_name || data.agent_id;
      routedAgentDesc.textContent = `Query classified for ${data.agent_id}. Click "Start Analysis Run" to invoke reasoning pipeline.`;

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

  // Slack Modal Handlers
  function openSlackModal() {
    slackModal.classList.remove('hidden');
    loadSlackTasks();
  }

  function closeSlackModal() {
    slackModal.classList.add('hidden');
  }

  async function loadSlackTasks() {
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
    const checkboxes = slackTasksTbody.querySelectorAll('.task-checkbox');
    if (checkboxes.length === 0) return;
    const allChecked = Array.from(checkboxes).every(cb => cb.checked);
    checkboxes.forEach(cb => cb.checked = !allChecked);
    chkSelectAllSlack.checked = !allChecked;
    updateSlackSendButtonState();
  }

  function updateSlackSendButtonState() {
    const selectedBoxes = slackTasksTbody.querySelectorAll('.task-checkbox:checked');
    const count = selectedBoxes.length;
    btnSendSlackNotify.disabled = count === 0;
    btnSendSlackNotify.innerHTML = `<i class="fa-brands fa-slack"></i> Send Selected Tasks (${count})`;
  }

  async function sendSlackNotification() {
    const webhookUrl = slackWebhookUrl.value.trim();
    if (!webhookUrl || !webhookUrl.startsWith('https://')) {
      showToast('Please enter a valid Slack Webhook URL (starts with https://)', 'error');
      slackWebhookUrl.focus();
      return;
    }

    const selectedBoxes = slackTasksTbody.querySelectorAll('.task-checkbox:checked');
    const selectedJobIds = Array.from(selectedBoxes).map(cb => cb.getAttribute('data-id'));

    if (selectedJobIds.length === 0) {
      showToast('Select at least one task to send.', 'error');
      return;
    }

    localStorage.setItem('slack_webhook_url', webhookUrl);

    btnSendSlackNotify.disabled = true;
    btnSendSlackNotify.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Sending...';

    try {
      const res = await fetch('/slack/notify', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          webhook_url: webhookUrl,
          job_ids: selectedJobIds
        })
      });

      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.detail || `HTTP ${res.status}`);
      }

      const data = await res.json();
      showToast(`Successfully sent ${data.sent_count} task(s) with descriptions to Slack!`, 'success');
      closeSlackModal();
    } catch (err) {
      showToast(`Slack dispatch failed: ${err.message}`, 'error');
    } finally {
      updateSlackSendButtonState();
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
