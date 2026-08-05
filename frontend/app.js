/**
 * Medical Multi-Agent Intelligence Web Application Script
 * Communicates with FastAPI REST API endpoints (/analyze, /analyze/async, /parse, /route, /models, /health)
 */

document.addEventListener('DOMContentLoaded', () => {
  // State management
  let parsedDocument = null; // { filename, markdown, metadata }
  let currentJob = null;     // active async job
  let pollInterval = null;   // polling timer
  let activeFiles = {};      // generated report file paths
  let loadedReportTexts = {};// cached report markdown strings

  const SUPPORTED_EXTENSIONS = ['pdf', 'docx', 'doc', 'txt', 'md', 'rtf', 'png', 'jpg', 'jpeg', 'webp', 'gif'];

  // DOM Elements
  const apiStatusBadge = document.getElementById('apiStatusBadge');
  const apiStatusText = document.getElementById('apiStatusText');
  const queryInput = document.getElementById('queryInput');
  const modelSelect = document.getElementById('modelSelect');
  const agentSelect = document.getElementById('agentSelect');
  const webSearchToggle = document.getElementById('webSearchToggle');

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

  // 1. API Health Check
  async function checkApiHealth() {
    try {
      const res = await fetch('/health');
      if (res.ok) {
        const data = await res.json();
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
    // Sample Chips
    document.querySelectorAll('.chip').forEach(chip => {
      chip.addEventListener('click', () => {
        queryInput.value = chip.getAttribute('data-query');
        queryInput.focus();
      });
    });

    // Drag & Drop
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
        handleFileSelected(e.dataTransfer.files[0]);
      }
    });

    fileInput.addEventListener('change', (e) => {
      if (e.target.files && e.target.files.length > 0) {
        handleFileSelected(e.target.files[0]);
      }
    });

    fileErrorClose.addEventListener('click', () => {
      fileErrorAlert.classList.add('hidden');
    });

    btnToggleDocPreview.addEventListener('click', () => {
      parsedPreviewBox.classList.toggle('hidden');
      const isHidden = parsedPreviewBox.classList.contains('hidden');
      btnToggleDocPreview.innerHTML = isHidden 
        ? '<i class="fa-solid fa-chevron-down"></i> Preview' 
        : '<i class="fa-solid fa-chevron-up"></i> Hide';
    });

    btnRemoveDoc.addEventListener('click', removeParsedDocument);

    btnCopyParsed.addEventListener('click', () => {
      if (parsedDocument && parsedDocument.markdown) {
        navigator.clipboard.writeText(parsedDocument.markdown);
        showToast('Parsed markdown copied to clipboard!', 'success');
      }
    });

    // Action Buttons
    btnAnalyze.addEventListener('click', startAnalysis);
    btnRoute.addEventListener('click', runRouteOnly);
    btnClear.addEventListener('click', resetForm);

    // Report Tab Buttons
    document.querySelectorAll('.tab-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        const tabType = btn.getAttribute('data-tab');
        renderReportTab(tabType);
      });
    });

    btnCopyReport.addEventListener('click', () => {
      const activeTab = document.querySelector('.tab-btn.active').getAttribute('data-tab');
      const text = loadedReportTexts[activeTab] || '';
      if (text) {
        navigator.clipboard.writeText(text);
        showToast('Report content copied to clipboard!', 'success');
      }
    });

    // Slack Modal Handlers
    if (btnOpenSlackModal) {
      btnOpenSlackModal.addEventListener('click', openSlackModal);
    }
    if (btnCloseSlackModal) {
      btnCloseSlackModal.addEventListener('click', closeSlackModal);
    }
    if (btnCancelSlackModal) {
      btnCancelSlackModal.addEventListener('click', closeSlackModal);
    }
    if (btnSlackSelectAll) {
      btnSlackSelectAll.addEventListener('click', toggleSelectAllSlackTasks);
    }
    if (chkSelectAllSlack) {
      chkSelectAllSlack.addEventListener('change', (e) => {
        const checkboxes = slackTasksTbody.querySelectorAll('.task-checkbox');
        checkboxes.forEach(cb => cb.checked = e.target.checked);
        updateSlackSendButtonState();
      });
    }
    if (btnSendSlackNotify) {
      btnSendSlackNotify.addEventListener('click', sendSlackNotification);
    }
  }

  // 4. File Format Validation & Parsing (/parse)
  async function handleFileSelected(file) {
    fileErrorAlert.classList.add('hidden');
    const ext = file.name.split('.').pop().toLowerCase();

    // Check unsupported file formats (e.g. png, jpg, exe, zip, mp4)
    if (!SUPPORTED_EXTENSIONS.includes(ext)) {
      fileInput.value = '';
      fileErrorMsg.innerHTML = `<strong>Unsupported File Format (.${ext.toUpperCase()}):</strong> File "${file.name}" is not supported.<br>Supported formats: <strong>PDF (.pdf), Word (.docx, .doc), Images (.png, .jpg, .webp), Text (.txt, .md), and RTF (.rtf)</strong>.`;
      fileErrorAlert.classList.remove('hidden');
      showToast(`Unsupported file format: .${ext}`, 'error');
      return;
    }

    // Process Supported File
    showToast(`Parsing ${file.name}...`, 'info');
    parsedFileName.textContent = file.name;
    parsedFormat.textContent = ext.toUpperCase();
    parsedPages.textContent = 'Parsing...';
    parsedChars.textContent = '...';
    parsedStatus.textContent = 'Processing';
    parsedStatus.className = 'stat-badge';
    parsedCardShow();

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch('/parse', {
        method: 'POST',
        body: formData
      });

      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.detail || 'Parsing error');
      }

      const data = await res.json();
      parsedDocument = data;

      // Update Card UI
      parsedFileName.textContent = data.filename;
      parsedFormat.textContent = (data.metadata.file_format || ext).toUpperCase();
      parsedPages.textContent = `${data.metadata.page_count || 1} page(s)`;
      parsedChars.textContent = `${(data.metadata.char_count || data.markdown.length).toLocaleString()} chars`;
      parsedStatus.textContent = data.status || 'success';
      parsedStatus.className = 'stat-badge success';

      parsedMarkdownContent.textContent = data.markdown || '(No text content extracted)';
      showToast(`Successfully parsed ${data.filename}`, 'success');

    } catch (err) {
      parsedStatus.textContent = 'Failed';
      parsedStatus.className = 'stat-badge error';
      fileErrorMsg.innerHTML = `<strong>Document Parsing Error:</strong> ${err.message}`;
      fileErrorAlert.classList.remove('hidden');
      showToast(`Error parsing document: ${err.message}`, 'error');
    }
  }

  function parsedCardShow() {
    parsedDocCard.classList.remove('hidden');
  }

  function removeParsedDocument() {
    parsedDocument = null;
    fileInput.value = '';
    parsedDocCard.classList.add('hidden');
    parsedPreviewBox.classList.add('hidden');
    fileErrorAlert.classList.add('hidden');
    showToast('Attached document removed', 'info');
  }

  // 5. Run Query Routing Only (/route)
  async function runRouteOnly() {
    const query = queryInput.value.trim();
    if (!query) {
      showToast('Please enter a medical query or subject.', 'error');
      queryInput.focus();
      return;
    }

    btnRoute.disabled = true;
    showToast('Routing query...', 'info');

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
      routeResultCard.classList.remove('hidden');

      const agentMap = {
        'medication_agent': { name: 'Medication Specialist', desc: 'Focuses on drug pharmacology, interactions, side effects, dosing, and safety.' },
        'procedure_agent': { name: 'Procedure Specialist', desc: 'Provides organ-by-organ perioperative care, anatomical risks, and surgical analysis.' },
        'diagnostic_agent': { name: 'Diagnostic Specialist', desc: 'Differential diagnosis pipeline linking symptoms to potential conditions.' },
        'general_agent': { name: 'Fact-Checker Specialist', desc: 'Open medical evidence analysis with multi-perspective synthesis.' }
      };

      const info = agentMap[data.routed_agent_id] || { name: data.routed_agent_id, desc: 'Specialized clinical reasoning agent.' };
      routedAgentName.textContent = info.name;
      routedAgentDesc.textContent = info.desc;
      activeAgentBadge.textContent = info.name;

      showToast(`Routed to ${info.name}`, 'success');

    } catch (err) {
      showToast(`Routing failed: ${err.message}`, 'error');
    } finally {
      btnRoute.disabled = false;
    }
  }

  // 6. Start Full Multi-Agent Analysis (/analyze/async)
  async function startAnalysis() {
    let query = queryInput.value.trim();
    if (!query) {
      showToast('Please enter a medical query or subject.', 'error');
      queryInput.focus();
      return;
    }

    // Attach Grounded Parsed Document Context if checked
    if (parsedDocument && attachContextCheck.checked && parsedDocument.markdown) {
      query += `\n\n--- ATTACHED CLINICAL DOCUMENT: ${parsedDocument.filename} ---\n${parsedDocument.markdown}`;
    }

    btnAnalyze.disabled = true;
    placeholderState.classList.add('hidden');
    routeResultCard.classList.add('hidden');
    filesContainer.classList.add('hidden');
    reportPreviewCard.classList.add('hidden');
    jobProgressCard.classList.remove('hidden');

    updateProgressUI(10, 'Submitting Analysis Job...', 'stepRoute');

    try {
      const reqPayload = {
        query: query,
        model: modelSelect.value,
        implementation: 'langchain',
        web_search: webSearchToggle.checked,
        timeout: 300
      };

      const res = await fetch('/analyze/async', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(reqPayload)
      });

      if (!res.ok) {
        const errJson = await res.json();
        throw new Error(errJson.detail || 'Failed to start job');
      }

      const jobData = await res.json();
      currentJob = jobData;
      jobIdTag.textContent = `Job #${jobData.job_id.substring(0, 8)}`;

      // Start Polling Loop
      pollJobStatus(jobData.job_id);

    } catch (err) {
      btnAnalyze.disabled = false;
      jobProgressCard.classList.add('hidden');
      showToast(`Failed to start analysis: ${err.message}`, 'error');
    }
  }

  // 7. Poll Job Status (/jobs/{id})
  function pollJobStatus(jobId) {
    if (pollInterval) clearInterval(pollInterval);

    let stepCount = 0;
    pollInterval = setInterval(async () => {
      stepCount++;
      try {
        const res = await fetch(`/jobs/${jobId}`);
        if (!res.ok) throw new Error('Job poll error');
        const job = await res.json();

        if (job.status === 'running') {
          if (stepCount <= 3) {
            updateProgressUI(30, 'Routing and Prompt Synthesis...', 'stepRoute');
          } else if (stepCount <= 8) {
            updateProgressUI(60, 'Invoking Multi-Agent Reasoning Pipeline...', 'stepReasoning');
          } else {
            updateProgressUI(85, 'Validating Citation URLs & Building Reports...', 'stepValidation');
          }
        } else if (job.status === 'completed') {
          clearInterval(pollInterval);
          updateProgressUI(100, 'Analysis Complete!', 'stepReports');
          setTimeout(() => {
            jobProgressCard.classList.add('hidden');
            handleAnalysisCompleted(job);
            btnAnalyze.disabled = false;
          }, 600);
        } else if (job.status === 'failed') {
          clearInterval(pollInterval);
          jobProgressCard.classList.add('hidden');
          btnAnalyze.disabled = false;
          showToast(`Analysis Job Failed: ${job.error || 'Unknown error'}`, 'error');
        }
      } catch (err) {
        console.warn('Poll error:', err);
      }
    }, 2000);
  }

  function updateProgressUI(pct, label, activeStepId) {
    progressBarFill.style.width = `${pct}%`;
    jobStatusHeading.textContent = label;

    [stepRoute, stepReasoning, stepValidation, stepReports].forEach(el => {
      el.className = 'step';
    });

    if (activeStepId === 'stepRoute') {
      stepRoute.className = 'step step-active';
    } else if (activeStepId === 'stepReasoning') {
      stepRoute.className = 'step step-done';
      stepReasoning.className = 'step step-active';
    } else if (activeStepId === 'stepValidation') {
      stepRoute.className = 'step step-done';
      stepReasoning.className = 'step step-done';
      stepValidation.className = 'step step-active';
    } else if (activeStepId === 'stepReports') {
      stepRoute.className = 'step step-done';
      stepReasoning.className = 'step step-done';
      stepValidation.className = 'step step-done';
      stepReports.className = 'step step-active';
    }
  }

  // 8. Handle Analysis Complete & Render Reports
  async function handleAnalysisCompleted(job) {
    const resultData = job.result || {};
    // Extract generated files from top-level job.files or nested job.result.files
    activeFiles = job.files || resultData.files || {};
    loadedReportTexts = {};

    const agentId = job.agent_id || resultData.agent_id || 'Agent';
    activeAgentBadge.textContent = agentId.replace(/_/g, ' ').toUpperCase();
    showToast('Analysis completed successfully!', 'success');

    // Render Files Grid
    renderFilesGrid(activeFiles);

    // Load Report Text Content
    await loadReportContents(activeFiles);

    // Display Preview Box
    reportPreviewCard.classList.remove('hidden');

    // Render Default Active Tab (Patient Report or Summary)
    renderReportTab('patient');
  }

  function renderFilesGrid(files) {
    filesGrid.innerHTML = '';
    filesContainer.classList.remove('hidden');

    const fileEntries = Object.entries(files);
    if (fileEntries.length === 0) {
      filesGrid.innerHTML = '<p style="color:var(--text-muted)">No output files generated.</p>';
      return;
    }

    fileEntries.forEach(([key, path]) => {
      const isPdf = path.endsWith('.pdf');
      const isJson = path.endsWith('.json');
      const filename = path.split('/').pop();

      const card = document.createElement('div');
      card.className = 'file-card';

      let iconClass = 'fa-file-lines';
      if (isPdf) iconClass = 'fa-file-pdf';
      if (isJson) iconClass = 'fa-file-code';

      // Standardize download link via static mount /outputs
      const fileUrl = `/outputs/${filename}`;

      card.innerHTML = `
        <div class="file-card-top">
          <i class="fa-solid ${iconClass} file-card-icon"></i>
          <div>
            <div class="file-card-title">${formatFileKey(key)}</div>
            <div class="file-card-meta">${filename}</div>
          </div>
        </div>
        <div class="file-card-actions">
          <a href="${fileUrl}" target="_blank" class="btn btn-xs btn-outline">
            <i class="fa-solid fa-arrow-up-right-from-square"></i> Open
          </a>
          <a href="${fileUrl}" download class="btn btn-xs btn-primary">
            <i class="fa-solid fa-download"></i> Download
          </a>
        </div>
      `;

      filesGrid.appendChild(card);
    });
  }

  function formatFileKey(key) {
    return key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
  }

  // Load report markdown content via fetch
  async function loadReportContents(files) {
    const keysToFetch = {
      patient: files.patient_report || files.patient_md,
      practitioner: files.practitioner_report || files.practitioner_md,
      summary: files.summary_report || files.markdown_report || files.summary_md || files.medication_summary,
      json: files.json_session || files.session || files.result || files.analysis_result || files.medication_analysis
    };

    for (const [tabKey, filePath] of Object.entries(keysToFetch)) {
      if (filePath) {
        const filename = filePath.split('/').pop();
        const fileUrl = `/outputs/${filename}`;
        try {
          const res = await fetch(fileUrl);
          if (res.ok) {
            loadedReportTexts[tabKey] = await res.text();
          }
        } catch (e) {
          console.warn(`Failed to fetch report ${filePath}`, e);
        }
      }
    }
  }

  // Render Selected Tab Content
  function renderReportTab(tabType) {
    let rawContent = loadedReportTexts[tabType];

    // Fallback to summary/markdown report if specific tab content is missing
    if (!rawContent && (tabType === 'patient' || tabType === 'practitioner')) {
      rawContent = loadedReportTexts['summary'];
    }

    btnDownloadCurrent.setAttribute('href', '#');

    // Update download link for active tab
    const fileKeyMap = {
      patient: activeFiles.patient_report || activeFiles.markdown_report || activeFiles.summary_report,
      practitioner: activeFiles.practitioner_report || activeFiles.markdown_report || activeFiles.summary_report,
      summary: activeFiles.summary_report || activeFiles.markdown_report || activeFiles.medication_summary,
      json: activeFiles.json_session || activeFiles.session || activeFiles.result || activeFiles.analysis_result
    };
    const currentPath = fileKeyMap[tabType];
    if (currentPath) {
      const filename = currentPath.split('/').pop();
      btnDownloadCurrent.setAttribute('href', `/outputs/${filename}`);
    }

    if (!rawContent) {
      markdownViewer.innerHTML = `<p style="color:var(--text-muted); padding:2rem; text-align:center;">No report content available for <strong>${tabType}</strong>.</p>`;
      return;
    }

    if (tabType === 'json') {
      try {
        const parsedJson = JSON.parse(rawContent);
        markdownViewer.innerHTML = `<pre><code>${escapeHtml(JSON.stringify(parsedJson, null, 2))}</code></pre>`;
      } catch (e) {
        markdownViewer.innerHTML = `<pre><code>${escapeHtml(rawContent)}</code></pre>`;
      }
    } else {
      // Render Markdown using marked.js
      if (window.marked) {
        let renderedHtml = window.marked.parse(rawContent);
        // Highlight hardcoded medical disclaimer
        renderedHtml = renderedHtml.replace(
          /(Disclaimer:[\s\S]*?)(?=<h|$)/gi, 
          '<div class="disclaimer-box"><i class="fa-solid fa-triangle-exclamation"></i> <strong>Medical Disclaimer:</strong> $1</div>'
        );
        markdownViewer.innerHTML = renderedHtml;
      } else {
        markdownViewer.innerText = rawContent;
      }
    }
  }

  // 9. Reset Form
  function resetForm() {
    queryInput.value = '';
    removeParsedDocument();
    placeholderState.classList.remove('hidden');
    jobProgressCard.classList.add('hidden');
    routeResultCard.classList.add('hidden');
    filesContainer.classList.add('hidden');
    reportPreviewCard.classList.add('hidden');
    activeAgentBadge.textContent = 'Ready';
    showToast('Workbench reset', 'info');
  }

  // Utility Helpers
  function showToast(msg, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    let icon = 'fa-circle-info';
    if (type === 'error') icon = 'fa-circle-xmark';
    if (type === 'success') icon = 'fa-circle-check';

    toast.innerHTML = `<i class="fa-solid ${icon}"></i> <span>${msg}</span>`;
    toastContainer.appendChild(toast);

    setTimeout(() => {
      toast.style.opacity = '0';
      toast.style.transform = 'translateY(10px)';
      setTimeout(() => toast.remove(), 300);
    }, 4000);
  }

  function escapeHtml(str) {
    return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }

  // 10. Slack Notifications Modal Logic
  function openSlackModal() {
    const savedUrl = localStorage.getItem('slack_webhook_url') || '';
    slackWebhookUrl.value = savedUrl;
    slackModal.classList.remove('hidden');
    loadSlackTasks();
  }

  function closeSlackModal() {
    slackModal.classList.add('hidden');
  }

  async function loadSlackTasks() {
    slackTasksTbody.innerHTML = '<tr><td colspan="5" class="empty-tasks-row"><i class="fa-solid fa-spinner fa-spin"></i> Loading tasks...</td></tr>';
    chkSelectAllSlack.checked = false;
    updateSlackSendButtonState();

    try {
      const res = await fetch('/jobs');
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const tasks = await res.json();

      if (!tasks || tasks.length === 0) {
        slackTasksTbody.innerHTML = '<tr><td colspan="5" class="empty-tasks-row"><i class="fa-solid fa-folder-open"></i> No tasks found. Run an analysis first to generate tasks.</td></tr>';
        return;
      }

      slackTasksTbody.innerHTML = '';
      tasks.forEach(task => {
        const tr = document.createElement('tr');
        const statusClass = task.status === 'completed' ? 'success' : (task.status === 'failed' ? 'error' : '');
        const dateStr = task.created_at ? new Date(task.created_at).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'}) : 'N/A';
        const agentName = task.agent_id ? task.agent_id.replace('_agent', '').replace('_', ' ') : 'Auto';

        tr.innerHTML = `
          <td><input type="checkbox" class="task-checkbox" data-id="${task.id}"></td>
          <td><strong style="color:var(--text-bright);">${escapeHtml(task.query || 'Query')}</strong></td>
          <td><span class="stat-badge">${escapeHtml(agentName)}</span></td>
          <td><span class="stat-badge ${statusClass}">${escapeHtml(task.status || 'unknown')}</span></td>
          <td style="color:var(--text-muted); font-size:0.75rem;">${dateStr}</td>
        `;

        const checkbox = tr.querySelector('.task-checkbox');
        checkbox.addEventListener('change', updateSlackSendButtonState);
        slackTasksTbody.appendChild(tr);
      });

    } catch (err) {
      console.error('Error fetching jobs:', err);
      slackTasksTbody.innerHTML = `<tr><td colspan="5" class="empty-tasks-row" style="color:#fca5a5;"><i class="fa-solid fa-triangle-exclamation"></i> Failed to load tasks: ${escapeHtml(err.message)}</td></tr>`;
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
});
