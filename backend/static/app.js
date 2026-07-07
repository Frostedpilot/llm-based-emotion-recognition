// backend/static/app.js

// ── Application State ────────────────────────────────────────────────────────
const state = {
    datasets: [],
    selectedDataset: '',
    dialogues: [],
    selectedDialogueId: '',
    dialogueContent: null,
    selectedUtteranceIndex: -1,
    emotionLabels: [],
    chartInstance: null,
    streamController: null
};

// ── DOM Elements ─────────────────────────────────────────────────────────────
const el = {
    datasetSelect: document.getElementById('dataset-select'),
    dialogueSearch: document.getElementById('dialogue-search'),
    dialogueList: document.getElementById('dialogue-list'),
    historyList: document.getElementById('history-list'),
    uttSpeaker: document.getElementById('utt-speaker'),
    uttText: document.getElementById('utt-text'),
    uttGt: document.getElementById('utt-gt'),
    sourceBadge: document.getElementById('source-badge'),
    mediaWrapper: document.getElementById('media-wrapper'),
    
    // Configurations
    modelPresetSelect: document.getElementById('model-preset-select'),
    providerSelect: document.getElementById('provider-select'),
    modelIdInput: document.getElementById('model-id-input'),
    windowSlider: document.getElementById('window-slider'),
    windowVal: document.getElementById('window-val'),
    tempSlider: document.getElementById('temp-slider'),
    tempVal: document.getElementById('temp-val'),
    framesInput: document.getElementById('frames-input'),
    disableThinkingCheck: document.getElementById('disable-thinking-check'),
    maxTokensInput: document.getElementById('max-tokens-input'),
    reasoningMaxTokensInput: document.getElementById('reasoning-max-tokens-input'),
    runBtn: document.getElementById('run-btn'),
    
    // Outputs
    agentTabsBar: document.getElementById('agent-tabs-bar'),
    agentPanesContainer: document.getElementById('agent-panes-container'),
    
    // Evaluation
    evalGt: document.getElementById('eval-gt'),
    evalPred: document.getElementById('eval-pred'),
    evalMatchIndicator: document.getElementById('eval-match-indicator'),
    evalStatusText: document.getElementById('eval-status-text'),
    evalJsd: document.getElementById('eval-jsd'),
    distributionChart: document.getElementById('distribution-chart')
};

// Emotion labels mapping for HSL colors matching style.css
const EMOTION_CLASSES = {
    'neutral': 'emotion-neutral',
    'neu': 'emotion-neutral',
    'joy': 'emotion-joy',
    'happiness': 'emotion-joy',
    'excited': 'emotion-joy',
    'excitement': 'emotion-joy',
    'hap': 'emotion-joy',
    'exc': 'emotion-joy',
    'sadness': 'emotion-sadness',
    'sad': 'emotion-sadness',
    'anger': 'emotion-anger',
    'ang': 'emotion-anger',
    'fear': 'emotion-fear',
    'fea': 'emotion-fear',
    'surprise': 'emotion-surprise',
    'sur': 'emotion-surprise',
    'frustration': 'emotion-frustration',
    'fru': 'emotion-frustration',
    'disgust': 'emotion-disgust',
    'dis': 'emotion-disgust'
};

const EMOTION_LABELS_MELD = ["neutral", "surprise", "fear", "sadness", "joy", "disgust", "anger"];
const EMOTION_LABELS_IEMOCAP = ["neutral", "happiness", "sadness", "anger", "fear", "frustration", "excitement"];
const EMOTION_LABELS_IEMOCAP_6CLASS = ["neutral", "frustration", "excitement", "sadness", "anger", "happiness"];

// ── Initialization ───────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
    fetchDatasets();
    initChart();
});

function setupEventListeners() {
    // Select selectors
    el.datasetSelect.addEventListener('change', (e) => {
        state.selectedDataset = e.target.value;
        fetchDialogues(state.selectedDataset);
        updateEmotionLabels();
    });
    
    el.dialogueSearch.addEventListener('input', () => {
        renderDialogueList();
    });
    
    // Bind configuration sliders
    el.windowSlider.addEventListener('input', (e) => {
        el.windowVal.textContent = e.target.value;
    });
    el.tempSlider.addEventListener('input', (e) => {
        el.tempVal.textContent = parseFloat(e.target.value).toFixed(1);
    });
    
    // Modality and label mode updates
    el.modelPresetSelect.addEventListener('change', updateConfigPresets);
    document.querySelectorAll('input[name="label_mode"]').forEach(radio => {
        radio.addEventListener('change', updateConfigPresets);
    });
    
    // Execute Button
    el.runBtn.addEventListener('click', runProbe);

    // Main tab switching
    document.querySelectorAll('.main-tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const targetTab = btn.getAttribute('data-tab');
            document.querySelectorAll('.main-tab-btn').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(pane => pane.classList.add('hidden'));
            
            btn.classList.add('active');
            document.getElementById(targetTab).classList.remove('hidden');
        });
    });
}

// ── API Functions ────────────────────────────────────────────────────────────
async function fetchDatasets() {
    try {
        const resp = await fetch('/datasets');
        state.datasets = await resp.json();
        
        el.datasetSelect.innerHTML = '<option value="" disabled selected>Select a dataset...</option>';
        state.datasets.forEach(ds => {
            if (ds.status === 'ready') {
                const opt = document.createElement('option');
                opt.value = ds.id;
                opt.textContent = ds.name;
                el.datasetSelect.appendChild(opt);
            }
        });
    } catch (err) {
        console.error('Failed to load datasets:', err);
    }
}

async function fetchDialogues(datasetId) {
    try {
        el.dialogueList.innerHTML = '<div class="list-item"><span class="item-id">Loading dialogues...</span></div>';
        const resp = await fetch(`/dialogues/${datasetId}`);
        state.dialogues = await resp.json();
        renderDialogueList();
    } catch (err) {
        console.error('Failed to load dialogues:', err);
    }
}

function renderDialogueList() {
    const query = el.dialogueSearch.value.toLowerCase().trim();
    const filtered = state.dialogues.filter(d => d.id.toLowerCase().includes(query));
    
    el.dialogueList.innerHTML = '';
    
    if (filtered.length === 0) {
        el.dialogueList.innerHTML = '<div class="list-item"><span class="item-id">No matches</span></div>';
        return;
    }
    
    filtered.forEach(d => {
        const item = document.createElement('div');
        item.className = 'list-item';
        if (d.id === state.selectedDialogueId) item.classList.add('selected');
        
        item.innerHTML = `
            <span class="item-id">${d.id}</span>
            <span class="item-desc">${d.turns} turns — ${d.first_line}</span>
        `;
        
        item.addEventListener('click', () => {
            document.querySelectorAll('#dialogue-list .list-item').forEach(li => li.classList.remove('selected'));
            item.classList.add('selected');
            loadDialogueDetails(d.id);
        });
        
        el.dialogueList.appendChild(item);
    });
}

async function loadDialogueDetails(dialogueId) {
    try {
        state.selectedDialogueId = dialogueId;
        const resp = await fetch(`/dialogue/${state.selectedDataset}/${dialogueId}`);
        state.dialogueContent = await resp.json();
        
        // Select target index if specified, otherwise the last turn
        let defaultTargetIdx = state.dialogueContent.target_index !== undefined ? 
            state.dialogueContent.target_index : 
            state.dialogueContent.utterances.length - 1;
            
        renderHistoryList(defaultTargetIdx);
        selectUtterance(defaultTargetIdx);
    } catch (err) {
        console.error('Failed to load dialogue content:', err);
    }
}

function renderHistoryList(targetIndex) {
    el.historyList.innerHTML = '';
    const utts = state.dialogueContent.utterances;
    
    // Set window bounds to highlight context
    const windowSize = parseInt(el.windowSlider.value);
    const startContext = Math.max(0, targetIndex - windowSize);
    
    utts.forEach((u, idx) => {
        const item = document.createElement('div');
        item.className = 'history-item';
        
        if (idx === targetIndex) {
            item.classList.add('target-selected');
        } else if (idx >= startContext && idx < targetIndex) {
            item.classList.add('context-selected');
        }
        
        const shortEmotion = unifyLabel(u.emotion);
        const emotionClass = EMOTION_CLASSES[shortEmotion.toLowerCase()] || '';
        
        item.innerHTML = `
            <span class="item-meta">${u.speaker} (${idx})</span>
            <span class="item-text" title="${u.text}">${u.text}</span>
            <span class="emotion-badge-gt ${emotionClass}">${shortEmotion}</span>
        `;
        
        item.addEventListener('click', () => {
            selectUtterance(idx);
        });
        
        el.historyList.appendChild(item);
    });
}

function selectUtterance(index) {
    state.selectedUtteranceIndex = index;
    const utts = state.dialogueContent.utterances;
    const u = utts[index];
    
    el.uttSpeaker.textContent = `Speaker: ${u.speaker} (Turn ${index})`;
    el.uttText.textContent = u.text;
    
    const uniGt = unifyLabel(u.emotion);
    el.uttGt.textContent = uniGt;
    // Clear old emotion classes and add new one
    el.uttGt.className = 'emotion-badge-gt';
    const emotionClass = EMOTION_CLASSES[uniGt.toLowerCase()];
    if (emotionClass) el.uttGt.classList.add(emotionClass);
    
    // Update active highlights in dialogue history
    renderHistoryList(index);
    
    // Source identifier
    const isMeld = state.selectedDataset.toLowerCase().includes('meld');
    el.sourceBadge.textContent = isMeld ? 'MELD' : 'IEMOCAP';
    
    // Media Player Rendering
    renderMediaPlayer(u, isMeld);
    
    // Update summary expectations
    el.evalGt.textContent = uniGt;
    el.evalGt.className = 'v-val badge-gt-big';
    if (emotionClass) el.evalGt.classList.add(emotionClass);
    
    el.evalPred.textContent = '-';
    el.evalPred.className = 'v-val badge-pred-big';
    
    el.evalMatchIndicator.className = 'verdict-match';
    el.evalMatchIndicator.innerHTML = '<span class="match-icon-placeholder">-</span>';
    el.evalStatusText.textContent = 'Ready';
    el.evalJsd.textContent = '-';
    
    updateChart(null, u.soft_labels || null);
}

function renderMediaPlayer(utterance, isMeld) {
    const playerWrapper = el.mediaWrapper.querySelector('.player-container');
    playerWrapper.innerHTML = '';
    
    const videoPath = utterance.video_path;
    const audioPath = utterance.audio_path;
    
    if (!videoPath && !audioPath) {
        playerWrapper.innerHTML = '<div class="media-placeholder">No media paths available</div>';
        return;
    }
    
    if (isMeld) {
        // MELD has native MP4 video clips
        if (videoPath) {
            const videoUrl = '/' + videoPath;
            playerWrapper.innerHTML = `
                <video src="${videoUrl}" controls preload="metadata" playsinline>
                    Your browser does not support HTML5 video playing.
                </video>
            `;
        } else {
            playerWrapper.innerHTML = '<div class="media-placeholder">Video path missing</div>';
        }
    } else {
        // IEMOCAP has WAV audio sentences and AVI video dialogues
        // Since browsers don't play AVI natively, we offer the WAV audio player, which is excellent.
        if (audioPath) {
            const audioUrl = '/' + audioPath;
            playerWrapper.innerHTML = `
                <div class="video-fallback-msg">
                    <p>Playing acoustic audio segment (WAV waveform):</p>
                    <audio src="${audioUrl}" controls preload="metadata"></audio>
                    <p class="subtitle" style="margin-top: 10px;">Note: Video dialogue resides in <code>.avi</code> format which is not supported natively by browsers.</p>
                </div>
            `;
        } else {
            playerWrapper.innerHTML = '<div class="media-placeholder">Acoustic WAV path missing</div>';
        }
    }
}

// ── Configurations Handling ──────────────────────────────────────────────────
function updateEmotionLabels() {
    const dsLower = state.selectedDataset.toLowerCase();
    if (dsLower.includes('meld') || dsLower.includes('camer')) {
        state.emotionLabels = EMOTION_LABELS_MELD;
    } else if (dsLower.includes('6class')) {
        state.emotionLabels = EMOTION_LABELS_IEMOCAP_6CLASS;
    } else {
        state.emotionLabels = EMOTION_LABELS_IEMOCAP;
    }
}

function getSelectedLabelMode() {
    const el = document.querySelector('input[name="label_mode"]:checked');
    return el ? el.value : 'soft';
}

function setModelId(id) {
    el.modelIdInput.value = id;
}

function toggleAdvancedConfig() {
    const wrapper = document.querySelector('.collapsible-section');
    wrapper.classList.toggle('open');
}


function updateConfigPresets() {
    const modality = el.modelPresetSelect.value;
    const labelMode = getSelectedLabelMode();
    
    // Auto-update template settings based on thesis baselines
    if (modality === 'mono_t') {
        el.modelIdInput.value = 'google/gemini-2.5-flash-lite';
        // Check text templates
        if (labelMode === 'soft') {
            setTemplatePreset('erc_cot_soft_label');
        } else {
            setTemplatePreset('erc_cot');
        }
    } else if (modality === 'mono_v') {
        el.modelIdInput.value = 'google/gemini-2.5-flash-lite';
        if (labelMode === 'soft') {
            setTemplatePreset('vision_only_soft_label_cot');
        } else {
            setTemplatePreset('vision_only_cot');
        }
    } else if (modality === 'mono_a') {
        el.modelIdInput.value = 'google/gemini-2.5-flash-lite';
        if (labelMode === 'soft') {
            setTemplatePreset('audio_only_soft_label_cot');
        } else {
            setTemplatePreset('audio_only');
        }
    } else if (modality === 'mono_tva') {
        el.modelIdInput.value = 'google/gemini-2.5-flash-lite';
        if (labelMode === 'soft') {
            setTemplatePreset('erc_multimodal_soft_label_cot');
        } else {
            setTemplatePreset('erc_multimodal_cot');
        }
    } else if (modality === 'agentic') {
        el.modelIdInput.value = 'google/gemini-2.5-flash-lite';
        // Agentic workflow selection happens in the execution payload
    }
}

let activeTemplatePreset = 'erc_default';
function setTemplatePreset(name) {
    activeTemplatePreset = name;
}

// Helper to map emotion shorthands (especially from IEMOCAP)
function unifyLabel(label) {
    if (!label) return '';
    const map = {
        "neu": "neutral",
        "hap": "happiness",
        "sad": "sadness",
        "ang": "anger",
        "fru": "frustration",
        "exc": "excitement",
        "fea": "fear",
        "sur": "surprise",
        "dis": "disgust",
        "oth": "other"
    };
    const low = label.toLowerCase().trim();
    return map[low] || low;
}

// ── Streaming Inference Orchestration ─────────────────────────────────────────
async function runProbe() {
    if (state.selectedUtteranceIndex === -1) {
        alert('Please choose an utterance to probe first!');
        return;
    }
    
    // Terminate any active streams
    if (state.streamController) {
        state.streamController.abort();
    }
    
    clearConsole();
    el.runBtn.disabled = true;
    el.runBtn.textContent = 'Executing...';
    
    const datasetId = state.selectedDataset;
    const dialogueId = state.selectedDialogueId;
    const targetIdx = state.selectedUtteranceIndex;
    
    const modality = el.modelPresetSelect.value;
    const labelMode = getSelectedLabelMode();
    const provider = el.providerSelect.value;
    const modelId = el.modelIdInput.value;
    const windowSize = parseInt(el.windowSlider.value);
    const temp = parseFloat(el.tempSlider.value);
    const disableThinking = el.disableThinkingCheck.checked;
    const frames = parseInt(el.framesInput.value);
    const maxTokens = el.maxTokensInput ? parseInt(el.maxTokensInput.value) : null;
    const reasoningMaxTokens = el.reasoningMaxTokensInput ? parseInt(el.reasoningMaxTokensInput.value) : null;
    
    const isAgentic = (modality === 'agentic');
    const isSoft = (labelMode === 'soft');
    
    state.streamController = new AbortController();
    
    try {
        // Auto-switch to the Real-time Stream Console tab
        const consoleTabBtn = document.querySelector('.main-tab-btn[data-tab="tab-console"]');
        if (consoleTabBtn) consoleTabBtn.click();

        if (isAgentic) {
            // MULTI-AGENT RESOLVER ROUTE
            const workflow = isSoft ? 'tva_theory_soft' : 'tva_theory';
            
            const payload = {
                dataset_name: datasetId,
                dialogue_id: dialogueId,
                target_index: targetIdx,
                model_id: modelId,
                provider: provider,
                window_size: windowSize,
                template_name: 'erc_default',
                workflow: workflow,
                vision_frames: frames,
                soft_label: isSoft,
                max_tokens: maxTokens,
                reasoning_max_tokens: reasoningMaxTokens
            };
            
            const resp = await fetch('/agent/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
                signal: state.streamController.signal
            });
            
            await readAgentStream(resp.body);
            
        } else {
            // SINGLE MODALITY / MONOLITHIC ROUTE
            const payload = {
                dataset_name: datasetId,
                dialogue_id: dialogueId,
                target_index: targetIdx,
                model_id: modelId,
                provider: provider,
                window_size: windowSize,
                soft_label: isSoft,
                stream: true,
                template_name: activeTemplatePreset,
                disable_thinking: disableThinking,
                include_video: (modality === 'mono_v' || modality === 'mono_tva'),
                include_audio: (modality === 'mono_a' || modality === 'mono_tva'),
                vision_frames: frames,
                max_tokens: maxTokens,
                reasoning_max_tokens: reasoningMaxTokens
            };
            
            const resp = await fetch('/inference', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
                signal: state.streamController.signal
            });
            
            await readMonolithicStream(resp.body);
        }
    } catch (err) {
        if (err.name !== 'AbortError') {
            console.error(err);
            el.evalStatusText.textContent = 'Execution Error';
        }
    } finally {
        el.runBtn.disabled = false;
        el.runBtn.textContent = 'Run Thesis Probe';
        state.streamController = null;
    }
}

// ── Stream Readers ───────────────────────────────────────────────────────────
function getOrCreateAgentTab(agentName) {
    let tab = el.agentTabsBar.querySelector(`.agent-tab-btn[data-agent="${agentName}"]`);
    let pane = el.agentPanesContainer.querySelector(`.agent-pane[data-agent="${agentName}"]`);
    
    if (!tab) {
        // Clear placeholder message if it exists
        const placeholder = el.agentPanesContainer.querySelector('.console-placeholder-msg');
        if (placeholder) placeholder.remove();
        
        // Create tab button
        tab = document.createElement('button');
        tab.className = 'agent-tab-btn';
        tab.setAttribute('data-agent', agentName);
        tab.innerHTML = `<span class="status-dot"></span> <span>${agentName}</span>`;
        el.agentTabsBar.appendChild(tab);
        
        // Create pane
        pane = document.createElement('div');
        pane.className = 'agent-pane hidden';
        pane.setAttribute('data-agent', agentName);
        pane.innerHTML = `
            <div class="sub-tabs-bar">
                <button class="sub-tab-btn active" data-subtab="response">Streaming Response & Thoughts</button>
                <button class="sub-tab-btn" data-subtab="prompt">Prompt Context Sent to LLM</button>
            </div>
            <div class="sub-tab-content" data-subtab="response">
                <div class="agent-box-response-body"></div>
            </div>
            <div class="sub-tab-content hidden" data-subtab="prompt">
                <pre class="agent-box-prompt-body">Loading prompt...</pre>
            </div>
        `;
        el.agentPanesContainer.appendChild(pane);
        
        // Bind sub-tab click events
        pane.querySelectorAll('.sub-tab-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const subtabVal = btn.getAttribute('data-subtab');
                pane.querySelectorAll('.sub-tab-btn').forEach(b => b.classList.remove('active'));
                pane.querySelectorAll('.sub-tab-content').forEach(c => c.classList.add('hidden'));
                
                btn.classList.add('active');
                pane.querySelector(`.sub-tab-content[data-subtab="${subtabVal}"]`).classList.remove('hidden');
            });
        });
        
        // Bind click event to tab button
        tab.addEventListener('click', () => {
            selectAgentTab(agentName);
        });
        
        // If this is the first tab, activate it
        if (el.agentTabsBar.querySelectorAll('.agent-tab-btn').length === 1) {
            selectAgentTab(agentName);
        }
    }
    
    return {
        tab: tab,
        pane: pane,
        promptBody: pane.querySelector('.agent-box-prompt-body'),
        responseBody: pane.querySelector('.agent-box-response-body')
    };
}

function selectAgentTab(agentName) {
    // Deactivate all agent tabs and panes
    el.agentTabsBar.querySelectorAll('.agent-tab-btn').forEach(btn => btn.classList.remove('active'));
    el.agentPanesContainer.querySelectorAll('.agent-pane').forEach(p => p.classList.add('hidden'));
    
    // Activate target agent tab and pane
    const tab = el.agentTabsBar.querySelector(`.agent-tab-btn[data-agent="${agentName}"]`);
    const pane = el.agentPanesContainer.querySelector(`.agent-pane[data-agent="${agentName}"]`);
    if (tab) tab.classList.add('active');
    if (pane) pane.classList.remove('hidden');
}

async function readMonolithicStream(bodyStream) {
    const reader = bodyStream.getReader();
    const decoder = new TextDecoder();
    
    let buffer = '';
    let systemPrompt = '';
    let userPrompt = '';
    
    let promptParsed = false;
    let inThinking = false;
    
    const agentName = 'Monolithic Model';
    const agentObj = getOrCreateAgentTab(agentName);
    
    agentObj.tab.classList.add('active-streaming');
    agentObj.responseBody.textContent = '';
    
    while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        buffer += decoder.decode(value, { stream: true });
        
        // Extract prompts if present
        if (!promptParsed) {
            const systemMatch = buffer.match(/<system>(.*?)<\/system>/s);
            if (systemMatch) {
                systemPrompt = systemMatch[1];
                buffer = buffer.replace(/<system>.*?<\/system>/s, '');
            }
            const userMatch = buffer.match(/<prompt>(.*?)<\/prompt>/s);
            if (userMatch) {
                userPrompt = userMatch[1];
                buffer = buffer.replace(/<prompt>.*?<\/prompt>/s, '');
                promptParsed = true;
                
                // Update prompt box
                agentObj.promptBody.textContent = `[SYSTEM INSTRUCTIONS]:\n${systemPrompt.trim()}\n\n[USER DIALOGUE PAYLOAD]:\n${userPrompt.trim()}`;
            }
        }
        
        if (promptParsed) {
            // Process tags within response
            while (true) {
                if (!inThinking) {
                    const idx = buffer.indexOf('<thought>');
                    if (idx === -1) {
                        // Safely print answer characters
                        const clean = buffer;
                        buffer = '';
                        if (clean) {
                            agentObj.responseBody.textContent += clean;
                            agentObj.responseBody.scrollTop = agentObj.responseBody.scrollHeight;
                        }
                        break;
                    } else {
                        // Flush text before thought
                        const before = buffer.substring(0, idx);
                        if (before) agentObj.responseBody.textContent += before;
                        buffer = buffer.substring(idx + '<thought>'.length);
                        inThinking = true;
                        
                        agentObj.responseBody.textContent += "\n[REASONING PATH]:\n";
                    }
                } else {
                    const idx = buffer.indexOf('</thought>');
                    if (idx === -1) {
                        // Append to thought
                        const clean = buffer;
                        buffer = '';
                        if (clean) {
                            agentObj.responseBody.textContent += clean;
                            agentObj.responseBody.scrollTop = agentObj.responseBody.scrollHeight;
                        }
                        break;
                    } else {
                        const inBetween = buffer.substring(0, idx);
                        if (inBetween) agentObj.responseBody.textContent += inBetween;
                        buffer = buffer.substring(idx + '</thought>'.length);
                        inThinking = false;
                        agentObj.responseBody.textContent += "\n\n[FINAL RESPONSE]:\n";
                    }
                }
            }
        }
    }
    
    // Flush remaining buffer
    if (buffer) {
        agentObj.responseBody.textContent += buffer;
    }
    
    agentObj.tab.classList.remove('active-streaming');
    agentObj.tab.classList.add('done-streaming');
    
    evaluateSingleRunOutput(agentObj.responseBody.textContent);
}

async function readAgentStream(bodyStream) {
    const reader = bodyStream.getReader();
    const decoder = new TextDecoder();
    
    let buffer = '';
    
    while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        buffer += decoder.decode(value, { stream: true });
        
        const lines = buffer.split('\n');
        // Keep the last partial line in buffer
        buffer = lines.pop();
        
        for (const line of lines) {
            if (!line.trim()) continue;
            try {
                const ev = JSON.parse(line);
                const agent = ev.agent;
                const event = ev.event;
                
                if (agent) {
                    const agentObj = getOrCreateAgentTab(agent);
                    
                    if (event === 'prompt') {
                        // Set active and inject prompts
                        selectAgentTab(agent);
                        
                        agentObj.tab.classList.add('active-streaming');
                        
                        const msgs = ev.messages || [];
                        const systemMsg = msgs.find(m => m.role === 'system');
                        const userMsg = msgs.find(m => m.role === 'user');
                        let promptText = '';
                        if (systemMsg) promptText += `[SYSTEM INSTRUCTIONS]:\n${systemMsg.content.trim()}\n\n`;
                        if (userMsg) {
                            let userContent = userMsg.content;
                            if (Array.isArray(userContent)) {
                                // Extract text from multimodal payload
                                const txtItem = userContent.find(i => i.type === 'text');
                                userContent = txtItem ? txtItem.text : JSON.stringify(userContent);
                            }
                            promptText += `[USER DIALOGUE PAYLOAD]:\n${userContent.trim()}`;
                        }
                        agentObj.promptBody.textContent = promptText || '(No prompt context)';
                    } else if (event === 'chunk') {
                        // Append text
                        agentObj.responseBody.textContent += ev.text;
                        agentObj.responseBody.scrollTop = agentObj.responseBody.scrollHeight;
                    } else if (event === 'done') {
                        agentObj.tab.classList.remove('active-streaming');
                        agentObj.tab.classList.add('done-streaming');
                    }
                }
                
                if (event === 'final') {
                    // Update final labels
                    const pred = unifyLabel(ev.label);
                    const gt = unifyLabel(state.dialogueContent.utterances[state.selectedUtteranceIndex].emotion);
                    
                    el.evalPred.textContent = pred;
                    el.evalPred.className = 'v-val badge-pred-big';
                    const predClass = EMOTION_CLASSES[pred.toLowerCase()];
                    if (predClass) el.evalPred.classList.add(predClass);
                    
                    const isMatch = (pred.toLowerCase() === gt.toLowerCase());
                    el.evalMatchIndicator.className = isMatch ? 'verdict-match match' : 'verdict-match mismatch';
                    el.evalMatchIndicator.innerHTML = isMatch ? '✓' : '✗';
                    el.evalStatusText.textContent = isMatch ? 'Match' : 'Mismatch';
                    
                    if (ev.soft_labels) {
                        // It was a soft resolver workflow
                        const parsedSoft = ev.soft_labels;
                        const gtSoft = state.dialogueContent.utterances[state.selectedUtteranceIndex].soft_labels || null;
                        const jsd = calculateJSD(parsedSoft, gtSoft || {});
                        el.evalJsd.textContent = jsd.toFixed(4);
                        updateChart(parsedSoft, gtSoft);
                    } else {
                        el.evalJsd.textContent = 'N/A (Hard)';
                        updateChart(null, null);
                    }
                }
            } catch (err) {
                console.warn('NDJSON line parse failed:', err, line);
            }
        }
    }
}

function clearConsole() {
    el.agentTabsBar.innerHTML = '';
    el.agentPanesContainer.innerHTML = '<div class="console-placeholder-msg">Console ready. Select configurations and click Run.</div>';
}

// ── Metrics Parser & Charting ────────────────────────────────────────────────
function evaluateSingleRunOutput(rawText) {
    const gt = unifyLabel(state.dialogueContent.utterances[state.selectedUtteranceIndex].emotion);
    const isSoft = (getSelectedLabelMode() === 'soft');
    
    let pred = '';
    let predSoft = null;
    
    if (isSoft) {
        predSoft = parseSoftLabels(rawText, state.emotionLabels);
        if (predSoft) {
            pred = getArgmaxLabel(predSoft);
        } else {
            pred = normalizePrediction(rawText, state.emotionLabels);
        }
    } else {
        pred = normalizePrediction(rawText, state.emotionLabels);
    }
    
    pred = unifyLabel(pred);
    
    el.evalPred.textContent = pred || 'Unknown';
    el.evalPred.className = 'v-val badge-pred-big';
    const predClass = EMOTION_CLASSES[pred.toLowerCase()];
    if (predClass) el.evalPred.classList.add(predClass);
    
    const isMatch = (pred.toLowerCase() === gt.toLowerCase());
    el.evalMatchIndicator.className = isMatch ? 'verdict-match match' : 'verdict-match mismatch';
    el.evalMatchIndicator.innerHTML = isMatch ? '✓' : '✗';
    el.evalStatusText.textContent = isMatch ? 'Match' : 'Mismatch';
    
    if (isSoft && predSoft) {
        const gtSoft = state.dialogueContent.utterances[state.selectedUtteranceIndex].soft_labels || null;
        const jsd = calculateJSD(predSoft, gtSoft || {});
        el.evalJsd.textContent = jsd.toFixed(4);
        updateChart(predSoft, gtSoft);
    } else {
        el.evalJsd.textContent = 'N/A';
        updateChart(null, null);
    }
}

// JS client side parser replicating backend utils.py
function normalizePrediction(raw, validLabels) {
    const rawClean = raw.trim();
    
    // 1. Check "Emotion: <label>"
    const match = rawClean.match(/Emotion:\s*[*_`"']*([a-zA-Z]+)/i);
    if (match) {
        const candidate = match[1].toLowerCase();
        const found = validLabels.find(l => l.toLowerCase() === candidate);
        if (found) return found;
    }
    
    // 2. Check last line
    const lines = rawClean.split('\n').map(l => l.trim()).filter(l => l);
    if (lines.length > 0) {
        const lastLine = lines[lines.length - 1].replace(/[*_`"'.]/g, '').toLowerCase();
        const found = validLabels.find(l => l.toLowerCase() === lastLine);
        if (found) return found;
    }
    
    // 3. Containment matching
    const rawLower = rawClean.replace(/[*_`"'.]/g, '').toLowerCase();
    for (const label of validLabels) {
        const regex = new RegExp(`\\b${label.toLowerCase()}\\b`, 'i');
        if (regex.test(rawLower)) return label;
    }
    
    return rawClean.substring(0, 15);
}

function parseSoftLabels(text, validLabels) {
    try {
        const cleanText = text.trim();
        let dictStr = "";
        
        const blockMatch = cleanText.match(/Soft\s*Labels\s*[:\s]*({.+?})/is);
        if (blockMatch) {
            dictStr = blockMatch[1];
        } else {
            const allBlocks = cleanText.match(/({.+?})/gs);
            if (allBlocks) {
                dictStr = allBlocks[allBlocks.length - 1];
            } else {
                dictStr = cleanText;
            }
        }
        
        let data = {};
        try {
            data = JSON.parse(dictStr);
        } catch (e) {
            const pairs = [...dictStr.matchAll(/['"]?([a-zA-Z]+)['"]?\s*:\s*(\d*\.?\d+)/g)];
            pairs.forEach(p => {
                data[p[1]] = parseFloat(p[2]);
            });
        }
        
        const result = {};
        let total = 0;
        
        validLabels.forEach(label => {
            const key = Object.keys(data).find(k => k.toLowerCase() === label.toLowerCase());
            const val = key !== undefined ? parseFloat(data[key]) : 0.0;
            result[label.toLowerCase()] = Math.max(0.0, isNaN(val) ? 0.0 : val);
            total += result[label.toLowerCase()];
        });
        
        if (total > 0) {
            for (let label in result) {
                result[label] /= total;
            }
            return result;
        }
        return null;
    } catch (e) {
        return null;
    }
}

function getArgmaxLabel(dist) {
    return Object.keys(dist).reduce((a, b) => dist[a] > dist[b] ? a : b);
}

function normalizeDistribution(dist, labels) {
    if (!dist) return null;
    const result = {};
    let total = 0.0;
    
    labels.forEach(label => {
        const key = Object.keys(dist).find(k => unifyLabel(k).toLowerCase() === label.toLowerCase());
        const val = key !== undefined ? parseFloat(dist[key]) : 0.0;
        result[label.toLowerCase()] = Math.max(0.0, isNaN(val) ? 0.0 : val);
        total += result[label.toLowerCase()];
    });
    
    if (total > 0) {
        for (let label in result) {
            result[label] /= total;
        }
        return result;
    }
    return null;
}

function calculateJSD(pDist, gDist) {
    const labels = state.emotionLabels;
    const pNorm = normalizeDistribution(pDist, labels);
    const gNorm = normalizeDistribution(gDist, labels);
    
    if (!pNorm || !gNorm) return 0.0;
    
    const p = labels.map(l => pNorm[l.toLowerCase()] || 0.0);
    const g = labels.map(l => gNorm[l.toLowerCase()] || 0.0);
    
    const m = p.map((x, idx) => 0.5 * (x + g[idx]));
    
    const kld = (a, b) => {
        return a.reduce((sum, val, idx) => {
            if (val > 0) {
                return sum + val * Math.log2(val / (b[idx] + 1e-12));
            }
            return sum;
        }, 0);
    };
    
    const jsd = 0.5 * kld(p, m) + 0.5 * kld(g, m);
    return Math.max(0.0, jsd);
}

// ── Chart.js Setup ───────────────────────────────────────────────────────────
function initChart() {
    const ctx = el.distributionChart.getContext('2d');
    state.chartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Model Prediction',
                    backgroundColor: 'rgba(124, 58, 237, 0.75)', // HSL violet
                    borderColor: 'rgb(124, 58, 237)',
                    borderWidth: 1,
                    data: []
                },
                {
                    label: 'Ground Truth',
                    backgroundColor: 'rgba(245, 158, 11, 0.5)', // HSL gold
                    borderColor: 'rgb(245, 158, 11)',
                    borderWidth: 1,
                    data: []
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: { color: '#e2e8f0', font: { family: 'Outfit', size: 10 } }
                }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(148, 163, 184, 0.05)' },
                    ticks: { color: '#94a3b8', font: { family: 'Outfit', size: 10 } }
                },
                y: {
                    grid: { color: 'rgba(148, 163, 184, 0.05)' },
                    ticks: { color: '#94a3b8', font: { family: 'Outfit', size: 10 } },
                    min: 0,
                    max: 1
                }
            }
        }
    });
}

function updateChart(predDist, gtDist) {
    if (!state.chartInstance) return;
    
    const labels = state.emotionLabels;
    state.chartInstance.data.labels = labels.map(l => l.charAt(0).toUpperCase() + l.slice(1));
    
    const normPred = predDist ? normalizeDistribution(predDist, labels) : null;
    const normGt = gtDist ? normalizeDistribution(gtDist, labels) : null;
    
    if (normPred) {
        state.chartInstance.data.datasets[0].data = labels.map(l => normPred[l.toLowerCase()] || 0.0);
    } else {
        state.chartInstance.data.datasets[0].data = [];
    }
    
    if (normGt) {
        state.chartInstance.data.datasets[1].data = labels.map(l => normGt[l.toLowerCase()] || 0.0);
    } else {
        state.chartInstance.data.datasets[1].data = [];
    }
    
    state.chartInstance.update();
}
