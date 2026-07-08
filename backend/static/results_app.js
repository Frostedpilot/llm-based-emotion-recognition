// backend/static/results_app.js

// ── Application State ────────────────────────────────────────────────────────
const state = {
    resultsFiles: [],
    selectedFilename: '',
    selectedFileData: null,
    counterpartFileData: null,
    filteredResults: [],
    selectedResultIndex: -1,
    activeFilter: 'all',
    searchQuery: '',
    dialogueContent: null,
    gatheredAgents: {},
    emotionLabels: []
};

// Chart.js instance for comparison
let compareChartInstance = null;

// Emotion labels mapping for styling HSL colors matching results_style.css
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

// ── DOM Elements ─────────────────────────────────────────────────────────────
const el = {
    resultsFileSelect: document.getElementById('results-file-select'),
    resultsSearch: document.getElementById('results-search'),
    resultsList: document.getElementById('results-list'),
    resultsHistoryList: document.getElementById('results-history-list'),
    
    // Utterance detail nodes
    uttSpeaker: document.getElementById('results-utt-speaker'),
    uttText: document.getElementById('results-utt-text'),
    uttGt: document.getElementById('results-utt-gt'),
    uttPred: document.getElementById('results-utt-pred'),
    evalMatchIndicator: document.getElementById('results-eval-match-indicator'),
    sourceBadge: document.getElementById('results-source-badge'),
    mediaWrapper: document.getElementById('results-media-wrapper'),
    
    // Layout and output containers
    agentTabsBar: document.getElementById('results-agent-tabs-bar'),
    agentPanesContainer: document.getElementById('results-agent-panes-container'),
    rawResponse: document.getElementById('results-raw-response'),
    rawPrompt: document.getElementById('results-raw-prompt'),
    runMetaSummary: document.getElementById('run-meta-summary'),
    metaBadges: document.getElementById('meta-badges')
};

// ── Initialization ───────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
    fetchResultsFiles();
});

function setupEventListeners() {
    // Dropdown change
    el.resultsFileSelect.addEventListener('change', (e) => {
        state.selectedFilename = e.target.value;
        loadResultsFile(state.selectedFilename);
    });
    
    // Search input
    el.resultsSearch.addEventListener('input', () => {
        state.searchQuery = el.resultsSearch.value.toLowerCase().trim();
        renderResultsList();
    });
    
    // Filter buttons click
    document.querySelectorAll('.filter-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            state.activeFilter = btn.getAttribute('data-filter');
            renderResultsList();
        });
    });
    
    // Bind comparison filter change
    const compFilter = document.getElementById('results-comparison-filter');
    if (compFilter) {
        compFilter.addEventListener('change', () => {
            renderResultsList();
        });
    }

    // Bind click for the static tabs
    el.agentTabsBar.querySelector('.agent-tab-btn[data-agent="final"]').addEventListener('click', () => {
        activateAgentTab('final');
    });

    const compareBtn = document.getElementById('tab-btn-compare');
    if (compareBtn) {
        compareBtn.addEventListener('click', () => {
            activateAgentTab('compare');
            // Render comparison distribution chart when tab is activated
            if (state.selectedResultIndex !== -1) {
                renderCounterpartComparison();
            }
        });
    }
    
    // Bind main response pane subtabs
    bindSubTabs(document.querySelector('.agent-pane[data-agent="final"]'));
    bindSubTabs(document.querySelector('.agent-pane[data-agent="compare"]'));
}

function bindSubTabs(pane) {
    if (!pane) return;
    pane.querySelectorAll('.sub-tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const subtabVal = btn.getAttribute('data-subtab');
            pane.querySelectorAll('.sub-tab-btn').forEach(b => b.classList.remove('active'));
            pane.querySelectorAll('.sub-tab-content').forEach(c => c.classList.add('hidden'));
            
            btn.classList.add('active');
            const targetContent = pane.querySelector(`.sub-tab-content[data-subtab="${subtabVal}"]`);
            if (targetContent) targetContent.classList.remove('hidden');
        });
    });
}

// ── API Functions ────────────────────────────────────────────────────────────
async function fetchResultsFiles() {
    try {
        const resp = await fetch('/results');
        state.resultsFiles = await resp.json();
        
        el.resultsFileSelect.innerHTML = '<option value="" disabled selected>Select a results file...</option>';
        state.resultsFiles.forEach(file => {
            const opt = document.createElement('option');
            opt.value = file;
            opt.textContent = file;
            el.resultsFileSelect.appendChild(opt);
        });
    } catch (err) {
        console.error('Failed to load results list:', err);
    }
}

async function loadResultsFile(filename) {
    try {
        el.resultsList.innerHTML = '<div class="list-placeholder">Loading file data...</div>';
        const resp = await fetch(`/results/${filename}`);
        state.selectedFileData = await resp.json();
        
        // Determine emotion set labels
        updateEmotionLabels();

        // Update header metadata display
        updateHeaderMeta();
        
        // Reset selections
        state.selectedResultIndex = -1;
        state.dialogueContent = null;
        state.gatheredAgents = {};
        state.counterpartFileData = null;
        
        // Hide comparison components initially
        document.getElementById('comparison-filter-wrapper').style.display = 'none';
        document.getElementById('tab-btn-compare').style.display = 'none';
        
        // Find counterpart file (soft vs hard)
        let counterpartFilename = '';
        if (filename.includes('_hard_full.json')) {
            counterpartFilename = filename.replace('_hard_full.json', '_full.json');
        } else if (filename.includes('_full.json')) {
            counterpartFilename = filename.replace('_full.json', '_hard_full.json');
        }
        
        // Check if counterpart exists in list
        if (counterpartFilename && state.resultsFiles.includes(counterpartFilename)) {
            try {
                const compResp = await fetch(`/results/${counterpartFilename}`);
                state.counterpartFileData = await compResp.json();
                
                // Show comparison elements
                document.getElementById('comparison-filter-wrapper').style.display = 'block';
                document.getElementById('tab-btn-compare').style.display = 'block';
            } catch (err) {
                console.error('Failed to load counterpart file:', err);
            }
        }
        
        // Render list
        renderResultsList();
    } catch (err) {
        console.error('Failed to load result file:', err);
        el.resultsList.innerHTML = '<div class="list-placeholder text-rose">Error loading file.</div>';
    }
}

function updateEmotionLabels() {
    if (!state.selectedFileData || !state.selectedFileData.meta) return;
    const dsLower = (state.selectedFileData.meta.dataset_file || '').toLowerCase();
    
    if (dsLower.includes('meld') || dsLower.includes('camer')) {
        state.emotionLabels = EMOTION_LABELS_MELD;
    } else if (dsLower.includes('6class')) {
        state.emotionLabels = EMOTION_LABELS_IEMOCAP_6CLASS;
    } else {
        state.emotionLabels = EMOTION_LABELS_IEMOCAP;
    }
}

function updateHeaderMeta() {
    if (!state.selectedFileData || !state.selectedFileData.meta) return;
    const meta = state.selectedFileData.meta;
    
    el.runMetaSummary.innerHTML = `
        <strong>Model:</strong> ${meta.model || 'N/A'} &nbsp;|&nbsp; 
        <strong>Dataset:</strong> ${meta.dataset_file || 'N/A'} &nbsp;|&nbsp; 
        <strong>Template:</strong> ${meta.template_path || 'N/A'}
    `;
    
    el.metaBadges.innerHTML = '';
    
    // Modality indicator badge
    const badgeMod = document.createElement('span');
    badgeMod.className = 'contrib-badge';
    const isMulti = meta.agent_mode === 'multi';
    badgeMod.textContent = isMulti ? 'Agentic Resolver (Multi)' : 'Single Modality';
    el.metaBadges.appendChild(badgeMod);
    
    if (meta.soft_label) {
        const badgeSoft = document.createElement('span');
        badgeSoft.className = 'contrib-badge';
        badgeSoft.textContent = 'Soft Label';
        el.metaBadges.appendChild(badgeSoft);
    } else if (state.selectedFilename.includes('hard')) {
        const badgeHard = document.createElement('span');
        badgeHard.className = 'contrib-badge';
        badgeHard.textContent = 'Hard Label';
        el.metaBadges.appendChild(badgeHard);
    }
}

function renderResultsList() {
    if (!state.selectedFileData || !state.selectedFileData.results) return;
    
    const results = state.selectedFileData.results;
    el.resultsList.innerHTML = '';
    
    // Build counterpart lookup dictionary
    const counterpartLookup = {};
    if (state.counterpartFileData && state.counterpartFileData.results) {
        state.counterpartFileData.results.forEach(res => {
            counterpartLookup[`${res.dialogue_id}_${res.utterance_id}`] = res;
        });
    }
    
    const compFilterVal = document.getElementById('results-comparison-filter') ? 
        document.getElementById('results-comparison-filter').value : 'all';
        
    const meta = state.selectedFileData.meta || {};
    const currentIsHard = (meta.agent_mode !== 'multi') && 
        ((meta.template_path && meta.template_path.includes('hard')) || 
         (state.selectedFilename && state.selectedFilename.includes('hard')));

    // Filter and search
    state.filteredResults = results.map((r, idx) => ({ ...r, originalIndex: idx })).filter(r => {
        const matchesSearch = r.dialogue_id.toLowerCase().includes(state.searchQuery) ||
                             r.text.toLowerCase().includes(state.searchQuery);
        if (!matchesSearch) return false;
                             
        const isCorrect = unifyLabel(r.prediction).toLowerCase() === unifyLabel(r.ground_truth).toLowerCase();
        
        // Primary filters
        if (state.activeFilter === 'correct') {
            if (!isCorrect) return false;
        } else if (state.activeFilter === 'incorrect') {
            if (isCorrect) return false;
        }
        
        // Hard/Soft Comparison Filters
        if (state.counterpartFileData && compFilterVal !== 'all') {
            const c = counterpartLookup[`${r.dialogue_id}_${r.utterance_id}`];
            if (c) {
                const isCounterpartCorrect = unifyLabel(c.prediction).toLowerCase() === unifyLabel(c.ground_truth).toLowerCase();
                
                // Identify correct flags
                const isSoftCorrect = currentIsHard ? isCounterpartCorrect : isCorrect;
                const isHardCorrect = currentIsHard ? isCorrect : isCounterpartCorrect;
                
                const softPred = currentIsHard ? unifyLabel(c.prediction) : unifyLabel(r.prediction);
                const hardPred = currentIsHard ? unifyLabel(r.prediction) : unifyLabel(c.prediction);
                
                if (compFilterVal === 'disagree') {
                    if (softPred.toLowerCase() === hardPred.toLowerCase()) return false;
                } else if (compFilterVal === 'soft_correct') {
                    if (!(isSoftCorrect && !isHardCorrect)) return false;
                } else if (compFilterVal === 'hard_correct') {
                    if (!(isHardCorrect && !isSoftCorrect)) return false;
                } else if (compFilterVal === 'both_correct') {
                    if (!(isSoftCorrect && isHardCorrect)) return false;
                } else if (compFilterVal === 'both_incorrect') {
                    if (isSoftCorrect || isHardCorrect) return false;
                }
            } else {
                // If counterpart entry not found, exclude from filters
                return false;
            }
        }
        
        return true;
    });
    
    if (state.filteredResults.length === 0) {
        el.resultsList.innerHTML = '<div class="list-placeholder">No matching results found</div>';
        return;
    }
    
    state.filteredResults.forEach(r => {
        const item = document.createElement('div');
        item.className = 'list-item';
        if (r.originalIndex === state.selectedResultIndex) item.classList.add('selected');
        
        const isCorrect = unifyLabel(r.prediction).toLowerCase() === unifyLabel(r.ground_truth).toLowerCase();
        const badgeClass = isCorrect ? 'correct' : 'incorrect';
        const badgeText = isCorrect ? 'Correct' : 'Incorrect';
        
        item.innerHTML = `
            <div style="display: flex; flex-direction: column; max-width: 75%;">
                <span class="item-id" title="${r.dialogue_id}">${r.dialogue_id} (Turn ${r.utterance_id})</span>
                <span class="item-desc" title="${r.speaker}: ${r.text}"><strong>${r.speaker}:</strong> ${r.text}</span>
            </div>
            <span class="status-badge ${badgeClass}">${badgeText}</span>
        `;
        
        item.addEventListener('click', () => {
            document.querySelectorAll('#results-list .list-item').forEach(li => li.classList.remove('selected'));
            item.classList.add('selected');
            selectResultEntry(r.originalIndex);
        });
        
        el.resultsList.appendChild(item);
    });
}

async function selectResultEntry(index) {
    state.selectedResultIndex = index;
    const resultEntry = state.selectedFileData.results[index];
    const meta = state.selectedFileData.meta;
    
    // Set Ground Truth and Prediction Text
    const uniGt = unifyLabel(resultEntry.ground_truth);
    const uniPred = unifyLabel(resultEntry.prediction);
    
    el.uttSpeaker.textContent = `Speaker: ${resultEntry.speaker} (Turn ${resultEntry.utterance_id})`;
    el.uttText.textContent = resultEntry.text;
    
    el.uttGt.textContent = uniGt;
    el.uttGt.className = 'emotion-badge-gt';
    const gtClass = EMOTION_CLASSES[uniGt.toLowerCase()];
    if (gtClass) el.uttGt.classList.add(gtClass);
    
    el.uttPred.textContent = uniPred;
    el.uttPred.className = 'emotion-badge-pred';
    const predClass = EMOTION_CLASSES[uniPred.toLowerCase()];
    if (predClass) el.uttPred.classList.add(predClass);
    
    const isMatch = uniGt.toLowerCase() === uniPred.toLowerCase();
    el.evalMatchIndicator.className = isMatch ? 'verdict-match match' : 'verdict-match mismatch';
    el.evalMatchIndicator.textContent = isMatch ? '✓ Correct' : '✗ Incorrect';
    
    // Load dataset context and media files
    const datasetId = (meta.dataset_file || '').replace('.json', '');
    el.sourceBadge.textContent = datasetId.toLowerCase().includes('meld') ? 'MELD' : 'IEMOCAP';
    
    // Clear old media/context
    el.mediaWrapper.querySelector('.player-container').innerHTML = '<div class="media-placeholder">Loading media...</div>';
    el.resultsHistoryList.innerHTML = '<div style="padding: 20px; color: var(--text-muted); text-align: center;">Loading context...</div>';
    
    try {
        const resp = await fetch(`/dialogue/${datasetId}/${resultEntry.dialogue_id}`);
        state.dialogueContent = await resp.json();
        
        // Find index of target utterance in original dialogue
        const targetUtteranceId = resultEntry.utterance_id;
        const targetIndex = state.dialogueContent.utterances.findIndex(u => u.utterance_id == targetUtteranceId);
        
        if (targetIndex !== -1) {
            const targetUtt = state.dialogueContent.utterances[targetIndex];
            
            // Render media
            renderMediaPlayer(targetUtt, datasetId.toLowerCase().includes('meld'));
            
            // Render context history
            renderHistoryList(targetIndex);
            
            // Load and preview prompt sent to the LLM
            loadPromptPreview('final', {
                dataset_name: datasetId,
                dialogue_id: resultEntry.dialogue_id,
                target_index: targetIndex,
                template_name: meta.template_path,
                window_size: meta.window || 5,
                soft_label: meta.soft_label || false
            });
            
            // Render final response raw text
            el.rawResponse.textContent = resultEntry.prediction_raw || resultEntry.prediction;
            
            // Gather specialist agents outputs if Resolver mode
            if (meta.agent_mode === 'multi') {
                await gatherSpecialistAgents(datasetId, meta.model, resultEntry.dialogue_id, targetUtteranceId, targetIndex);
            } else {
                resetAgentTabs();
            }
            
            // Render comparison data if counterpart exists
            if (state.counterpartFileData) {
                renderCounterpartComparison();
            }
        } else {
            console.error('Target index not found in original utterances.');
            el.mediaWrapper.querySelector('.player-container').innerHTML = '<div class="media-placeholder">Utterance index mismatch</div>';
            el.resultsHistoryList.innerHTML = '<div style="padding: 20px; color: var(--text-rose); text-align: center;">Utterance index mismatch</div>';
        }
    } catch (err) {
        console.error('Failed to load context dialogue details:', err);
        el.mediaWrapper.querySelector('.player-container').innerHTML = '<div class="media-placeholder">Failed to load media</div>';
        el.resultsHistoryList.innerHTML = '<div style="padding: 20px; color: var(--text-rose); text-align: center;">Failed to load history context</div>';
    }
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
    
    if (videoPath) {
        let videoUrl = '/' + videoPath;
        if (!isMeld && videoPath.endsWith('.avi') && audioPath) {
            videoUrl = '/' + audioPath.replace('.wav', '.mp4');
        }
        const audioUrl = audioPath ? '/' + audioPath : null;

        playerWrapper.innerHTML = `
            <div class="multimodal-media-container" style="display: flex; flex-direction: column; gap: 10px;">
                <video src="${videoUrl}" controls preload="metadata" playsinline style="width: 100%; border-radius: 8px;">
                    Your browser does not support HTML5 video playing.
                </video>
                ${(audioUrl && !isMeld) ? `
                    <div class="audio-segment-box" style="background: rgba(148, 163, 184, 0.05); padding: 10px; border-radius: 6px; border: 1px solid rgba(148, 163, 184, 0.1);">
                        <p style="font-size: 11px; color: #94a3b8; margin: 0 0 6px 0;">Sentence Audio Segment (WAV):</p>
                        <audio src="${audioUrl}" controls preload="metadata" style="width: 100%; height: 32px;"></audio>
                    </div>
                ` : ''}
            </div>
        `;
    } else if (audioPath) {
        const audioUrl = '/' + audioPath;
        playerWrapper.innerHTML = `
            <div class="video-fallback-msg">
                <p>Acoustic segment (WAV):</p>
                <audio src="${audioUrl}" controls preload="metadata" style="margin-top: 4px;"></audio>
            </div>
        `;
    }
}

function renderHistoryList(targetIndex) {
    el.resultsHistoryList.innerHTML = '';
    const utts = state.dialogueContent.utterances;
    
    const meta = state.selectedFileData.meta;
    const windowSize = meta.window || 5;
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
        
        el.resultsHistoryList.appendChild(item);
    });
}

async function loadPromptPreview(agentKey, payload) {
    const promptElement = agentKey === 'final' ? el.rawPrompt : document.getElementById(`prompt-${agentKey}`);
    if (promptElement) promptElement.textContent = 'Loading prompt preview...';
    
    try {
        const resp = await fetch('/render_prompt_preview', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const data = await resp.json();
        
        if (promptElement) {
            promptElement.textContent = `[SYSTEM INSTRUCTIONS]:\n${data.system.trim()}\n\n[USER DIALOGUE PAYLOAD]:\n${data.prompt.trim()}`;
        }
    } catch (err) {
        console.error('Failed to preview prompt:', err);
        if (promptElement) promptElement.textContent = 'Error rendering prompt preview.';
    }
}

async function gatherSpecialistAgents(datasetId, model, dialogueId, utteranceId, targetIndex) {
    resetAgentTabs();
    
    try {
        const resp = await fetch(`/results/gather_agents?dataset_file=${datasetId}.json&model=${model}&dialogue_id=${dialogueId}&utterance_id=${utteranceId}&current_filename=${state.selectedFilename}`);
        state.gatheredAgents = await resp.json();
        
        // Populate tabs
        Object.keys(state.gatheredAgents).forEach(agentName => {
            const agentData = state.gatheredAgents[agentName];
            
            // Create tab button
            const tabBtn = document.createElement('button');
            tabBtn.className = 'agent-tab-btn';
            tabBtn.setAttribute('data-agent', agentName);
            tabBtn.textContent = agentName;
            el.agentTabsBar.appendChild(tabBtn);
            
            // Create pane
            const pane = document.createElement('div');
            pane.className = 'agent-pane hidden';
            pane.setAttribute('data-agent', agentName);
            
            // Soft indicator badge
            const isSoft = agentData.template.includes('soft');
            
            pane.innerHTML = `
                <div class="sub-tabs-bar">
                    <button class="sub-tab-btn active" data-subtab="response">Agent Response (Raw)</button>
                    <button class="sub-tab-btn" data-subtab="prompt">Prompt Sent to LLM</button>
                    <span style="font-size: 11px; color: var(--text-muted); margin-left: auto; align-self: center;">Loaded from: <code>${agentData.filename}</code></span>
                </div>
                
                <div class="sub-tab-content" data-subtab="response">
                    <div class="response-box">
                        <div style="margin-bottom: 12px; display:flex; align-items:center; gap: 8px;">
                            <strong>Extracted Emotion:</strong> 
                            <span class="emotion-badge-pred ${EMOTION_CLASSES[unifyLabel(agentData.prediction).toLowerCase()] || ''}">${unifyLabel(agentData.prediction)}</span>
                        </div>
                        <pre id="response-${agentName}">${agentData.prediction_raw}</pre>
                    </div>
                </div>
                <div class="sub-tab-content hidden" data-subtab="prompt">
                    <div class="response-box">
                        <pre id="prompt-${agentName}">Loading agent prompt...</pre>
                    </div>
                </div>
            `;
            el.agentPanesContainer.appendChild(pane);
            
            // Bind tab listeners
            bindSubTabs(pane);
            
            tabBtn.addEventListener('click', () => {
                activateAgentTab(agentName);
                
                // Load prompt preview on demand
                const promptPre = document.getElementById(`prompt-${agentName}`);
                if (promptPre && promptPre.textContent.startsWith('Loading')) {
                    loadPromptPreview(agentName, {
                        dataset_name: datasetId,
                        dialogue_id: dialogueId,
                        target_index: targetIndex,
                        template_name: agentData.template,
                        window_size: 5,
                        soft_label: isSoft
                    });
                }
            });
        });
    } catch (err) {
        console.error('Failed to gather specialist agents outputs:', err);
    }
}

function resetAgentTabs() {
    // Remove all dynamically added tabs and panes except final and compare
    el.agentTabsBar.querySelectorAll('.agent-tab-btn').forEach(btn => {
        const ag = btn.getAttribute('data-agent');
        if (ag !== 'final' && ag !== 'compare') {
            btn.remove();
        }
    });
    
    el.agentPanesContainer.querySelectorAll('.agent-pane').forEach(p => {
        const ag = p.getAttribute('data-agent');
        if (ag !== 'final' && ag !== 'compare') {
            p.remove();
        }
    });
    
    // Toggle compare tab button
    if (state.counterpartFileData) {
        document.getElementById('tab-btn-compare').style.display = 'block';
    } else {
        document.getElementById('tab-btn-compare').style.display = 'none';
    }
    
    activateAgentTab('final');
}

function activateAgentTab(agentName) {
    el.agentTabsBar.querySelectorAll('.agent-tab-btn').forEach(btn => {
        if (btn.getAttribute('data-agent') === agentName) {
            btn.classList.add('active');
        } else {
            btn.classList.remove('active');
        }
    });
    
    el.agentPanesContainer.querySelectorAll('.agent-pane').forEach(pane => {
        if (pane.getAttribute('data-agent') === agentName) {
            pane.classList.remove('hidden');
        } else {
            pane.classList.add('hidden');
        }
    });
}

function renderCounterpartComparison() {
    if (state.selectedResultIndex === -1 || !state.counterpartFileData) return;
    
    const r = state.selectedFileData.results[state.selectedResultIndex];
    const meta = state.selectedFileData.meta || {};
    const currentIsHard = (meta.agent_mode !== 'multi') && 
        ((meta.template_path && meta.template_path.includes('hard')) || 
         (state.selectedFilename && state.selectedFilename.includes('hard')));
         
    // Find counterpart entry
    const c = state.counterpartFileData.results.find(res => 
        res.dialogue_id === r.dialogue_id && res.utterance_id == r.utterance_id
    );
    
    if (!c) return;
    
    const softEntry = currentIsHard ? c : r;
    const hardEntry = currentIsHard ? r : c;
    
    const softPred = unifyLabel(softEntry.prediction);
    const hardPred = unifyLabel(hardEntry.prediction);
    
    // Set labels
    document.getElementById('compare-soft-pred').textContent = softPred;
    document.getElementById('compare-soft-pred').className = 'emotion-badge-pred ' + (EMOTION_CLASSES[softPred.toLowerCase()] || '');
    
    document.getElementById('compare-hard-pred').textContent = hardPred;
    document.getElementById('compare-hard-pred').className = 'emotion-badge-pred ' + (EMOTION_CLASSES[hardPred.toLowerCase()] || '');
    
    const softJsd = softEntry.js_divergence !== undefined ? softEntry.js_divergence : null;
    if (softJsd !== null) {
        document.getElementById('compare-soft-jsd').textContent = parseFloat(softJsd).toFixed(4);
        document.getElementById('compare-soft-jsd-wrapper').style.display = 'block';
    } else {
        document.getElementById('compare-soft-jsd-wrapper').style.display = 'none';
    }
    
    const hardJsd = hardEntry.js_divergence !== undefined ? hardEntry.js_divergence : null;
    if (hardJsd !== null) {
        document.getElementById('compare-hard-jsd').textContent = parseFloat(hardJsd).toFixed(4);
        document.getElementById('compare-hard-jsd-wrapper').style.display = 'block';
    } else {
        document.getElementById('compare-hard-jsd-wrapper').style.display = 'none';
    }
    
    document.getElementById('compare-soft-raw').textContent = softEntry.prediction_raw || softEntry.prediction;
    document.getElementById('compare-hard-raw').textContent = hardEntry.prediction_raw || hardEntry.prediction;
    
    // Draw probability comparison chart
    const softDist = softEntry.prediction_soft || null;
    updateCompareChart(softDist, hardPred);
}

function updateCompareChart(softDistribution, hardLabel) {
    const canvas = document.getElementById('comparison-distribution-chart');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    
    if (compareChartInstance) {
        compareChartInstance.destroy();
    }
    
    const labels = state.emotionLabels;
    
    // Format soft dataset
    const softData = labels.map(l => {
        const key = l.toLowerCase();
        return (softDistribution && softDistribution[key] !== undefined) ? softDistribution[key] : 0.0;
    });
    
    // Format hard dataset (one-hot)
    const hardData = labels.map(l => {
        return l.toLowerCase() === hardLabel.toLowerCase() ? 1.0 : 0.0;
    });
    
    compareChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels.map(l => l.charAt(0).toUpperCase() + l.slice(1)),
            datasets: [
                {
                    label: 'Soft Probability',
                    data: softData,
                    backgroundColor: 'rgba(54, 162, 235, 0.5)',
                    borderColor: 'rgba(54, 162, 235, 1)',
                    borderWidth: 1
                },
                {
                    label: 'Hard One-Hot',
                    data: hardData,
                    backgroundColor: 'rgba(255, 99, 132, 0.5)',
                    borderColor: 'rgba(255, 99, 132, 1)',
                    borderWidth: 1
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 1.0,
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: 'rgba(255, 255, 255, 0.6)' }
                },
                x: {
                    grid: { display: false },
                    ticks: { color: 'rgba(255, 255, 255, 0.6)' }
                }
            },
            plugins: {
                legend: {
                    labels: { color: 'rgba(255, 255, 255, 0.8)', font: { size: 10 } }
                }
            }
        }
    });
}

// Helper mapping emotion shorthands (copied from app.js)
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
