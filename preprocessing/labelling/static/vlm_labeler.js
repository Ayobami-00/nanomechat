// Global state
let currentContextData = null;
let selectedUserMessageIdx = null;
let selectedAssistantMessageIdx = null;
let stats = null;
let persona = null;
let contextWindowStart = 0;  // For load more functionality
let contextWindowEnd = 0;
let userCandidates = [];
let assistantCandidates = [];

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    // Get persona from stats API first
    fetch('/api/stats')
        .then(response => response.json())
        .then(data => {
            persona = data.persona;
            loadContextWindow();
            loadStats();
        })
        .catch(error => console.error('Error loading persona:', error));
    
    // Set up keyboard shortcuts
    document.addEventListener('keydown', handleKeyboardShortcut);
    
    // Refresh stats every 2 seconds
    setInterval(loadStats, 2000);
});

// Load context window and image
function loadContextWindow() {
    fetch('/api/vlm/context-window')
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                showCompletionMessage();
                return;
            }
            
            currentContextData = data;
            contextWindowStart = 0;
            contextWindowEnd = data.preceding.length + data.following.length;
            
            displayImage(data.image_message);
            displayUserMessages(data.preceding);
            displayAssistantMessages(data.following);
            updateProgress(data);
            
            // Reset selections
            selectedUserMessageIdx = null;
            selectedAssistantMessageIdx = null;
            clearCustomMessages();
            updatePreview();
        })
        .catch(error => console.error('Error loading context:', error));
}

// Display image
function displayImage(imageMessage) {
    const imageDisplay = document.getElementById('imageDisplay');
    const imageLoading = document.getElementById('imageLoading');
    
    if (imageMessage.image && persona) {
        const imageUrl = `/api/vlm/image/${persona}/${imageMessage.image}`;
        imageDisplay.src = imageUrl;
        imageDisplay.style.display = 'block';
        imageLoading.style.display = 'none';
        
        imageDisplay.onerror = function() {
            imageLoading.textContent = 'Failed to load image: ' + imageMessage.image;
            imageDisplay.style.display = 'none';
        };
    } else if (!imageMessage.image) {
        imageLoading.textContent = 'No image in this message';
    } else {
        imageLoading.textContent = 'Loading persona...';
    }
}

// Display user messages (preceding messages)
function displayUserMessages(precedingMessages) {
    const container = document.getElementById('userMessageList');
    
    // Filter to persona ("user") text messages only
    userCandidates = (precedingMessages || []).filter(m => {
        const role = (m.role || 'user').toLowerCase();
        return role === 'user' && m.content && m.content.trim().length > 0;
    });

    if (userCandidates.length === 0) {
        container.innerHTML = '<p style="color: #999; padding: 10px;">No preceding messages</p>';
        return;
    }
    
    let html = '';
    userCandidates.forEach((msg, index) => {
        const content = msg.content ? msg.content.substring(0, 150) : '(no content)';
        const isImage = msg.image ? ' [IMAGE]' : '';
        const isSelected = selectedUserMessageIdx === index ? 'selected' : '';
        const label = persona ? persona.toUpperCase() : 'USER';
        
        html += `
            <div class="message-item ${isSelected}" onclick="selectUserMessage(${index})" id="user-msg-${index}">
                <div class="message-item-role">${label}${isImage}</div>
                <div class="message-item-text">${escapeHtml(content)}${content.length > 150 ? '...' : ''}</div>
            </div>
        `;
    });
    
    container.innerHTML = html;
}

// Display assistant messages (following messages)
function displayAssistantMessages(followingMessages) {
    const container = document.getElementById('assistantMessageList');
    
    // Filter to YOUR reply ("assistant") text messages only
    assistantCandidates = (followingMessages || []).filter(m => {
        const role = (m.role || 'assistant').toLowerCase();
        return role === 'assistant' && m.content && m.content.trim().length > 0;
    });

    if (assistantCandidates.length === 0) {
        container.innerHTML = '<p style="color: #999; padding: 10px;">No following messages</p>';
        return;
    }
    
    let html = '';
    assistantCandidates.forEach((msg, index) => {
        const content = msg.content ? msg.content.substring(0, 150) : '(no content)';
        const isSelected = selectedAssistantMessageIdx === index ? 'selected' : '';
        const label = 'YOU';
        
        html += `
            <div class="message-item ${isSelected}" onclick="selectAssistantMessage(${index})" id="asst-msg-${index}">
                <div class="message-item-role">${label}</div>
                <div class="message-item-text">${escapeHtml(content)}${content.length > 150 ? '...' : ''}</div>
            </div>
        `;
    });
    
    container.innerHTML = html;
}

// Select user message
function selectUserMessage(index) {
    selectedUserMessageIdx = index;
    clearCustomMessages();
    displayUserMessages(currentContextData.preceding);
    updatePreview();
}

// Select assistant message
function selectAssistantMessage(index) {
    selectedAssistantMessageIdx = index;
    clearCustomMessages();
    displayAssistantMessages(currentContextData.following);
    updatePreview();
}

// Toggle custom user message input
function toggleCustomUserMessage() {
    const input = document.getElementById('customUserMessage');
    input.classList.toggle('active');
    if (input.classList.contains('active')) {
        input.focus();
        selectedUserMessageIdx = null;
        displayUserMessages(currentContextData.preceding);
    } else {
        input.value = '';
    }
    updatePreview();
}

// Toggle custom assistant message input
function toggleCustomAssistantMessage() {
    const input = document.getElementById('customAssistantMessage');
    input.classList.toggle('active');
    if (input.classList.contains('active')) {
        input.focus();
        selectedAssistantMessageIdx = null;
        displayAssistantMessages(currentContextData.following);
    } else {
        input.value = '';
    }
    updatePreview();
}

// Clear custom messages
function clearCustomMessages() {
    document.getElementById('customUserMessage').classList.remove('active');
    document.getElementById('customUserMessage').value = '';
    document.getElementById('customAssistantMessage').classList.remove('active');
    document.getElementById('customAssistantMessage').value = '';
}


// Update preview based on selected messages
function updatePreview() {
    const preview = document.getElementById('examplePreview');
    
    if (selectedUserMessageIdx === null && !document.getElementById('customUserMessage').value.trim() &&
        selectedAssistantMessageIdx === null && !document.getElementById('customAssistantMessage').value.trim()) {
        preview.innerHTML = '<p style="color: #999;">Select user message and assistant reply to preview</p>';
        return;
    }
    
    let html = '';
    
    // Add selected user message (from filtered candidates)
    if (selectedUserMessageIdx !== null && userCandidates[selectedUserMessageIdx]) {
        const msg = userCandidates[selectedUserMessageIdx];
        const content = msg.content || '(no content)';
        html += `
            <div class="preview-message user">
                <div class="preview-role">User</div>
                <div class="preview-content">${escapeHtml(content)}</div>
            </div>
        `;
    } else if (document.getElementById('customUserMessage').value.trim()) {
        html += `
            <div class="preview-message user">
                <div class="preview-role">User (Custom)</div>
                <div class="preview-content">${escapeHtml(document.getElementById('customUserMessage').value)}</div>
            </div>
        `;
    }
    
    // Add image message
    const imageMsg = currentContextData.image_message;
    const imageRole = imageMsg.role || 'user';
    let imageContent = '';
    if (imageMsg.image) {
        imageContent += `<strong>[IMAGE: ${escapeHtml(imageMsg.image)}]</strong>`;
    }
    if (imageMsg.content) {
        imageContent += `<br>${escapeHtml(imageMsg.content)}`;
    }
    
    html += `
        <div class="preview-message image">
            <div class="preview-role">Image</div>
            <div class="preview-content">${imageContent}</div>
        </div>
    `;
    
    // Add selected assistant message (from filtered candidates)
    if (selectedAssistantMessageIdx !== null && assistantCandidates[selectedAssistantMessageIdx]) {
        const msg = assistantCandidates[selectedAssistantMessageIdx];
        const content = msg.content || '(no content)';
        html += `
            <div class="preview-message assistant">
                <div class="preview-role">Assistant</div>
                <div class="preview-content">${escapeHtml(content)}</div>
            </div>
        `;
    } else if (document.getElementById('customAssistantMessage').value.trim()) {
        html += `
            <div class="preview-message assistant">
                <div class="preview-role">Assistant (Custom)</div>
                <div class="preview-content">${escapeHtml(document.getElementById('customAssistantMessage').value)}</div>
            </div>
        `;
    }
    
    preview.innerHTML = html;
}

// Accept and save example
function acceptExample() {
    if (!currentContextData) {
        alert('No image loaded');
        return;
    }
    
    // Get user message (from filtered candidates)
    let userMessage = null;
    const customUserMsg = document.getElementById('customUserMessage').value.trim();
    if (customUserMsg) {
        userMessage = { content: customUserMsg };
    } else if (selectedUserMessageIdx !== null && userCandidates[selectedUserMessageIdx]) {
        userMessage = userCandidates[selectedUserMessageIdx];
    }
    
    // Get assistant message (from filtered candidates)
    let assistantMessage = null;
    const customAssistantMsg = document.getElementById('customAssistantMessage').value.trim();
    if (customAssistantMsg) {
        assistantMessage = { content: customAssistantMsg };
    } else if (selectedAssistantMessageIdx !== null && assistantCandidates[selectedAssistantMessageIdx]) {
        assistantMessage = assistantCandidates[selectedAssistantMessageIdx];
    }
    
    if (!userMessage || !userMessage.content) {
        alert('Please select or provide a user message');
        return;
    }
    
    if (!assistantMessage || !assistantMessage.content) {
        alert('Please select or provide an assistant message');
        return;
    }
    
    // Build messages array
    const messages = [];
    const imageMsg = currentContextData.image_message;
    
    // Add user message
    messages.push({
        role: 'user',
        content: [{ type: 'text', text: userMessage.content }]
    });
    
    // Add image message
    const imageContent = [];
    if (imageMsg.image) {
        imageContent.push({ type: 'image', image: imageMsg.image });
    }
    if (imageMsg.content) {
        imageContent.push({ type: 'text', text: imageMsg.content });
    }
    
    messages.push({
        role: 'user',
        content: imageContent.length > 0 ? imageContent : [{ type: 'text', text: '' }]
    });
    
    // Add assistant message
    messages.push({
        role: 'assistant',
        content: [{ type: 'text', text: assistantMessage.content }]
    });
    
    // Save example
    fetch('/api/vlm/save-example', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            images: imageMsg.image ? [imageMsg.image] : [],
            messages: messages,
            timestamp: imageMsg.timestamp
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            selectedUserMessageIdx = null;
            selectedAssistantMessageIdx = null;
            clearCustomMessages();
            loadContextWindow();
            loadStats();
        } else {
            alert(data.error || 'Error saving example');
        }
    })
    .catch(error => console.error('Error saving example:', error));
}

// Skip image
function skipImage() {
    fetch('/api/vlm/skip-image', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            selectedUserMessageIdx = null;
            selectedAssistantMessageIdx = null;
            clearCustomMessages();
            loadContextWindow();
            loadStats();
        } else {
            alert(data.error || 'Error skipping image');
        }
    })
    .catch(error => console.error('Error skipping image:', error));
}

// Undo last example
function undoExample() {
    fetch('/api/vlm/undo', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            selectedUserMessageIdx = null;
            selectedAssistantMessageIdx = null;
            clearCustomMessages();
            loadContextWindow();
            loadStats();
        } else {
            alert(data.error || 'Nothing to undo');
        }
    })
    .catch(error => console.error('Error undoing example:', error));
}

// Update progress bar
function updateProgress(data) {
    const percent = data.current_index && data.total_images ? 
        (data.current_index / data.total_images * 100) : 0;
    document.getElementById('progressFill').style.width = percent + '%';
    document.getElementById('progressText').textContent = 
        `${data.current_index} / ${data.total_images} images (${Math.round(percent)}%)`;
}

// Load statistics
function loadStats() {
    fetch('/api/stats')
        .then(response => response.json())
        .then(data => {
            stats = data;
            document.getElementById('labeledCount').textContent = data.labeled_conversations;
            
            // Update progress
            if (data.total_messages > 0) {
                const percent = (data.current_index / data.total_messages) * 100;
                document.getElementById('progressFill').style.width = percent + '%';
                document.getElementById('progressText').textContent = 
                    `${data.current_index} / ${data.total_messages} images (${Math.round(percent)}%)`;
            }
        })
        .catch(error => console.error('Error loading stats:', error));
}

// Handle keyboard shortcuts
function handleKeyboardShortcut(event) {
    // Don't trigger if user is typing in textarea
    if (event.target.tagName === 'TEXTAREA') {
        return;
    }
    
    switch(event.key.toLowerCase()) {
        case 'enter':
            event.preventDefault();
            acceptExample();
            break;
        case 's':
            event.preventDefault();
            skipImage();
            break;
        case 'u':
            event.preventDefault();
            undoExample();
            break;
    }
}

// Show completion message
function showCompletionMessage() {
    const mainContent = document.querySelector('.vlm-main');
    mainContent.innerHTML = `
        <div style="text-align: center; padding: 40px; grid-column: 1 / -1;">
            <h2 style="color: #4caf50; margin-bottom: 20px;">✓ All Done!</h2>
            <p style="font-size: 18px; color: #666; margin-bottom: 20px;">All images have been labeled!</p>
            <p style="font-size: 16px; color: #333;">
                Total Examples Labeled: <strong id="finalCount">0</strong>
            </p>
            <form method="POST" action="/logout" style="margin-top: 30px;">
                <button type="submit" class="btn btn-primary">Logout</button>
            </form>
        </div>
    `;
    
    // Disable all buttons
    document.getElementById('acceptBtn').disabled = true;
    document.getElementById('skipBtn').disabled = true;
    document.getElementById('undoBtn').disabled = true;
    
    // Update final count
    if (stats) {
        document.getElementById('finalCount').textContent = stats.labeled_conversations;
    }
}

// Utility function to escape HTML
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
