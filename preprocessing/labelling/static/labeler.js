// Global state
let currentMessage = null;
let stats = null;

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    loadNextMessage();
    loadStats();
    updateConversationPreview();
    
    // Set up keyboard shortcuts
    document.addEventListener('keydown', handleKeyboardShortcut);
    
    // Refresh stats every 2 seconds
    setInterval(loadStats, 2000);
});

// Load next message
function loadNextMessage() {
    fetch('/api/next-message')
        .then(response => response.json())
        .then(data => {
            if (data.done) {
                showCompletionMessage(data);
            } else {
                currentMessage = data;
                displayMessage(data);
                updateProgress(data);
            }
        })
        .catch(error => console.error('Error loading message:', error));
}

// Display message in UI
function displayMessage(data) {
    const msg = data.message;
    document.getElementById('senderName').textContent = msg.sender;
    document.getElementById('messageTime').textContent = formatTime(msg.timestamp);
    document.getElementById('messageContent').textContent = msg.content;
}

// Format timestamp
function formatTime(timestamp) {
    try {
        const date = new Date(timestamp);
        return date.toLocaleString();
    } catch (e) {
        return timestamp;
    }
}

// Update progress bar
function updateProgress(data) {
    const percent = (data.index / data.total) * 100;
    document.getElementById('progressFill').style.width = percent + '%';
    document.getElementById('progressText').textContent = 
        `${data.index} / ${data.total} messages (${Math.round(percent)}%)`;
}

// Add message to conversation
function addToConversation() {
    if (!currentMessage) return;
    
    fetch('/api/add-to-conversation', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'}
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            updateConversationPreview();
            loadNextMessage();
        }
    })
    .catch(error => console.error('Error adding to conversation:', error));
}

// Skip message
function skipMessage() {
    if (!currentMessage) return;
    
    fetch('/api/skip-message', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'}
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            loadNextMessage();
        }
    })
    .catch(error => console.error('Error skipping message:', error));
}

// Undo last addition
function undo() {
    fetch('/api/undo', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'}
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            updateConversationPreview();
            loadNextMessage();
        }
    })
    .catch(error => console.error('Error undoing:', error));
}

// End conversation
function endConversation() {
    fetch('/api/end-conversation', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'}
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            updateConversationPreview();
            loadStats();
            loadNextMessage();
        } else {
            alert(data.error || 'Error ending conversation');
        }
    })
    .catch(error => console.error('Error ending conversation:', error));
}

// Update conversation preview
function updateConversationPreview() {
    fetch('/api/stats')
        .then(response => response.json())
        .then(data => {
            const preview = document.getElementById('conversationPreview');
            if (data.labeled_conversations === 0) {
                preview.innerHTML = '<p class="empty-state">No messages in conversation yet</p>';
            } else {
                // Show last few messages from current conversation
                preview.innerHTML = '<p class="empty-state">Current conversation in progress...</p>';
            }
            document.getElementById('convSize').textContent = 
                data.labeled_messages - (data.labeled_conversations > 0 ? 
                Math.floor(data.labeled_messages / data.labeled_conversations) : 0);
        });
}

// Load statistics
function loadStats() {
    fetch('/api/stats')
        .then(response => response.json())
        .then(data => {
            stats = data;
            document.getElementById('labeledCount').textContent = data.labeled_conversations;
            document.getElementById('labeledMessages').textContent = data.labeled_messages;
        })
        .catch(error => console.error('Error loading stats:', error));
}

// Handle keyboard shortcuts
function handleKeyboardShortcut(event) {
    // Don't trigger if user is typing in an input
    if (event.target.tagName === 'INPUT' || event.target.tagName === 'TEXTAREA') {
        return;
    }
    
    switch(event.key.toLowerCase()) {
        case 'a':
            event.preventDefault();
            addToConversation();
            break;
        case 's':
            event.preventDefault();
            skipMessage();
            break;
        case 'u':
            event.preventDefault();
            undo();
            break;
        case 'e':
            event.preventDefault();
            endConversation();
            break;
    }
}

// Show completion message
function showCompletionMessage(data) {
    const messageCard = document.getElementById('messageCard');
    messageCard.innerHTML = `
        <div style="text-align: center; padding: 40px;">
            <h2 style="color: #4caf50; margin-bottom: 20px;">✓ All Done!</h2>
            <p style="font-size: 18px; color: #666; margin-bottom: 20px;">${data.message}</p>
            <p style="font-size: 16px; color: #333;">
                Total Conversations Labeled: <strong>${data.total_labeled}</strong>
            </p>
            <form method="POST" action="/logout" style="margin-top: 30px;">
                <button type="submit" class="btn btn-primary">Logout</button>
            </form>
        </div>
    `;
    
    // Disable all buttons
    document.getElementById('addBtn').disabled = true;
    document.getElementById('skipBtn').disabled = true;
    document.getElementById('undoBtn').disabled = true;
    document.getElementById('endBtn').disabled = true;
}
