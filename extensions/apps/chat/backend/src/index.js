import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

// Load .env from project root
const __dirname = dirname(fileURLToPath(import.meta.url));
const envPath = join(__dirname, '..', '..', '..', '..', '..', '.env');
dotenv.config({ path: envPath });

import db from './database.js';
import { chat } from './agent.js';

const app = express();
const PORT = process.env.PORT || 8001;

app.use(cors());
app.use(express.json());

// List conversations
app.get('/api/conversations', (req, res) => {
  const conversations = db
    .prepare('SELECT * FROM conversations ORDER BY updated_at DESC')
    .all();
  res.json(conversations);
});

// Create conversation
app.post('/api/conversations', (req, res) => {
  const { title = 'New Conversation', system_prompt = 'You are a helpful assistant.' } = req.body;

  const result = db
    .prepare('INSERT INTO conversations (title, system_prompt) VALUES (?, ?)')
    .run(title, system_prompt);

  const conversation = db
    .prepare('SELECT * FROM conversations WHERE id = ?')
    .get(result.lastInsertRowid);

  res.json(conversation);
});

// Get conversation with messages
app.get('/api/conversations/:id', (req, res) => {
  const { id } = req.params;

  const conversation = db
    .prepare('SELECT * FROM conversations WHERE id = ?')
    .get(id);

  if (!conversation) {
    return res.status(404).json({ detail: 'Conversation not found' });
  }

  const messages = db
    .prepare('SELECT * FROM messages WHERE conversation_id = ? ORDER BY created_at')
    .all(id);

  res.json({ ...conversation, messages });
});

// Delete conversation
app.delete('/api/conversations/:id', (req, res) => {
  const { id } = req.params;

  const conversation = db
    .prepare('SELECT * FROM conversations WHERE id = ?')
    .get(id);

  if (!conversation) {
    return res.status(404).json({ detail: 'Conversation not found' });
  }

  db.prepare('DELETE FROM conversations WHERE id = ?').run(id);
  res.json({ message: 'Conversation deleted' });
});

// Chat - send message and get AI response
app.post('/api/conversations/:id/chat', async (req, res) => {
  const { id } = req.params;
  const { content } = req.body;

  const conversation = db
    .prepare('SELECT * FROM conversations WHERE id = ?')
    .get(id);

  if (!conversation) {
    return res.status(404).json({ detail: 'Conversation not found' });
  }

  // Save user message
  const userResult = db
    .prepare('INSERT INTO messages (conversation_id, role, content) VALUES (?, ?, ?)')
    .run(id, 'user', content);

  const userMessage = db
    .prepare('SELECT * FROM messages WHERE id = ?')
    .get(userResult.lastInsertRowid);

  // Get all messages for context
  const messages = db
    .prepare('SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY created_at')
    .all(id);

  try {
    // Get AI response
    const responseText = await chat(messages, conversation.system_prompt);

    // Save assistant message
    const assistantResult = db
      .prepare('INSERT INTO messages (conversation_id, role, content) VALUES (?, ?, ?)')
      .run(id, 'assistant', responseText);

    const assistantMessage = db
      .prepare('SELECT * FROM messages WHERE id = ?')
      .get(assistantResult.lastInsertRowid);

    // Update conversation title if first message
    const messageCount = db
      .prepare('SELECT COUNT(*) as count FROM messages WHERE conversation_id = ?')
      .get(id).count;

    if (messageCount === 2) {
      const title = content.substring(0, 50) + (content.length > 50 ? '...' : '');
      db.prepare('UPDATE conversations SET title = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?')
        .run(title, id);
    } else {
      db.prepare('UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE id = ?')
        .run(id);
    }

    res.json({
      user_message: userMessage,
      assistant_message: assistantMessage,
    });
  } catch (error) {
    res.status(500).json({ detail: `AI service error: ${error.message}` });
  }
});

// Root endpoint
app.get('/', (req, res) => {
  res.json({ message: 'Underthesea Chat API', docs: '/api' });
});

app.listen(PORT, () => {
  console.log(`Backend running on http://localhost:${PORT}`);
});
