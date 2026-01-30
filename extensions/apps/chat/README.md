# Underthesea Chat

A chat application for AI Agent powered by Underthesea.

## Installation

```bash
npm install @undertheseanlp/chat
```

Or install globally:

```bash
npm install -g @undertheseanlp/chat
```

## Environment Variables

Set one of the following environment variable configurations:

### OpenAI
```bash
export OPENAI_API_KEY=sk-...
```

### Azure OpenAI
```bash
export AZURE_OPENAI_API_KEY=...
export AZURE_OPENAI_ENDPOINT=https://....openai.azure.com
```

## Usage

### CLI

```bash
# Start the chat app
underthesea-chat

# With custom ports
underthesea-chat --backend-port 8002 --frontend-port 3001
```

### npm scripts

```bash
cd extensions/apps/chat

# Install all dependencies
npm install

# Development mode (with hot reload)
npm run dev

# Production mode
npm start
```

This starts:
- Frontend at http://localhost:3000
- Backend API at http://localhost:8001

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/conversations` | List all conversations |
| POST | `/api/conversations` | Create new conversation |
| GET | `/api/conversations/{id}` | Get conversation with messages |
| DELETE | `/api/conversations/{id}` | Delete conversation |
| POST | `/api/conversations/{id}/chat` | Send message, get AI response |

## Architecture

```
chat/
├── bin/
│   └── cli.js            # CLI entry point
├── backend/              # Express.js backend
│   ├── package.json
│   └── src/
│       ├── index.js      # Express app
│       ├── database.js   # SQLite setup
│       └── agent.js      # OpenAI integration
└── frontend/             # Next.js frontend
    └── src/
        ├── app/          # Next.js App Router
        ├── components/
        ├── lib/          # API client
        └── types/
```
