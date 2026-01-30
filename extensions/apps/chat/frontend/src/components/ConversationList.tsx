'use client';

import { Conversation } from '@/types';

interface ConversationListProps {
  conversations: Conversation[];
  selectedId: number | null;
  onSelect: (id: number) => void;
  onNew: () => void;
  onDelete: (id: number) => void;
}

export default function ConversationList({
  conversations,
  selectedId,
  onSelect,
  onNew,
  onDelete,
}: ConversationListProps) {
  return (
    <div className="flex h-full flex-col bg-gray-900 text-white">
      {/* Header */}
      <div className="border-b border-gray-700 p-4">
        <button
          onClick={onNew}
          className="w-full rounded-lg border border-gray-600 px-4 py-2 text-sm hover:bg-gray-800"
        >
          + New Chat
        </button>
      </div>

      {/* Conversations */}
      <div className="flex-1 overflow-y-auto">
        {conversations.length === 0 ? (
          <div className="p-4 text-center text-gray-500">
            No conversations yet
          </div>
        ) : (
          conversations.map((conv) => (
            <div
              key={conv.id}
              className={`group flex items-center justify-between border-b border-gray-800 px-4 py-3 cursor-pointer hover:bg-gray-800 ${
                selectedId === conv.id ? 'bg-gray-800' : ''
              }`}
              onClick={() => onSelect(conv.id)}
            >
              <div className="flex-1 overflow-hidden">
                <p className="truncate text-sm">{conv.title}</p>
                <p className="text-xs text-gray-500">
                  {new Date(conv.updated_at).toLocaleDateString()}
                </p>
              </div>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  onDelete(conv.id);
                }}
                className="ml-2 hidden rounded p-1 text-gray-500 hover:bg-gray-700 hover:text-red-400 group-hover:block"
              >
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  className="h-4 w-4"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
                  />
                </svg>
              </button>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
