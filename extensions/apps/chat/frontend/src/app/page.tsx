'use client';

import { useState, useEffect, useCallback } from 'react';
import { Conversation, ConversationDetail, Message } from '@/types';
import {
  getConversations,
  createConversation,
  getConversation,
  deleteConversation,
  sendMessage,
} from '@/lib/api';
import ConversationList from '@/components/ConversationList';
import ChatWindow from '@/components/ChatWindow';

export default function Home() {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [selectedConversation, setSelectedConversation] = useState<ConversationDetail | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadConversations = useCallback(async () => {
    try {
      const data = await getConversations();
      setConversations(data);
    } catch (err) {
      setError('Failed to load conversations');
      console.error(err);
    }
  }, []);

  useEffect(() => {
    loadConversations();
  }, [loadConversations]);

  const handleSelectConversation = async (id: number) => {
    try {
      const data = await getConversation(id);
      setSelectedConversation(data);
      setError(null);
    } catch (err) {
      setError('Failed to load conversation');
      console.error(err);
    }
  };

  const handleNewConversation = async () => {
    try {
      const newConv = await createConversation();
      await loadConversations();
      await handleSelectConversation(newConv.id);
    } catch (err) {
      setError('Failed to create conversation');
      console.error(err);
    }
  };

  const handleDeleteConversation = async (id: number) => {
    try {
      await deleteConversation(id);
      await loadConversations();
      if (selectedConversation?.id === id) {
        setSelectedConversation(null);
      }
    } catch (err) {
      setError('Failed to delete conversation');
      console.error(err);
    }
  };

  const handleSendMessage = async (content: string) => {
    if (!selectedConversation) return;

    setIsLoading(true);
    setError(null);

    // Optimistically add user message
    const tempUserMessage: Message = {
      id: Date.now(),
      role: 'user',
      content,
      created_at: new Date().toISOString(),
    };

    setSelectedConversation((prev) =>
      prev
        ? { ...prev, messages: [...prev.messages, tempUserMessage] }
        : null
    );

    try {
      const response = await sendMessage(selectedConversation.id, content);

      // Update with real messages
      setSelectedConversation((prev) => {
        if (!prev) return null;
        const messages = prev.messages.filter((m) => m.id !== tempUserMessage.id);
        return {
          ...prev,
          messages: [...messages, response.user_message, response.assistant_message],
        };
      });

      // Reload conversations to update titles
      await loadConversations();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to send message');
      // Remove optimistic message on error
      setSelectedConversation((prev) =>
        prev
          ? { ...prev, messages: prev.messages.filter((m) => m.id !== tempUserMessage.id) }
          : null
      );
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex h-screen">
      {/* Sidebar */}
      <div className="w-64 flex-shrink-0">
        <ConversationList
          conversations={conversations}
          selectedId={selectedConversation?.id ?? null}
          onSelect={handleSelectConversation}
          onNew={handleNewConversation}
          onDelete={handleDeleteConversation}
        />
      </div>

      {/* Main content */}
      <div className="flex flex-1 flex-col bg-gray-50">
        {error && (
          <div className="bg-red-100 border-b border-red-400 text-red-700 px-4 py-2 text-sm">
            {error}
            <button
              onClick={() => setError(null)}
              className="ml-2 font-bold"
            >
              &times;
            </button>
          </div>
        )}

        {selectedConversation ? (
          <ChatWindow
            messages={selectedConversation.messages}
            onSendMessage={handleSendMessage}
            isLoading={isLoading}
            conversationTitle={selectedConversation.title}
          />
        ) : (
          <div className="flex h-full items-center justify-center text-gray-500">
            <div className="text-center">
              <h2 className="text-xl font-semibold mb-2">Welcome to Underthesea Chat</h2>
              <p>Create a new conversation or select one from the sidebar</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
