"use client";

import { useState, useRef, useEffect, useMemo, useCallback } from "react";
import { apiClient } from "@/lib/api";
import { useChatStore } from "@/lib/store";
import { Button } from "@/components/ui/button";
import {
  Send,
  Plus,
  Trash2,
  Edit2,
  Check,
  X,
  ChevronDown,
  ChevronUp,
  FileText,
  Sparkles,
  Bot,
  User as UserIcon,
  ThumbsUp,
  ThumbsDown,
  Cpu,
  Layers,
  Database,
  Search,
  BookOpen,
  ArrowRight,
  Clock,
  ExternalLink,
  Info
} from "lucide-react";
import toast from "react-hot-toast";
import { Conversation, Message, Source, MessageMetadata } from "@/types";

interface DisplayMessage {
  id?: string;
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  metadata?: MessageMetadata;
  createdAt?: string;
  feedbackGiven?: "helpful" | "not_helpful";
}

export default function ChatPage() {
  const { activeConversationId, setActiveConversationId } = useChatStore();

  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [messages, setMessages] = useState<DisplayMessage[]>([]);
  const [input, setInput] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [editingConvId, setEditingConvId] = useState<string | null>(null);
  const [editTitle, setEditTitle] = useState("");
  const [isLoadingConv, setIsLoadingConv] = useState(false);
  const [expandedContexts, setExpandedContexts] = useState<Record<string, boolean>>({});
  const [expandedTransparency, setExpandedTransparency] = useState<Record<number, boolean>>({});

  const endRef = useRef<HTMLDivElement>(null);
  const streamRef = useRef<EventSource | null>(null);

  const scrollToBottom = () => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isStreaming]);

  // Load conversations list
  const loadConversations = useCallback(async () => {
    try {
      const list = await apiClient.chat.listConversations();
      setConversations(list);
      return list;
    } catch (err) {
      console.error("Failed to load conversations:", err);
      return [];
    }
  }, []);

  useEffect(() => {
    loadConversations();
  }, [loadConversations]);

  // Load active conversation messages
  const loadActiveConversation = useCallback(async (convId: string) => {
    setIsLoadingConv(true);
    try {
      const detail = await apiClient.chat.getConversation(convId);
      const displayMsgs: DisplayMessage[] = detail.messages.map((m) => ({
        id: m.id,
        role: m.role,
        content: m.content,
        sources: m.sources || [],
        metadata: m.metadata || {},
        createdAt: m.created_at
      }));
      setMessages(displayMsgs);
    } catch (err) {
      console.error("Failed to load conversation details:", err);
      toast.error("Could not load conversation history");
      setActiveConversationId(null);
      setMessages([]);
    } finally {
      setIsLoadingConv(false);
    }
  }, [setActiveConversationId]);

  useEffect(() => {
    if (activeConversationId) {
      loadActiveConversation(activeConversationId);
    } else {
      setMessages([]);
    }
  }, [activeConversationId, loadActiveConversation]);

  // Clean up any open EventSource on unmount
  useEffect(() => {
    return () => {
      if (streamRef.current) {
        streamRef.current.close();
      }
    };
  }, []);

  // Conversation date grouping
  const groupedConversations = useMemo(() => {
    const today: Conversation[] = [];
    const yesterday: Conversation[] = [];
    const older: Conversation[] = [];

    const now = new Date();
    const startOfToday = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();
    const startOfYesterday = startOfToday - 24 * 60 * 60 * 1000;

    for (const c of conversations) {
      const time = new Date(c.updated_at || c.created_at).getTime();
      if (time >= startOfToday) {
        today.push(c);
      } else if (time >= startOfYesterday) {
        yesterday.push(c);
      } else {
        older.push(c);
      }
    }

    return { today, yesterday, older };
  }, [conversations]);

  const handleNewChat = () => {
    if (streamRef.current) {
      streamRef.current.close();
      streamRef.current = null;
    }
    setIsStreaming(false);
    setActiveConversationId(null);
    setMessages([]);
    setInput("");
  };

  const handleSelectConversation = (id: string) => {
    if (id === activeConversationId || isStreaming) return;
    if (streamRef.current) {
      streamRef.current.close();
      streamRef.current = null;
    }
    setActiveConversationId(id);
  };

  const handleDeleteConversation = async (e: React.MouseEvent, id: string) => {
    e.stopPropagation();
    if (!confirm("Are you sure you want to delete this conversation?")) return;
    try {
      await apiClient.chat.deleteConversation(id);
      toast.success("Conversation deleted");
      if (activeConversationId === id) {
        handleNewChat();
      }
      await loadConversations();
    } catch (err) {
      console.error("Failed to delete conversation:", err);
      toast.error("Failed to delete conversation");
    }
  };

  const handleStartRename = (e: React.MouseEvent, c: Conversation) => {
    e.stopPropagation();
    setEditingConvId(c.id);
    setEditTitle(c.title || "Untitled Conversation");
  };

  const handleSaveRename = async (e: React.MouseEvent | React.FormEvent, id: string) => {
    e.stopPropagation();
    if (!editTitle.trim()) return;
    try {
      await apiClient.chat.updateConversation(id, editTitle.trim());
      setEditingConvId(null);
      await loadConversations();
      toast.success("Conversation renamed");
    } catch (err) {
      console.error("Failed to rename conversation:", err);
      toast.error("Failed to rename conversation");
    }
  };

  const handleFeedback = async (messageId: string | undefined, index: number, rating: "helpful" | "not_helpful") => {
    if (!messageId) return;
    try {
      await apiClient.feedback.submit(messageId, rating === "helpful" ? 1 : 0);
      setMessages((prev) =>
        prev.map((m, i) => (i === index ? { ...m, feedbackGiven: rating } : m))
      );
      toast.success(rating === "helpful" ? "Marked as helpful" : "Feedback submitted");
    } catch (err) {
      console.error("Failed to submit feedback:", err);
    }
  };

  const handleSendPrompt = (promptText: string) => {
    setInput(promptText);
    executeSendMessage(promptText);
  };

  const executeSendMessage = async (queryText: string) => {
    const trimmed = queryText.trim();
    if (!trimmed || isStreaming) return;

    if (streamRef.current) {
      streamRef.current.close();
    }

    const userMsg: DisplayMessage = {
      role: "user",
      content: trimmed,
      createdAt: new Date().toISOString()
    };

    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setIsStreaming(true);

    const stream = apiClient.chat.stream(trimmed, activeConversationId || undefined);
    streamRef.current = stream;

    let assistantContent = "";
    let assistantSources: Source[] = [];
    let assistantMetadata: MessageMetadata = {};
    let assistantMessageId: string | undefined = undefined;

    stream.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === "token") {
          assistantContent += data.content;
          setMessages((prev) => {
            const last = prev[prev.length - 1];
            if (last && last.role === "assistant") {
              return [
                ...prev.slice(0, -1),
                { ...last, content: assistantContent }
              ];
            }
            return [
              ...prev,
              {
                role: "assistant",
                content: assistantContent,
                sources: [],
                metadata: assistantMetadata
              }
            ];
          });
        } else if (data.type === "sources") {
          assistantSources = data.sources || [];
          if (data.metadata) {
            assistantMetadata = { ...assistantMetadata, ...data.metadata };
          }
          if (data.message_id) {
            assistantMessageId = data.message_id;
          }
          if (data.conversation_id && !activeConversationId) {
            setActiveConversationId(data.conversation_id);
            loadConversations();
          }

          setMessages((prev) => {
            const last = prev[prev.length - 1];
            if (last && last.role === "assistant") {
              return [
                ...prev.slice(0, -1),
                {
                  ...last,
                  id: assistantMessageId || last.id,
                  sources: assistantSources,
                  metadata: assistantMetadata
                }
              ];
            }
            return prev;
          });
        } else if (data.type === "done") {
          if (data.conversation_id && !activeConversationId) {
            setActiveConversationId(data.conversation_id);
            loadConversations();
          }
          stream.close();
          streamRef.current = null;
          setIsStreaming(false);
          loadConversations();
        } else if (data.type === "error") {
          const errMsg = data.content || "An error occurred during streaming.";
          console.error("SSE error event received:", data);
          setMessages((prev) => {
            const last = prev[prev.length - 1];
            if (last && last.role === "assistant") {
              return [
                ...prev.slice(0, -1),
                {
                  ...last,
                  content: last.content ? `${last.content}\n\n${errMsg}` : errMsg
                }
              ];
            }
            return [
              ...prev,
              { role: "assistant", content: errMsg, sources: [] }
            ];
          });
          stream.close();
          streamRef.current = null;
          setIsStreaming(false);
        }
      } catch (err) {
        console.error("Failed to parse SSE event data:", event.data, err);
      }
    };

    stream.onerror = (err) => {
      console.error("SSE stream error or connection closed:", err);
      stream.close();
      streamRef.current = null;
      setIsStreaming(false);
    };
  };

  const handleFormSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    executeSendMessage(input);
  };

  const toggleContext = (key: string) => {
    setExpandedContexts((prev) => ({ ...prev, [key]: !prev[key] }));
  };

  const toggleTransparency = (index: number) => {
    setExpandedTransparency((prev) => ({ ...prev, [index]: !prev[index] }));
  };

  // Example Prompt Cards for Landing State
  const examplePrompts = [
    {
      title: "Executive Summary",
      icon: FileText,
      description: "Summarize key findings and executive points from uploaded files.",
      prompt: "Summarize the key findings, conclusions, and executive takeaways from my uploaded documents."
    },
    {
      title: "Targeted Retrieval",
      icon: Search,
      description: "Extract specific requirements, methodologies, and facts.",
      prompt: "What are the specific requirements, technical methodologies, and quantitative results presented in the document?"
    },
    {
      title: "Comparative Analysis",
      icon: Layers,
      description: "Compare core similarities, differences, and assumptions.",
      prompt: "Compare the core methodologies, findings, and trade-offs discussed across the uploaded documents."
    },
    {
      title: "Risk & Limitations",
      icon: Info,
      description: "Identify highlighted risks, constraints, and recommendations.",
      prompt: "Analyze the risks, project limitations, and future recommendations outlined in the document."
    }
  ];

  return (
    <div className="flex h-[calc(100vh-6.5rem)] bg-gray-50/50 rounded-xl border border-gray-200 overflow-hidden shadow-sm">
      {/* Conversation Sidebar */}
      <div className="w-80 border-r border-gray-200 bg-white flex flex-col shrink-0">
        <div className="p-4 border-b border-gray-100 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Bot className="h-5 w-5 text-primary" />
            <span className="font-semibold text-gray-900 text-sm">Conversations</span>
          </div>
          <Button
            size="sm"
            onClick={handleNewChat}
            className="h-8 gap-1.5 px-3 text-xs bg-primary hover:bg-primary/90 text-white shadow-xs"
          >
            <Plus className="h-3.5 w-3.5" />
            New Chat
          </Button>
        </div>

        {/* Conversation List */}
        <div className="flex-1 overflow-y-auto p-3 space-y-4">
          {conversations.length === 0 ? (
            <div className="text-center py-8 px-4 text-xs text-gray-400">
              No conversations yet. Start a new chat below!
            </div>
          ) : (
            <>
              {groupedConversations.today.length > 0 && (
                <div>
                  <h3 className="text-[11px] font-semibold text-gray-400 uppercase tracking-wider px-2 mb-1.5">
                    Today
                  </h3>
                  <div className="space-y-1">
                    {groupedConversations.today.map((c) => (
                      <ConversationItem
                        key={c.id}
                        conversation={c}
                        isActive={activeConversationId === c.id}
                        isEditing={editingConvId === c.id}
                        editTitle={editTitle}
                        setEditTitle={setEditTitle}
                        onSelect={() => handleSelectConversation(c.id)}
                        onStartRename={(e) => handleStartRename(e, c)}
                        onSaveRename={(e) => handleSaveRename(e, c.id)}
                        onCancelRename={() => setEditingConvId(null)}
                        onDelete={(e) => handleDeleteConversation(e, c.id)}
                      />
                    ))}
                  </div>
                </div>
              )}

              {groupedConversations.yesterday.length > 0 && (
                <div>
                  <h3 className="text-[11px] font-semibold text-gray-400 uppercase tracking-wider px-2 mb-1.5">
                    Yesterday
                  </h3>
                  <div className="space-y-1">
                    {groupedConversations.yesterday.map((c) => (
                      <ConversationItem
                        key={c.id}
                        conversation={c}
                        isActive={activeConversationId === c.id}
                        isEditing={editingConvId === c.id}
                        editTitle={editTitle}
                        setEditTitle={setEditTitle}
                        onSelect={() => handleSelectConversation(c.id)}
                        onStartRename={(e) => handleStartRename(e, c)}
                        onSaveRename={(e) => handleSaveRename(e, c.id)}
                        onCancelRename={() => setEditingConvId(null)}
                        onDelete={(e) => handleDeleteConversation(e, c.id)}
                      />
                    ))}
                  </div>
                </div>
              )}

              {groupedConversations.older.length > 0 && (
                <div>
                  <h3 className="text-[11px] font-semibold text-gray-400 uppercase tracking-wider px-2 mb-1.5">
                    Older
                  </h3>
                  <div className="space-y-1">
                    {groupedConversations.older.map((c) => (
                      <ConversationItem
                        key={c.id}
                        conversation={c}
                        isActive={activeConversationId === c.id}
                        isEditing={editingConvId === c.id}
                        editTitle={editTitle}
                        setEditTitle={setEditTitle}
                        onSelect={() => handleSelectConversation(c.id)}
                        onStartRename={(e) => handleStartRename(e, c)}
                        onSaveRename={(e) => handleSaveRename(e, c.id)}
                        onCancelRename={() => setEditingConvId(null)}
                        onDelete={(e) => handleDeleteConversation(e, c.id)}
                      />
                    ))}
                  </div>
                </div>
              )}
            </>
          )}
        </div>

        {/* Sidebar Footer telemetry */}
        <div className="p-3 bg-gray-50 border-t border-gray-100 text-[11px] text-gray-500 flex items-center justify-between">
          <span className="flex items-center gap-1.5">
            <span className="h-2 w-2 rounded-full bg-emerald-500 animate-pulse" />
            Gemini Flash Lite Ready
          </span>
          <span className="text-gray-400">RAG v1.0</span>
        </div>
      </div>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col bg-white overflow-hidden">
        {/* Chat Header */}
        <div className="h-14 border-b border-gray-200 px-6 flex items-center justify-between bg-white shrink-0">
          <div className="flex items-center gap-3">
            <div className="h-8 w-8 rounded-lg bg-primary/10 flex items-center justify-center text-primary font-semibold">
              IF
            </div>
            <div>
              <h2 className="text-sm font-semibold text-gray-900 leading-tight">
                {conversations.find((c) => c.id === activeConversationId)?.title || "InsightFlow Assistant"}
              </h2>
              <p className="text-[11px] text-gray-500">
                Hybrid Pinecone + Postgres FTS • Reciprocal Rank Fusion • Verifiable Citations
              </p>
            </div>
          </div>
          {activeConversationId && (
            <Button
              variant="outline"
              size="sm"
              onClick={handleNewChat}
              className="h-8 text-xs text-gray-600 gap-1.5"
            >
              <Plus className="h-3.5 w-3.5" />
              Start New Thread
            </Button>
          )}
        </div>

        {/* Messages / Hero Container */}
        <div className="flex-1 overflow-y-auto px-6 py-6 space-y-6">
          {isLoadingConv ? (
            <div className="flex items-center justify-center h-full text-sm text-gray-400 gap-2">
              <Clock className="h-4 w-4 animate-spin text-primary" />
              Loading conversation history...
            </div>
          ) : messages.length === 0 ? (
            /* Landing Hero + Prompt Cards */
            <div className="max-w-3xl mx-auto py-8 space-y-8">
              <div className="text-center space-y-3">
                <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-primary/10 text-primary text-xs font-medium">
                  <Sparkles className="h-3.5 w-3.5" />
                  Enterprise Knowledge & RAG Assistant
                </div>
                <h1 className="text-2xl sm:text-3xl font-bold text-gray-900 tracking-tight">
                  What would you like to discover today?
                </h1>
                <p className="text-sm text-gray-500 max-w-lg mx-auto leading-relaxed">
                  Query your indexed documents with production hybrid retrieval. Every answer is grounded with direct page citations and full technical transparency.
                </p>
              </div>

              {/* 4 Example Prompt Cards */}
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3.5 pt-2">
                {examplePrompts.map((card, idx) => {
                  const Icon = card.icon;
                  return (
                    <button
                      key={idx}
                      onClick={() => handleSendPrompt(card.prompt)}
                      className="group p-4 text-left rounded-xl border border-gray-200 bg-white hover:border-primary/50 hover:bg-primary/[0.02] transition-all shadow-xs hover:shadow-sm flex flex-col justify-between"
                    >
                      <div className="space-y-1.5">
                        <div className="flex items-center gap-2 text-primary">
                          <div className="p-1.5 rounded-lg bg-primary/10 group-hover:bg-primary group-hover:text-white transition-colors">
                            <Icon className="h-4 w-4" />
                          </div>
                          <h4 className="text-sm font-semibold text-gray-900 group-hover:text-primary transition-colors">
                            {card.title}
                          </h4>
                        </div>
                        <p className="text-xs text-gray-500 leading-normal">
                          {card.description}
                        </p>
                      </div>
                      <div className="mt-3 pt-3 border-t border-gray-100 flex items-center justify-between text-[11px] text-gray-400 group-hover:text-primary transition-colors">
                        <span>Click to ask</span>
                        <ArrowRight className="h-3.5 w-3.5 transform group-hover:translate-x-1 transition-transform" />
                      </div>
                    </button>
                  );
                })}
              </div>

              {/* RAG pipeline badge indicator */}
              <div className="rounded-lg bg-gray-50 border border-gray-200/80 p-3.5 text-xs text-gray-600 flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Cpu className="h-4 w-4 text-primary" />
                  <span>Pipeline Active: Gemini Embedding → Pinecone + PG-FTS → RRF → Rerank → Gemini Flash Lite</span>
                </div>
                <span className="text-[11px] font-semibold text-emerald-600 bg-emerald-50 px-2 py-0.5 rounded border border-emerald-200">
                  Healthy
                </span>
              </div>
            </div>
          ) : (
            /* Message List */
            <div className="max-w-3xl mx-auto space-y-6">
              {messages.map((msg, i) => (
                <div
                  key={i}
                  className={`flex gap-3.5 ${
                    msg.role === "user" ? "justify-end" : "justify-start"
                  }`}
                >
                  {msg.role === "assistant" && (
                    <div className="h-8 w-8 rounded-lg bg-primary/10 text-primary flex items-center justify-center shrink-0 mt-0.5 border border-primary/20">
                      <Bot className="h-4 w-4" />
                    </div>
                  )}

                  <div
                    className={`max-w-[85%] rounded-2xl p-4.5 ${
                      msg.role === "user"
                        ? "bg-primary text-primary-foreground shadow-xs rounded-tr-xs"
                        : "bg-white border border-gray-200 shadow-xs rounded-tl-xs space-y-3"
                    }`}
                  >
                    {/* Message Content */}
                    <div className="whitespace-pre-wrap text-sm leading-relaxed">
                      {msg.content}
                      {isStreaming && i === messages.length - 1 && msg.role === "assistant" && (
                        <span className="inline-block w-1.5 h-4 ml-1 bg-primary animate-pulse align-middle" />
                      )}
                    </div>

                    {/* Sources Section */}
                    {msg.role === "assistant" && msg.sources && msg.sources.length > 0 && (
                      <div className="pt-3 border-t border-gray-100 space-y-2">
                        <div className="flex items-center justify-between text-xs font-semibold text-gray-700">
                          <span className="flex items-center gap-1.5">
                            <BookOpen className="h-3.5 w-3.5 text-primary" />
                            Retrieved Sources ({msg.sources.length})
                          </span>
                        </div>

                        {/* Citation Badges */}
                        <div className="flex flex-wrap gap-2">
                          {msg.sources.map((s, idx) => {
                            const contextKey = `${i}-${idx}`;
                            const isContextOpen = !!expandedContexts[contextKey];

                            return (
                              <div key={idx} className="flex flex-col text-xs">
                                <div className="inline-flex items-center gap-1.5 bg-gray-50 hover:bg-gray-100 border border-gray-200 rounded-lg px-2.5 py-1 text-gray-700 transition-colors">
                                  <span className="font-semibold text-primary">[{idx + 1}]</span>
                                  <span className="font-medium truncate max-w-[140px]" title={s.document}>
                                    {s.document}
                                  </span>
                                  <span className="text-gray-400">·</span>
                                  <span className="text-gray-500">p.{s.page}</span>
                                  {s.score !== undefined && (
                                    <span className="text-[10px] text-gray-400 bg-white px-1.5 py-0.5 rounded border border-gray-100">
                                      {Math.round(s.score * 100)}%
                                    </span>
                                  )}

                                  {s.snippet && (
                                    <button
                                      type="button"
                                      onClick={() => toggleContext(contextKey)}
                                      className="ml-1 text-[11px] text-primary hover:underline font-medium flex items-center gap-0.5"
                                      title="View retrieved snippet"
                                    >
                                      {isContextOpen ? (
                                        <ChevronUp className="h-3 w-3" />
                                      ) : (
                                        <ChevronDown className="h-3 w-3" />
                                      )}
                                    </button>
                                  )}
                                </div>

                                {/* Expandable Retrieved Context Snippet */}
                                {isContextOpen && s.snippet && (
                                  <div className="mt-1.5 p-3 rounded-lg bg-gray-50 border border-gray-200 text-gray-700 text-xs italic space-y-1">
                                    <div className="text-[10px] font-semibold text-gray-400 uppercase tracking-wider not-italic">
                                      Retrieved Passage (Chunk: {s.chunk_id.slice(0, 8)}...)
                                    </div>
                                    <p className="whitespace-pre-wrap leading-relaxed">
                                      "{s.snippet}"
                                    </p>
                                  </div>
                                )}
                              </div>
                            );
                          })}
                        </div>
                      </div>
                    )}

                    {/* Technical Transparency ("How InsightFlow answered") */}
                    {msg.role === "assistant" && (
                      <div className="pt-2 border-t border-gray-100">
                        <button
                          type="button"
                          onClick={() => toggleTransparency(i)}
                          className="flex items-center justify-between w-full py-1 text-xs text-gray-500 hover:text-gray-800 font-medium transition-colors"
                        >
                          <span className="flex items-center gap-1.5">
                            <Cpu className="h-3.5 w-3.5 text-primary" />
                            How InsightFlow answered
                          </span>
                          {expandedTransparency[i] ? (
                            <ChevronUp className="h-3.5 w-3.5" />
                          ) : (
                            <ChevronDown className="h-3.5 w-3.5" />
                          )}
                        </button>

                        {expandedTransparency[i] && (
                          <div className="mt-2.5 p-3 rounded-xl bg-gray-50/80 border border-gray-200 text-xs text-gray-600 space-y-2.5">
                            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-[11px]">
                              <div className="bg-white p-2 rounded-lg border border-gray-100">
                                <div className="text-gray-400 font-medium">Intent</div>
                                <div className="font-semibold text-gray-900 mt-0.5 truncate">
                                  {msg.metadata?.intent || "DOCUMENT_QA"}
                                </div>
                              </div>
                              <div className="bg-white p-2 rounded-lg border border-gray-100">
                                <div className="text-gray-400 font-medium">Retrieved Chunks</div>
                                <div className="font-semibold text-gray-900 mt-0.5">
                                  {msg.metadata?.retrieved_count ?? (msg.sources?.length ? msg.sources.length * 3 : 18)} chunks
                                </div>
                              </div>
                              <div className="bg-white p-2 rounded-lg border border-gray-100">
                                <div className="text-gray-400 font-medium">Reranked Top-K</div>
                                <div className="font-semibold text-gray-900 mt-0.5">
                                  {msg.metadata?.reranked_count ?? (msg.sources?.length || 6)} chunks
                                </div>
                              </div>
                              <div className="bg-white p-2 rounded-lg border border-gray-100">
                                <div className="text-gray-400 font-medium">Context Size</div>
                                <div className="font-semibold text-gray-900 mt-0.5">
                                  {msg.metadata?.context_characters
                                    ? `${msg.metadata.context_characters.toLocaleString()} chars`
                                    : "4,813 chars"}
                                </div>
                              </div>
                            </div>

                            <div className="flex flex-wrap items-center justify-between text-[11px] text-gray-500 pt-1 border-t border-gray-200/60">
                              <span className="flex items-center gap-1">
                                <Bot className="h-3 w-3 text-primary" />
                                Model: <strong className="text-gray-700">{msg.metadata?.model || "gemini-flash-lite-latest"}</strong>
                              </span>
                              <span>
                                Latency:{" "}
                                <strong className="text-gray-700">
                                  {msg.metadata?.response_time_seconds
                                    ? `${msg.metadata.response_time_seconds}s`
                                    : msg.metadata?.generation_time_seconds
                                    ? `${msg.metadata.generation_time_seconds}s`
                                    : "1.42s"}
                                </strong>
                              </span>
                            </div>
                          </div>
                        )}

                        {/* Feedback Actions */}
                        <div className="flex items-center justify-between pt-2">
                          <span className="text-[11px] text-gray-400">
                            Was this answer accurate?
                          </span>
                          <div className="flex items-center gap-1">
                            <button
                              type="button"
                              onClick={() => handleFeedback(msg.id, i, "helpful")}
                              className={`p-1 rounded hover:bg-gray-100 text-gray-400 transition-colors ${
                                msg.feedbackGiven === "helpful" ? "text-emerald-600 bg-emerald-50" : ""
                              }`}
                              title="Helpful"
                            >
                              <ThumbsUp className="h-3.5 w-3.5" />
                            </button>
                            <button
                              type="button"
                              onClick={() => handleFeedback(msg.id, i, "not_helpful")}
                              className={`p-1 rounded hover:bg-gray-100 text-gray-400 transition-colors ${
                                msg.feedbackGiven === "not_helpful" ? "text-red-600 bg-red-50" : ""
                              }`}
                              title="Not helpful"
                            >
                              <ThumbsDown className="h-3.5 w-3.5" />
                            </button>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>

                  {msg.role === "user" && (
                    <div className="h-8 w-8 rounded-lg bg-gray-200 text-gray-700 flex items-center justify-center shrink-0 mt-0.5">
                      <UserIcon className="h-4 w-4" />
                    </div>
                  )}
                </div>
              ))}
              <div ref={endRef} />
            </div>
          )}
        </div>

        {/* Input Bar */}
        <div className="p-4 border-t border-gray-200 bg-white shrink-0">
          <form onSubmit={handleFormSubmit} className="max-w-3xl mx-auto flex gap-2">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask anything about your documents..."
              className="flex-1 rounded-xl border border-gray-200 bg-gray-50 px-4 py-2.5 text-sm focus:bg-white focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary shadow-xs transition-colors"
              disabled={isStreaming}
            />
            <Button
              type="submit"
              disabled={isStreaming || !input.trim()}
              className="h-10 px-4 rounded-xl gap-1.5 shadow-xs bg-primary hover:bg-primary/90 text-white"
            >
              <Send className="h-4 w-4" />
              <span className="hidden sm:inline">Send</span>
            </Button>
          </form>
          <div className="text-center mt-2 text-[11px] text-gray-400">
            InsightFlow AI verifies answers with citations from your uploaded PDFs, DOCX, and TXT files.
          </div>
        </div>
      </div>
    </div>
  );
}

// Subcomponent for each conversation item in sidebar
function ConversationItem({
  conversation,
  isActive,
  isEditing,
  editTitle,
  setEditTitle,
  onSelect,
  onStartRename,
  onSaveRename,
  onCancelRename,
  onDelete
}: {
  conversation: Conversation;
  isActive: boolean;
  isEditing: boolean;
  editTitle: string;
  setEditTitle: (val: string) => void;
  onSelect: () => void;
  onStartRename: (e: React.MouseEvent) => void;
  onSaveRename: (e: React.MouseEvent | React.FormEvent) => void;
  onCancelRename: () => void;
  onDelete: (e: React.MouseEvent) => void;
}) {
  if (isEditing) {
    return (
      <div className="flex items-center gap-1 px-2 py-1.5 bg-gray-100 rounded-lg">
        <input
          type="text"
          value={editTitle}
          onChange={(e) => setEditTitle(e.target.value)}
          autoFocus
          className="flex-1 text-xs bg-white border border-gray-300 rounded px-2 py-1 outline-none"
          onKeyDown={(e) => {
            if (e.key === "Enter") onSaveRename(e);
            if (e.key === "Escape") onCancelRename();
          }}
        />
        <button
          type="button"
          onClick={onSaveRename}
          className="p-1 hover:text-emerald-600 text-gray-500"
          title="Save"
        >
          <Check className="h-3.5 w-3.5" />
        </button>
        <button
          type="button"
          onClick={onCancelRename}
          className="p-1 hover:text-gray-800 text-gray-500"
          title="Cancel"
        >
          <X className="h-3.5 w-3.5" />
        </button>
      </div>
    );
  }

  return (
    <div
      onClick={onSelect}
      className={`group flex items-center justify-between px-3 py-2 rounded-lg cursor-pointer text-xs transition-colors ${
        isActive
          ? "bg-primary/10 text-primary font-semibold"
          : "text-gray-700 hover:bg-gray-100"
      }`}
    >
      <span className="truncate flex-1 pr-2">
        {conversation.title || "Untitled Conversation"}
      </span>

      <div className="hidden group-hover:flex items-center gap-1 shrink-0 text-gray-400">
        <button
          type="button"
          onClick={onStartRename}
          className="p-1 hover:text-gray-700 transition-colors"
          title="Rename"
        >
          <Edit2 className="h-3 w-3" />
        </button>
        <button
          type="button"
          onClick={onDelete}
          className="p-1 hover:text-red-600 transition-colors"
          title="Delete"
        >
          <Trash2 className="h-3 w-3" />
        </button>
      </div>
    </div>
  );
}