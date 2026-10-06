"use client"; import { useState, useRef, useEffect } from "react"; import { apiClient } from "@/lib/api"; import { Button } from "@/components/ui/button"; import { Input } from "@/components/ui/input"; import { Card } from "@/components/ui/card"; import { Send, ThumbsUp, ThumbsDown } from "lucide-react";
export default function Chat() {
  const [messages, setMessages] = useState<any[]>([]); const [input, setInput] = useState(""); const [isStreaming, setIsStreaming] = useState(false); const endRef = useRef<HTMLDivElement>(null);
  const scrollToBottom = () => endRef.current?.scrollIntoView({ behavior: "smooth" });
  useEffect(() => { scrollToBottom(); }, [messages]);
  const handleSend = async (e: React.FormEvent) => {
    e.preventDefault(); if (!input.trim() || isStreaming) return;
    const userMsg = { role: 'user', content: input }; setMessages(prev => [...prev, userMsg]); setInput(""); setIsStreaming(true);
    const stream = apiClient.chat.stream(userMsg.content); let assistantContent = "";
    stream.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === 'token') { assistantContent += data.content; setMessages(prev => { const newMsgs = [...prev]; if (newMsgs[newMsgs.length-1].role === 'assistant') { newMsgs[newMsgs.length-1].content = assistantContent; } else { newMsgs.push({ role: 'assistant', content: assistantContent, sources: [] }); } return newMsgs; }); }
      else if (data.type === 'sources') { setMessages(prev => { const newMsgs = [...prev]; if (newMsgs[newMsgs.length-1].role === 'assistant') { newMsgs[newMsgs.length-1].sources = data.sources; } return newMsgs; }); }
      else if (data.type === 'done') { stream.close(); setIsStreaming(false); }
    };
    stream.onerror = () => { stream.close(); setIsStreaming(false); };
  };
  return (<div className="flex flex-col h-[calc(100vh-8rem)]"><div className="flex-1 overflow-y-auto space-y-4 pb-4">{messages.map((msg, i) => (<div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}><div className={`max-w-[80%] rounded-lg p-4 ${msg.role === 'user' ? 'bg-primary text-primary-foreground' : 'bg-white border'}`}><div className="whitespace-pre-wrap text-sm">{msg.content}</div>{msg.sources?.length > 0 && (<div className="mt-3 pt-3 border-t text-xs text-gray-500"><p className="font-semibold mb-1">Sources:</p>{msg.sources.map((s:any, j:number) => (<span key={j} className="inline-block bg-gray-100 rounded px-2 py-1 mr-2 mb-2">{s.document} (p.{s.page})</span>))}</div>)}</div></div>))}<div ref={endRef} /></div><form onSubmit={handleSend} className="flex gap-2 mt-auto bg-white p-4 rounded-lg border shadow-sm"><Input value={input} onChange={(e) => setInput(e.target.value)} placeholder="Ask anything about your documents..." className="flex-1" disabled={isStreaming} /><Button type="submit" disabled={isStreaming || !input.trim()}><Send className="h-4 w-4" /></Button></form></div>);
}