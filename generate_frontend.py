import os

BASE_DIR = r"c:\Users\Varun S\OneDrive\Documents\new project\insightflow-ai\frontend"

def w(path, content):
    target = os.path.join(BASE_DIR, path)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    with open(target, 'w', encoding='utf-8') as f:
        f.write(content.strip())
    print(f"Created {path}")

# CONFIGS
w('package.json', '''{
  "name": "insightflow-frontend",
  "version": "1.0.0",
  "private": true,
  "scripts": { "dev": "next dev", "build": "next build", "start": "next start", "lint": "next lint", "type-check": "tsc --noEmit" },
  "dependencies": { "next": "14.2.4", "react": "^18", "react-dom": "^18", "@tanstack/react-query": "^5.45.1", "axios": "^1.7.2", "zustand": "^4.5.4", "lucide-react": "^0.394.0", "clsx": "^2.1.1", "tailwind-merge": "^2.3.0", "react-dropzone": "^14.2.3", "react-hot-toast": "^2.4.1", "recharts": "^2.12.7" },
  "devDependencies": { "typescript": "^5", "@types/node": "^20", "@types/react": "^18", "@types/react-dom": "^18", "tailwindcss": "^3.4.1", "autoprefixer": "^10.4.19", "postcss": "^8.4.38", "eslint": "^8", "eslint-config-next": "14.2.4" }
}''')
w('next.config.js', '''/** @type {import('next').NextConfig} */
const nextConfig = { output: 'standalone', env: { NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000' } }
module.exports = nextConfig''')
w('tailwind.config.js', '''/** @type {import('tailwindcss').Config} */
module.exports = { content: ['./app/**/*.{ts,tsx}', './components/**/*.{ts,tsx}'], theme: { extend: { colors: { background: '#ffffff', foreground: '#09090b', primary: { DEFAULT: '#4f46e5', foreground: '#ffffff' }, border: '#e2e8f0' } } }, plugins: [] }''')
w('postcss.config.js', '''module.exports = { plugins: { tailwindcss: {}, autoprefixer: {} } }''')
w('tsconfig.json', '''{ "compilerOptions": { "target": "es5", "lib": ["dom", "dom.iterable", "esnext"], "allowJs": true, "skipLibCheck": true, "strict": true, "noEmit": true, "esModuleInterop": true, "module": "esnext", "moduleResolution": "bundler", "resolveJsonModule": true, "isolatedModules": true, "jsx": "preserve", "incremental": true, "plugins": [{ "name": "next" }], "paths": { "@/*": ["./*"] } }, "include": ["next-env.d.ts", "**/*.ts", "**/*.tsx", ".next/types/**/*.ts"], "exclude": ["node_modules"] }''')
w('.env.local.example', '''NEXT_PUBLIC_API_URL=http://localhost:8000''')
w('.eslintrc.json', '''{ "extends": "next/core-web-vitals" }''')
w('Dockerfile', '''FROM node:20-alpine AS builder\nWORKDIR /app\nCOPY package*.json ./\nRUN npm install\nCOPY . .\nRUN npm run build\nFROM node:20-alpine AS runner\nWORKDIR /app\nCOPY --from=builder /app/public ./public\nCOPY --from=builder /app/.next/standalone ./\nCOPY --from=builder /app/.next/static ./.next/static\nEXPOSE 3000\nENV PORT 3000\nCMD ["node", "server.js"]''')

# TYPES & LIB
w('types/index.ts', '''export interface User { id: string; email: string; full_name: string; is_active: boolean; created_at: string; }
export interface Document { id: string; user_id: string; filename: string; original_filename: string; document_type: string; status: 'pending'|'processing'|'completed'|'failed'; page_count: number; chunk_count: number; created_at: string; updated_at: string; }
export interface Message { id: string; conversation_id: string; user_id: string; role: 'user'|'assistant'; content: string; sources: Source[]; created_at: string; }
export interface Source { document: string; page: number; chunk_id: string; score?: number; }
export interface ChatResponse { answer: string; sources: Source[]; confidence: number; conversation_id: string; message_id: string; }
export interface StreamEvent { type: 'token'|'sources'|'done'|'error'; content?: string; sources?: Source[]; }
export interface EvaluationRun { id: string; name: string; status: string; config: any; results: any; mlflow_run_id: string; started_at: string; completed_at: string; }
export interface HealthStatus { status: string; database: string; redis: string; pinecone: string; version: string; }
export interface Experiment { experiment_id: string; name: string; artifact_location: string; lifecycle_stage: string; }
export interface Token { access_token: string; token_type: string; }
export interface ComparisonResult { summary: string; similarities: string[]; differences: string[]; sources: Source[]; }
''')
w('lib/utils.ts', '''import { type ClassValue, clsx } from "clsx"; import { twMerge } from "tailwind-merge";
export function cn(...inputs: ClassValue[]) { return twMerge(clsx(inputs)); }
export function formatDate(date: string) { return new Date(date).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' }); }
export function formatFileSize(bytes: number) { if(bytes===0) return '0 B'; const k=1024; const dm=2; const sizes=['B','KB','MB','GB']; const i=Math.floor(Math.log(bytes)/Math.log(k)); return parseFloat((bytes/Math.pow(k,i)).toFixed(dm))+' '+sizes[i]; }
export function getStatusColor(status: string) { switch(status){ case 'completed': return 'bg-green-100 text-green-800'; case 'processing': return 'bg-blue-100 text-blue-800'; case 'failed': return 'bg-red-100 text-red-800'; default: return 'bg-yellow-100 text-yellow-800'; } }
export function truncate(str: string, n: number) { return str.length>n ? str.substr(0,n-1)+'...' : str; }
''')
w('lib/store.ts', '''import { create } from 'zustand'; import { persist } from 'zustand/middleware'; import { User } from '@/types';
interface AuthStore { token: string | null; user: User | null; isAuthenticated: boolean; setToken: (token: string) => void; setUser: (user: User) => void; logout: () => void; }
export const useAuthStore = create<AuthStore>()(persist((set) => ({ token: null, user: null, isAuthenticated: false, setToken: (token) => set({ token, isAuthenticated: true }), setUser: (user) => set({ user }), logout: () => set({ token: null, user: null, isAuthenticated: false }) }), { name: 'auth-storage' }));
''')
w('lib/api.ts', '''import axios from 'axios'; import { useAuthStore } from './store'; import { Token, User, Document, ChatResponse, EvaluationRun, Experiment, HealthStatus, ComparisonResult, Source } from '@/types';
export const api = axios.create({ baseURL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000' });
api.interceptors.request.use((config) => { const token = useAuthStore.getState().token; if (token) { config.headers.Authorization = `Bearer ${token}`; } return config; });
api.interceptors.response.use(r => r, error => { if (error.response?.status === 401) { useAuthStore.getState().logout(); if (typeof window !== 'undefined' && !window.location.pathname.includes('/login')) window.location.href = '/login'; } return Promise.reject(error); });
export const apiClient = {
  auth: { login: async (username, password) => { const data = new URLSearchParams(); data.append('username', username); data.append('password', password); return api.post<Token>('/api/auth/login', data, { headers: { 'Content-Type': 'application/x-www-form-urlencoded' } }).then(r => r.data); }, register: async (email, password, full_name) => api.post<Token>('/api/auth/register', { email, password, full_name }).then(r => r.data), me: async () => api.get<User>('/api/auth/me').then(r => r.data) },
  documents: { list: async () => api.get<Document[]>('/api/documents').then(r => r.data), get: async (id: string) => api.get<Document>(`/api/documents/${id}`).then(r => r.data), upload: async (file: File) => { const form = new FormData(); form.append('file', file); return api.post<Document>('/api/documents/upload', form).then(r => r.data); }, delete: async (id: string) => api.delete(`/api/documents/${id}`).then(r => r.data), summary: async (id: string) => api.post<{summary: string, sources: Source[]}>(`/api/documents/${id}/summary`).then(r => r.data), compare: async (ids: string[], query?: string) => api.post<ComparisonResult>('/api/documents/compare', { document_ids: ids, query }).then(r => r.data) },
  chat: { send: async (query: string, conversation_id?: string) => api.post<ChatResponse>('/api/chat', { query, conversation_id }).then(r => r.data), stream: (query: string, conversation_id?: string) => { const token = useAuthStore.getState().token; const url = new URL(`${api.defaults.baseURL}/api/chat/stream`); if (token) url.searchParams.append('token', token); url.searchParams.append('query', query); if (conversation_id) url.searchParams.append('conversation_id', conversation_id); return new EventSource(url.toString()); } },
  feedback: { submit: async (message_id: string, rating: number, text?: string) => api.post('/api/feedback', { message_id, rating, feedback_text: text }).then(r => r.data) },
  evaluations: { list: async () => api.get<EvaluationRun[]>('/api/evaluations').then(r => r.data), run: async (name?: string) => api.post<EvaluationRun>('/api/evaluations/run', { name }).then(r => r.data) },
  experiments: { list: async () => api.get<Experiment[]>('/api/experiments').then(r => r.data) },
  health: { check: async () => api.get<HealthStatus>('/api/health').then(r => r.data) }
};
''')

# HOOKS
w('hooks/useAuth.ts', '''import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'; import { apiClient } from '@/lib/api'; import { useAuthStore } from '@/lib/store'; import { useRouter } from 'next/navigation'; import toast from 'react-hot-toast';
export function useAuth() { const { token, setUser, setToken, logout, isAuthenticated } = useAuthStore(); const router = useRouter(); const queryClient = useQueryClient();
  const login = useMutation({ mutationFn: (d: any) => apiClient.auth.login(d.email, d.password), onSuccess: (data) => { setToken(data.access_token); router.push('/dashboard'); }, onError: () => toast.error("Login failed") });
  const register = useMutation({ mutationFn: (d: any) => apiClient.auth.register(d.email, d.password, d.full_name), onSuccess: (data) => { setToken(data.access_token); router.push('/dashboard'); }, onError: () => toast.error("Registration failed") });
  const user = useQuery({ queryKey: ['me'], queryFn: async () => { const u = await apiClient.auth.me(); setUser(u); return u; }, enabled: !!token });
  return { login, register, user: user.data, isLoading: user.isLoading, logout: () => { logout(); queryClient.clear(); router.push('/login'); }, isAuthenticated }; }
''')
w('hooks/useDocuments.ts', '''import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'; import { apiClient } from '@/lib/api'; import toast from 'react-hot-toast';
export function useDocuments() { return useQuery({ queryKey: ['documents'], queryFn: apiClient.documents.list }); }
export function useDocument(id: string) { return useQuery({ queryKey: ['document', id], queryFn: () => apiClient.documents.get(id), enabled: !!id }); }
export function useUploadDocument() { const queryClient = useQueryClient(); return useMutation({ mutationFn: apiClient.documents.upload, onSuccess: () => { queryClient.invalidateQueries({queryKey: ['documents']}); toast.success('Document uploaded successfully'); }, onError: () => toast.error('Upload failed') }); }
export function useDeleteDocument() { const queryClient = useQueryClient(); return useMutation({ mutationFn: apiClient.documents.delete, onSuccess: () => { queryClient.invalidateQueries({queryKey: ['documents']}); toast.success('Document deleted'); }, onError: () => toast.error('Deletion failed') }); }
''')

# UI COMPONENTS
w('components/ui/button.tsx', '''import * as React from "react"; import { cn } from "@/lib/utils";
export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> { variant?: "default" | "destructive" | "outline" | "secondary" | "ghost" | "link"; size?: "default" | "sm" | "lg" | "icon"; }
const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(({ className, variant = "default", size = "default", ...props }, ref) => {
  const variants = { default: "bg-primary text-primary-foreground hover:bg-primary/90", destructive: "bg-red-500 text-white hover:bg-red-500/90", outline: "border border-input bg-background hover:bg-accent hover:text-accent-foreground", secondary: "bg-gray-100 text-gray-900 hover:bg-gray-100/80", ghost: "hover:bg-gray-100 hover:text-gray-900", link: "text-primary underline-offset-4 hover:underline" };
  const sizes = { default: "h-10 px-4 py-2", sm: "h-9 rounded-md px-3", lg: "h-11 rounded-md px-8", icon: "h-10 w-10" };
  return <button className={cn("inline-flex items-center justify-center rounded-md text-sm font-medium ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50", variants[variant], sizes[size], className)} ref={ref} {...props} />; });
Button.displayName = "Button"; export { Button };
''')
w('components/ui/input.tsx', '''import * as React from "react"; import { cn } from "@/lib/utils";
export interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> { label?: string; error?: string; }
const Input = React.forwardRef<HTMLInputElement, InputProps>(({ className, type, label, error, ...props }, ref) => {
  return (<div className="w-full">{label && <label className="block text-sm font-medium mb-1">{label}</label>}<input type={type} className={cn("flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50", className)} ref={ref} {...props} />{error && <p className="mt-1 text-sm text-red-500">{error}</p>}</div>); });
Input.displayName = "Input"; export { Input };
''')
w('components/ui/badge.tsx', '''import * as React from "react"; import { cn } from "@/lib/utils";
export function Badge({ className, variant="default", ...props }: React.HTMLAttributes<HTMLDivElement> & { variant?: "default" | "secondary" | "destructive" | "outline" }) {
  const variants = { default: "bg-primary text-primary-foreground", secondary: "bg-gray-100 text-gray-900", destructive: "bg-red-500 text-white", outline: "text-foreground" };
  return (<div className={cn("inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-semibold transition-colors focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2", variants[variant], className)} {...props} />);
}
''')
w('components/ui/card.tsx', '''import * as React from "react"; import { cn } from "@/lib/utils";
export const Card = React.forwardRef<HTMLDivElement, React.HTMLAttributes<HTMLDivElement>>(({ className, ...props }, ref) => (<div ref={ref} className={cn("rounded-lg border bg-card text-card-foreground shadow-sm bg-white", className)} {...props} />));
export const CardHeader = React.forwardRef<HTMLDivElement, React.HTMLAttributes<HTMLDivElement>>(({ className, ...props }, ref) => (<div ref={ref} className={cn("flex flex-col space-y-1.5 p-6", className)} {...props} />));
export const CardTitle = React.forwardRef<HTMLParagraphElement, React.HTMLAttributes<HTMLHeadingElement>>(({ className, ...props }, ref) => (<h3 ref={ref} className={cn("text-lg font-semibold leading-none tracking-tight", className)} {...props} />));
export const CardContent = React.forwardRef<HTMLDivElement, React.HTMLAttributes<HTMLDivElement>>(({ className, ...props }, ref) => (<div ref={ref} className={cn("p-6 pt-0", className)} {...props} />));
''')
w('components/ui/spinner.tsx', '''import { Loader2 } from "lucide-react";
export function Spinner({ className }: { className?: string }) { return <Loader2 className={`animate-spin ${className || ''}`} />; }
''')

# LAYOUT
w('components/layout/Sidebar.tsx', '''import Link from "next/link"; import { usePathname } from "next/navigation"; import { LayoutDashboard, FileText, MessageSquare, Activity, TestTube, Settings } from "lucide-react"; import { cn } from "@/lib/utils";
export function Sidebar() {
  const pathname = usePathname();
  const links = [{ name: "Dashboard", href: "/dashboard", icon: LayoutDashboard }, { name: "Documents", href: "/documents", icon: FileText }, { name: "Chat", href: "/chat", icon: MessageSquare }, { name: "Evaluation", href: "/evaluation", icon: Activity }, { name: "Experiments", href: "/experiments", icon: TestTube }, { name: "Settings", href: "/settings", icon: Settings }];
  return (<div className="flex h-full w-64 flex-col border-r bg-gray-50"><div className="p-6"><h1 className="text-xl font-bold text-primary">InsightFlow AI</h1></div><nav className="flex-1 space-y-1 px-4">{links.map((link) => { const Icon = link.icon; return (<Link key={link.href} href={link.href} className={cn("flex items-center gap-3 rounded-md px-3 py-2 text-sm font-medium", pathname === link.href ? "bg-primary/10 text-primary" : "text-gray-600 hover:bg-gray-100")}><Icon className="h-5 w-5" />{link.name}</Link>); })}</nav></div>); }
''')
w('components/layout/Header.tsx', '''import { useAuth } from "@/hooks/useAuth"; import { Button } from "../ui/button"; import { LogOut, User } from "lucide-react";
export function Header() { const { user, logout } = useAuth(); return (<header className="flex h-16 items-center justify-end border-b px-6 bg-white"><div className="flex items-center gap-4"><div className="flex items-center gap-2 text-sm font-medium text-gray-700"><User className="h-5 w-5" />{user?.full_name || 'User'}</div><Button variant="ghost" size="sm" onClick={logout}><LogOut className="h-4 w-4 mr-2"/>Logout</Button></div></header>); }
''')

# APP
w('app/globals.css', '''@tailwind base; @tailwind components; @tailwind utilities;
@layer base { body { @apply bg-gray-50 text-gray-900; } }''')
w('app/layout.tsx', '''"use client"; import { QueryClient, QueryClientProvider } from "@tanstack/react-query"; import { Toaster } from "react-hot-toast"; import "./globals.css"; import { useState } from "react";
export default function RootLayout({ children }: { children: React.ReactNode }) {
  const [queryClient] = useState(() => new QueryClient());
  return (<html lang="en"><body><QueryClientProvider client={queryClient}>{children}<Toaster position="top-right" /></QueryClientProvider></body></html>);
}''')
w('app/(protected)/layout.tsx', '''"use client"; import { Sidebar } from "@/components/layout/Sidebar"; import { Header } from "@/components/layout/Header"; import { useAuth } from "@/hooks/useAuth"; import { useRouter } from "next/navigation"; import { useEffect } from "react";
export default function ProtectedLayout({ children }: { children: React.ReactNode }) {
  const { isAuthenticated, isLoading } = useAuth(); const router = useRouter();
  useEffect(() => { if (!isLoading && !isAuthenticated) { router.push("/login"); } }, [isAuthenticated, isLoading, router]);
  if (isLoading || !isAuthenticated) return null;
  return (<div className="flex h-screen overflow-hidden"><Sidebar /><div className="flex flex-1 flex-col overflow-hidden"><Header /><main className="flex-1 overflow-y-auto bg-gray-50 p-6">{children}</main></div></div>);
}''')
w('app/login/page.tsx', '''"use client"; import { useState } from "react"; import { useAuth } from "@/hooks/useAuth"; import { Button } from "@/components/ui/button"; import { Input } from "@/components/ui/input"; import Link from "next/link"; import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
export default function Login() { const [email, setEmail] = useState(""); const [password, setPassword] = useState(""); const { login } = useAuth();
  const handleSubmit = (e: React.FormEvent) => { e.preventDefault(); login.mutate({ email, password }); };
  return (<div className="flex min-h-screen items-center justify-center bg-gray-50 p-4"><Card className="w-full max-w-md"><CardHeader className="text-center"><CardTitle className="text-2xl text-primary">InsightFlow AI</CardTitle><p className="text-sm text-gray-500 mt-2">Sign in to your account</p></CardHeader><CardContent><form onSubmit={handleSubmit} className="space-y-4"><Input label="Email" type="email" value={email} onChange={(e)=>setEmail(e.target.value)} required /><Input label="Password" type="password" value={password} onChange={(e)=>setPassword(e.target.value)} required /><Button type="submit" className="w-full" disabled={login.isPending}>{login.isPending ? "Signing in..." : "Sign in"}</Button></form><div className="mt-4 text-center text-sm"><Link href="/register" className="text-primary hover:underline">Don't have an account? Register</Link></div></CardContent></Card></div>); }
''')
w('app/register/page.tsx', '''"use client"; import { useState } from "react"; import { useAuth } from "@/hooks/useAuth"; import { Button } from "@/components/ui/button"; import { Input } from "@/components/ui/input"; import Link from "next/link"; import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"; import toast from "react-hot-toast";
export default function Register() { const [form, setForm] = useState({full_name:"", email:"", password:"", confirm:""}); const { register } = useAuth();
  const handleSubmit = (e: React.FormEvent) => { e.preventDefault(); if(form.password !== form.confirm) { toast.error("Passwords don't match"); return; } register.mutate({ email: form.email, password: form.password, full_name: form.full_name }); };
  return (<div className="flex min-h-screen items-center justify-center bg-gray-50 p-4"><Card className="w-full max-w-md"><CardHeader className="text-center"><CardTitle className="text-2xl text-primary">InsightFlow AI</CardTitle><p className="text-sm text-gray-500 mt-2">Create an account</p></CardHeader><CardContent><form onSubmit={handleSubmit} className="space-y-4"><Input label="Full Name" value={form.full_name} onChange={(e)=>setForm({...form, full_name: e.target.value})} required /><Input label="Email" type="email" value={form.email} onChange={(e)=>setForm({...form, email: e.target.value})} required /><Input label="Password" type="password" value={form.password} onChange={(e)=>setForm({...form, password: e.target.value})} required /><Input label="Confirm Password" type="password" value={form.confirm} onChange={(e)=>setForm({...form, confirm: e.target.value})} required /><Button type="submit" className="w-full" disabled={register.isPending}>{register.isPending ? "Registering..." : "Register"}</Button></form><div className="mt-4 text-center text-sm"><Link href="/login" className="text-primary hover:underline">Already have an account? Sign in</Link></div></CardContent></Card></div>); }
''')
w('app/(protected)/dashboard/page.tsx', '''"use client"; import { useQuery } from "@tanstack/react-query"; import { apiClient } from "@/lib/api"; import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"; import { FileText, MessageSquare, Activity } from "lucide-react";
export default function Dashboard() {
  const { data: health } = useQuery({ queryKey: ['health'], queryFn: apiClient.health.check });
  const { data: docs } = useQuery({ queryKey: ['documents'], queryFn: apiClient.documents.list });
  return (<div className="space-y-6"><h1 className="text-2xl font-bold">Dashboard</h1><div className="grid gap-6 md:grid-cols-3"><Card><CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2"><CardTitle className="text-sm font-medium">Total Documents</CardTitle><FileText className="h-4 w-4 text-muted-foreground" /></CardHeader><CardContent><div className="text-2xl font-bold">{docs?.length || 0}</div></CardContent></Card><Card><CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2"><CardTitle className="text-sm font-medium">System Health</CardTitle><Activity className="h-4 w-4 text-muted-foreground" /></CardHeader><CardContent><div className="text-sm">Database: {health?.database || 'N/A'}<br/>Redis: {health?.redis || 'N/A'}<br/>Pinecone: {health?.pinecone || 'N/A'}</div></CardContent></Card></div></div>);
}''')
w('app/(protected)/documents/page.tsx', '''"use client"; import { useCallback } from "react"; import { useDropzone } from "react-dropzone"; import { useDocuments, useUploadDocument, useDeleteDocument } from "@/hooks/useDocuments"; import { Card, CardContent } from "@/components/ui/card"; import { Button } from "@/components/ui/button"; import { Badge } from "@/components/ui/badge"; import { UploadCloud, Trash2 } from "lucide-react"; import { formatDate, formatFileSize, getStatusColor } from "@/lib/utils";
export default function Documents() {
  const { data: docs, isLoading } = useDocuments(); const upload = useUploadDocument(); const del = useDeleteDocument();
  const onDrop = useCallback((files: File[]) => { if(files.length > 0) upload.mutate(files[0]); }, [upload]);
  const { getRootProps, getInputProps, isDragActive } = useDropzone({ onDrop, accept: { 'application/pdf': ['.pdf'], 'text/plain': ['.txt', '.md'], 'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'] } });
  return (<div className="space-y-6"><h1 className="text-2xl font-bold">Documents</h1><Card><CardContent className="p-6"><div {...getRootProps()} className={`border-2 border-dashed rounded-lg p-10 text-center cursor-pointer transition-colors ${isDragActive ? 'border-primary bg-primary/5' : 'border-gray-300 hover:border-primary'}`}><input {...getInputProps()} /><UploadCloud className="mx-auto h-12 w-12 text-gray-400 mb-4" /><p className="text-sm text-gray-600">Drag & drop a file here, or click to select</p><p className="text-xs text-gray-400 mt-2">Supports PDF, DOCX, TXT, MD</p></div></CardContent></Card><div className="bg-white rounded-lg border overflow-hidden"><table className="w-full text-sm text-left"><thead className="bg-gray-50 text-gray-700"><tr><th className="px-6 py-3">Name</th><th className="px-6 py-3">Status</th><th className="px-6 py-3">Pages / Chunks</th><th className="px-6 py-3">Date</th><th className="px-6 py-3">Actions</th></tr></thead><tbody>{docs?.map(doc => (<tr key={doc.id} className="border-b"><td className="px-6 py-4 font-medium"><a href={`/documents/${doc.id}`} className="text-primary hover:underline">{doc.original_filename}</a></td><td className="px-6 py-4"><span className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(doc.status)}`}>{doc.status}</span></td><td className="px-6 py-4">{doc.page_count} / {doc.chunk_count}</td><td className="px-6 py-4">{formatDate(doc.created_at)}</td><td className="px-6 py-4"><Button variant="ghost" size="sm" onClick={() => del.mutate(doc.id)} className="text-red-500 hover:text-red-700"><Trash2 className="h-4 w-4" /></Button></td></tr>))}</tbody></table>{docs?.length === 0 && <div className="p-8 text-center text-gray-500">No documents found. Upload one to get started.</div>}</div></div>);
}''')
w('app/(protected)/documents/[id]/page.tsx', '''"use client"; import { useParams } from "next/navigation"; import { useQuery } from "@tanstack/react-query"; import { apiClient } from "@/lib/api"; import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"; import { Button } from "@/components/ui/button"; import { useState } from "react";
export default function DocumentDetail() {
  const { id } = useParams(); const [summary, setSummary] = useState(""); const [loading, setLoading] = useState(false);
  const { data: doc } = useQuery({ queryKey: ['document', id], queryFn: () => apiClient.documents.get(id as string) });
  const getSummary = async () => { setLoading(true); try { const res = await apiClient.documents.summary(id as string); setSummary(res.summary); } finally { setLoading(false); } };
  if (!doc) return <div>Loading...</div>;
  return (<div className="space-y-6"><h1 className="text-2xl font-bold">{doc.original_filename}</h1><div className="grid gap-6 md:grid-cols-2"><Card><CardHeader><CardTitle>Metadata</CardTitle></CardHeader><CardContent><div className="space-y-2 text-sm"><p><strong>Status:</strong> {doc.status}</p><p><strong>Pages:</strong> {doc.page_count}</p><p><strong>Chunks:</strong> {doc.chunk_count}</p></div><div className="mt-4"><Button onClick={getSummary} disabled={loading}>{loading ? 'Summarizing...' : 'Generate Summary'}</Button></div></CardContent></Card>{summary && <Card><CardHeader><CardTitle>Summary</CardTitle></CardHeader><CardContent><p className="text-sm whitespace-pre-wrap">{summary}</p></CardContent></Card>}</div></div>);
}''')
w('app/(protected)/chat/page.tsx', '''"use client"; import { useState, useRef, useEffect } from "react"; import { apiClient } from "@/lib/api"; import { Button } from "@/components/ui/button"; import { Input } from "@/components/ui/input"; import { Card } from "@/components/ui/card"; import { Send, ThumbsUp, ThumbsDown } from "lucide-react";
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
}''')
w('app/(protected)/evaluation/page.tsx', '''"use client"; import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query"; import { apiClient } from "@/lib/api"; import { Button } from "@/components/ui/button"; import { Card, CardContent } from "@/components/ui/card";
export default function Evaluation() {
  const qc = useQueryClient(); const { data: evals } = useQuery({ queryKey: ['evals'], queryFn: apiClient.evaluations.list });
  const run = useMutation({ mutationFn: () => apiClient.evaluations.run(), onSuccess: () => qc.invalidateQueries({queryKey:['evals']}) });
  return (<div className="space-y-6"><div className="flex justify-between items-center"><h1 className="text-2xl font-bold">Evaluations</h1><Button onClick={() => run.mutate()} disabled={run.isPending}>{run.isPending ? 'Running...' : 'Run Evaluation'}</Button></div><Card><CardContent className="p-0"><table className="w-full text-sm text-left"><thead className="bg-gray-50"><tr><th className="px-6 py-3">Name</th><th className="px-6 py-3">Status</th><th className="px-6 py-3">Results</th></tr></thead><tbody>{evals?.map(e => (<tr key={e.id} className="border-b"><td className="px-6 py-4">{e.name}</td><td className="px-6 py-4">{e.status}</td><td className="px-6 py-4"><pre className="text-xs">{JSON.stringify(e.results, null, 2)}</pre></td></tr>))}</tbody></table></CardContent></Card></div>);
}''')
w('app/(protected)/experiments/page.tsx', '''"use client"; import { useQuery } from "@tanstack/react-query"; import { apiClient } from "@/lib/api"; import { Card, CardContent } from "@/components/ui/card";
export default function Experiments() {
  const { data: exp } = useQuery({ queryKey: ['experiments'], queryFn: apiClient.experiments.list });
  return (<div className="space-y-6"><h1 className="text-2xl font-bold">Experiments</h1><Card><CardContent className="p-0"><table className="w-full text-sm text-left"><thead className="bg-gray-50"><tr><th className="px-6 py-3">Name</th><th className="px-6 py-3">Stage</th><th className="px-6 py-3">Artifacts</th></tr></thead><tbody>{exp?.map(e => (<tr key={e.experiment_id} className="border-b"><td className="px-6 py-4">{e.name}</td><td className="px-6 py-4">{e.lifecycle_stage}</td><td className="px-6 py-4 truncate max-w-xs" title={e.artifact_location}>{e.artifact_location}</td></tr>))}</tbody></table></CardContent></Card></div>);
}''')
w('app/(protected)/settings/page.tsx', '''"use client"; import { useQuery } from "@tanstack/react-query"; import { apiClient } from "@/lib/api"; import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
export default function Settings() {
  const { data: health } = useQuery({ queryKey: ['health'], queryFn: apiClient.health.check });
  return (<div className="space-y-6"><h1 className="text-2xl font-bold">Settings</h1><Card><CardHeader><CardTitle>System Information</CardTitle></CardHeader><CardContent><div className="space-y-4"><div className="grid grid-cols-2 gap-4 border-b pb-4"><div><p className="text-sm text-gray-500">API Version</p><p className="font-medium">{health?.version || 'Unknown'}</p></div><div><p className="text-sm text-gray-500">Status</p><p className="font-medium text-green-600">{health?.status || 'Unknown'}</p></div></div><div className="text-sm text-gray-600"><p>Configure environment variables on the server to change LLM providers, database connections, and API keys.</p></div></div></CardContent></Card></div>);
}''')
