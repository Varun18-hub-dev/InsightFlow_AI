import { type ClassValue, clsx } from "clsx"; import { twMerge } from "tailwind-merge";
export function cn(...inputs: ClassValue[]) { return twMerge(clsx(inputs)); }
export function formatDate(date: string) { return new Date(date).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' }); }
export function formatFileSize(bytes: number) { if(bytes===0) return '0 B'; const k=1024; const dm=2; const sizes=['B','KB','MB','GB']; const i=Math.floor(Math.log(bytes)/Math.log(k)); return parseFloat((bytes/Math.pow(k,i)).toFixed(dm))+' '+sizes[i]; }
export function getStatusColor(status: string) { switch(status){ case 'completed': return 'bg-green-100 text-green-800'; case 'processing': return 'bg-blue-100 text-blue-800'; case 'failed': return 'bg-red-100 text-red-800'; default: return 'bg-yellow-100 text-yellow-800'; } }
export function truncate(str: string, n: number) { return str.length>n ? str.substr(0,n-1)+'...' : str; }