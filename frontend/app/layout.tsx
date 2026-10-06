"use client"; import { QueryClient, QueryClientProvider } from "@tanstack/react-query"; import { Toaster } from "react-hot-toast"; import "./globals.css"; import { useState } from "react";
export default function RootLayout({ children }: { children: React.ReactNode }) {
  const [queryClient] = useState(() => new QueryClient());
  return (<html lang="en"><body><QueryClientProvider client={queryClient}>{children}<Toaster position="top-right" /></QueryClientProvider></body></html>);
}