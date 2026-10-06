"use client"; import { Sidebar } from "@/components/layout/Sidebar"; import { Header } from "@/components/layout/Header"; import { useAuth } from "@/hooks/useAuth"; import { useRouter } from "next/navigation"; import { useEffect } from "react";
export default function ProtectedLayout({ children }: { children: React.ReactNode }) {
  const { isAuthenticated, isLoading } = useAuth(); const router = useRouter();
  useEffect(() => { if (!isLoading && !isAuthenticated) { router.push("/login"); } }, [isAuthenticated, isLoading, router]);
  if (isLoading || !isAuthenticated) return null;
  return (<div className="flex h-screen overflow-hidden"><Sidebar /><div className="flex flex-1 flex-col overflow-hidden"><Header /><main className="flex-1 overflow-y-auto bg-gray-50 p-6">{children}</main></div></div>);
}