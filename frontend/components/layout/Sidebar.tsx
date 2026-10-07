import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  LayoutDashboard,
  FileText,
  MessageSquare,
  Activity,
  TestTube,
  Settings,
  Sparkles
} from "lucide-react";
import { cn } from "@/lib/utils";

export function Sidebar() {
  const pathname = usePathname();

  const links = [
    { name: "Dashboard", href: "/dashboard", icon: LayoutDashboard },
    { name: "Documents", href: "/documents", icon: FileText },
    { name: "Chat", href: "/chat", icon: MessageSquare },
    { name: "Evaluation", href: "/evaluation", icon: Activity },
    { name: "Experiments", href: "/experiments", icon: TestTube },
    { name: "Settings", href: "/settings", icon: Settings },
  ];

  return (
    <div className="flex h-full w-64 flex-col border-r border-gray-200 bg-white">
      {/* Brand Header */}
      <div className="p-6 border-b border-gray-100 flex items-center justify-between">
        <Link href="/dashboard" className="flex items-center gap-2.5">
          <div className="h-8 w-8 rounded-lg bg-primary text-white flex items-center justify-center font-bold text-sm shadow-xs">
            IF
          </div>
          <div>
            <h1 className="text-base font-bold text-gray-900 leading-tight">InsightFlow AI</h1>
            <span className="text-[10px] text-gray-400 font-medium tracking-wide uppercase">Enterprise RAG</span>
          </div>
        </Link>
      </div>

      {/* Navigation */}
      <nav className="flex-1 space-y-1 p-4">
        {links.map((link) => {
          const Icon = link.icon;
          const isActive = pathname === link.href || (link.href !== "/dashboard" && pathname.startsWith(link.href));

          return (
            <Link
              key={link.href}
              href={link.href}
              className={cn(
                "flex items-center gap-3 rounded-xl px-3.5 py-2.5 text-sm font-medium transition-colors",
                isActive
                  ? "bg-primary/10 text-primary font-semibold"
                  : "text-gray-600 hover:bg-gray-50 hover:text-gray-900"
              )}
            >
              <Icon className={cn("h-4 w-4", isActive ? "text-primary" : "text-gray-400")} />
              {link.name}
            </Link>
          );
        })}
      </nav>

      {/* Footer Info */}
      <div className="p-4 border-t border-gray-100 bg-gray-50/50 m-3 rounded-xl border">
        <div className="flex items-center gap-1.5 text-xs font-semibold text-gray-700">
          <Sparkles className="h-3.5 w-3.5 text-primary" />
          <span>Gemini Flash Lite</span>
        </div>
        <p className="text-[11px] text-gray-400 mt-1">
          Hybrid Vector + BM25 Search & Reranking Active
        </p>
      </div>
    </div>
  );
}