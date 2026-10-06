import * as React from "react"; import { cn } from "@/lib/utils";
export function Badge({ className, variant="default", ...props }: React.HTMLAttributes<HTMLDivElement> & { variant?: "default" | "secondary" | "destructive" | "outline" }) {
  const variants = { default: "bg-primary text-primary-foreground", secondary: "bg-gray-100 text-gray-900", destructive: "bg-red-500 text-white", outline: "text-foreground" };
  return (<div className={cn("inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-semibold transition-colors focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2", variants[variant], className)} {...props} />);
}