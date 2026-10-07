import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

/**
 * Safely parses any date string from the backend into a valid Date object.
 * If the string lacks timezone info (e.g. "2026-10-07 17:30:00" or ISO without Z),
 * it ensures it is parsed as UTC because the backend stores datetimes in UTC.
 */
export function parseUtcDate(dateStr: string | Date | null | undefined): Date {
  if (!dateStr) return new Date();
  if (dateStr instanceof Date) return dateStr;
  let normalized = dateStr.trim();
  if (/^\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}/.test(normalized)) {
    normalized = normalized.replace(" ", "T");
  }
  if (!normalized.endsWith("Z") && !/[+-]\d{2}(:\d{2})?$/.test(normalized)) {
    normalized += "Z";
  }
  return new Date(normalized);
}

export function formatDateTime(dateStr: string | Date | null | undefined): string {
  if (!dateStr) return "—";
  const date = parseUtcDate(dateStr);
  return date.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function formatTime(dateStr: string | Date | null | undefined): string {
  if (!dateStr) return "—";
  const date = parseUtcDate(dateStr);
  return date.toLocaleTimeString(undefined, {
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function formatDate(dateStr: string | Date | null | undefined): string {
  if (!dateStr) return "—";
  const date = parseUtcDate(dateStr);
  return date.toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
    year: "numeric",
  });
}

export function formatFileSize(bytes: number) {
  if (bytes === 0) return "0 B";
  const k = 1024;
  const dm = 2;
  const sizes = ["B", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + " " + sizes[i];
}

export function getStatusColor(status: string) {
  switch (status) {
    case "completed":
      return "bg-green-100 text-green-800";
    case "processing":
    case "running":
      return "bg-blue-100 text-blue-800";
    case "failed":
      return "bg-red-100 text-red-800";
    default:
      return "bg-yellow-100 text-yellow-800";
  }
}

export function truncate(str: string, n: number) {
  return str.length > n ? str.substring(0, n - 1) + "..." : str;
}