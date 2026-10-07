"use client";

import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { apiClient } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import {
  TestTube,
  Play,
  RefreshCw,
  Clock,
  CheckCircle2,
  XCircle,
  Plus,
  Sliders,
  Trash2,
  X,
  Target,
  FileCheck,
  Sparkles,
  BarChart2
} from "lucide-react";
import toast from "react-hot-toast";
import { ExperimentRun, ExperimentRunRequest } from "@/types";
import { formatDateTime, cn } from "@/lib/utils";

export default function ExperimentsPage() {
  const qc = useQueryClient();

  const [showNewModal, setShowNewModal] = useState(false);
  const [selectedForCompare, setSelectedForCompare] = useState<string[]>([]);

  // Form state
  const [formName, setFormName] = useState("");
  const [formChunkSize, setFormChunkSize] = useState<number>(800);
  const [formChunkOverlap, setFormChunkOverlap] = useState<number>(150);
  const [formTopKRetrieval, setFormTopKRetrieval] = useState<number>(20);
  const [formTopKRerank, setFormTopKRerank] = useState<number>(6);
  const [formReranker, setFormReranker] = useState<string>("lightweight");
  const [formModel, setFormModel] = useState<string>("gemini-flash-lite-latest");

  // Fetch real PostgreSQL experiment runs
  const {
    data: experiments,
    isLoading,
    isFetching,
    error,
    refetch
  } = useQuery<ExperimentRun[]>({
    queryKey: ["experiments"],
    queryFn: apiClient.experiments.list,
    refetchInterval: (query) => {
      // Auto-poll if any experiment is pending or running
      const hasActive = query.state.data?.some(
        (e) => e.status === "running" || e.status === "pending"
      );
      return hasActive ? 3000 : false;
    }
  });

  // Run mutation
  const runMutation = useMutation({
    mutationFn: (req: ExperimentRunRequest) => apiClient.experiments.run(req),
    onSuccess: (newRun) => {
      toast.success("Experiment started in background");
      qc.invalidateQueries({ queryKey: ["experiments"] });
      setShowNewModal(false);
      setFormName("");
    },
    onError: (err: any) => {
      toast.error(err.response?.data?.detail || "Failed to start experiment");
    }
  });

  // Delete mutation
  const deleteMutation = useMutation({
    mutationFn: (id: string) => apiClient.experiments.delete(id),
    onSuccess: () => {
      toast.success("Experiment deleted");
      qc.invalidateQueries({ queryKey: ["experiments"] });
    },
    onError: () => {
      toast.error("Failed to delete experiment");
    }
  });

  const handleRefresh = async () => {
    try {
      await refetch();
      toast.success("Experiments refreshed");
    } catch {
      toast.error("Failed to refresh experiments");
    }
  };

  const handleStartExperiment = (e: React.FormEvent) => {
    e.preventDefault();
    runMutation.mutate({
      name: formName.trim() || undefined,
      chunk_size: formChunkSize,
      chunk_overlap: formChunkOverlap,
      top_k_retrieval: formTopKRetrieval,
      top_k_rerank: formTopKRerank,
      reranker_type: formReranker,
      model: formModel
    });
  };

  const toggleCompare = (id: string) => {
    setSelectedForCompare((prev) =>
      prev.includes(id) ? prev.filter((item) => item !== id) : [...prev, id]
    );
  };

  const comparedRuns = (experiments || []).filter((e) =>
    selectedForCompare.includes(e.id)
  );

  const getStatusBadge = (status: string) => {
    switch (status) {
      case "completed":
        return (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] font-medium bg-emerald-50 text-emerald-700 border border-emerald-200">
            <CheckCircle2 className="h-3 w-3" />
            Completed
          </span>
        );
      case "running":
        return (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] font-medium bg-blue-50 text-blue-700 border border-blue-200 animate-pulse">
            <Clock className="h-3 w-3 animate-spin" />
            Running
          </span>
        );
      case "failed":
        return (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] font-medium bg-red-50 text-red-700 border border-red-200">
            <XCircle className="h-3 w-3" />
            Failed
          </span>
        );
      default:
        return (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] font-medium bg-amber-50 text-amber-700 border border-amber-200">
            <Clock className="h-3 w-3" />
            Pending
          </span>
        );
    }
  };

  return (
    <div className="space-y-6 max-w-7xl mx-auto pb-10 min-w-0">
      {/* Page Header */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 border-b border-gray-200 pb-5">
        <div>
          <div className="flex items-center gap-2">
            <TestTube className="h-6 w-6 text-primary" />
            <h1 className="text-2xl font-bold text-gray-900 tracking-tight">
              RAG Experiments & Hyperparameters
            </h1>
          </div>
          <p className="text-sm text-gray-500 mt-1">
            Configure, benchmark, and compare RAG parameters (chunk sizes, reranker algorithms, top-k) with PostgreSQL persistence.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={handleRefresh}
            disabled={isFetching}
            className="h-9 gap-1.5 text-xs text-gray-600"
          >
            <RefreshCw className={cn("h-3.5 w-3.5", isFetching && "animate-spin text-primary")} />
            {isFetching ? "Refreshing..." : "Refresh"}
          </Button>

          <Button
            onClick={() => setShowNewModal(true)}
            className="h-9 gap-1.5 text-xs bg-primary hover:bg-primary/90 text-white shadow-xs"
          >
            <Plus className="h-3.5 w-3.5" />
            New Experiment
          </Button>
        </div>
      </div>

      {/* Modal: New Experiment Configuration */}
      {showNewModal && (
        <div className="fixed inset-0 z-50 bg-black/40 backdrop-blur-xs flex items-center justify-center p-4">
          <Card className="w-full max-w-lg shadow-xl bg-white border border-gray-200">
            <CardHeader className="pb-3 border-b border-gray-100 flex flex-row items-center justify-between">
              <div>
                <CardTitle className="text-base font-bold text-gray-900 flex items-center gap-2">
                  <Sliders className="h-4 w-4 text-primary" />
                  Configure Hyperparameter Experiment
                </CardTitle>
                <CardDescription className="text-xs text-gray-500 mt-0.5">
                  Execute benchmark questions against custom RAG retrieval configurations
                </CardDescription>
              </div>
              <button
                type="button"
                onClick={() => setShowNewModal(false)}
                className="p-1 rounded-md text-gray-400 hover:text-gray-600"
              >
                <X className="h-4 w-4" />
              </button>
            </CardHeader>

            <form onSubmit={handleStartExperiment}>
              <CardContent className="space-y-4 pt-4 text-xs">
                <div>
                  <label className="font-semibold text-gray-700 block mb-1">
                    Experiment Name (optional)
                  </label>
                  <input
                    type="text"
                    value={formName}
                    onChange={(e) => setFormName(e.target.value)}
                    placeholder="e.g., chunk800_lightweight_eval"
                    className="w-full rounded-lg border border-gray-300 px-3 py-2 text-xs focus:outline-none focus:ring-1 focus:ring-primary"
                  />
                </div>

                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className="font-semibold text-gray-700 block mb-1">
                      Chunk Size (characters)
                    </label>
                    <select
                      value={formChunkSize}
                      onChange={(e) => setFormChunkSize(Number(e.target.value))}
                      className="w-full rounded-lg border border-gray-300 px-3 py-2 text-xs bg-white"
                    >
                      <option value={500}>500 chars (Fine-grained)</option>
                      <option value={800}>800 chars (Balanced)</option>
                      <option value={1200}>1200 chars (Broad Context)</option>
                    </select>
                  </div>

                  <div>
                    <label className="font-semibold text-gray-700 block mb-1">
                      Chunk Overlap
                    </label>
                    <select
                      value={formChunkOverlap}
                      onChange={(e) => setFormChunkOverlap(Number(e.target.value))}
                      className="w-full rounded-lg border border-gray-300 px-3 py-2 text-xs bg-white"
                    >
                      <option value={50}>50 chars</option>
                      <option value={100}>100 chars</option>
                      <option value={150}>150 chars (Standard)</option>
                    </select>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className="font-semibold text-gray-700 block mb-1">
                      Retrieval Top-K
                    </label>
                    <select
                      value={formTopKRetrieval}
                      onChange={(e) => setFormTopKRetrieval(Number(e.target.value))}
                      className="w-full rounded-lg border border-gray-300 px-3 py-2 text-xs bg-white"
                    >
                      <option value={10}>Top 10 chunks</option>
                      <option value={20}>Top 20 chunks (Default)</option>
                      <option value={30}>Top 30 chunks</option>
                    </select>
                  </div>

                  <div>
                    <label className="font-semibold text-gray-700 block mb-1">
                      Rerank Top-K
                    </label>
                    <select
                      value={formTopKRerank}
                      onChange={(e) => setFormTopKRerank(Number(e.target.value))}
                      className="w-full rounded-lg border border-gray-300 px-3 py-2 text-xs bg-white"
                    >
                      <option value={3}>Top 3 chunks</option>
                      <option value={6}>Top 6 chunks (Default)</option>
                      <option value={10}>Top 10 chunks</option>
                    </select>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className="font-semibold text-gray-700 block mb-1">
                      Reranker Algorithm
                    </label>
                    <select
                      value={formReranker}
                      onChange={(e) => setFormReranker(e.target.value)}
                      className="w-full rounded-lg border border-gray-300 px-3 py-2 text-xs bg-white"
                    >
                      <option value="lightweight">Lightweight Lexical/Metadata</option>
                      <option value="cross_encoder">Cross-Encoder (MS-Marco MiniLM)</option>
                    </select>
                  </div>

                  <div>
                    <label className="font-semibold text-gray-700 block mb-1">
                      Generation Model
                    </label>
                    <select
                      value={formModel}
                      onChange={(e) => setFormModel(e.target.value)}
                      className="w-full rounded-lg border border-gray-300 px-3 py-2 text-xs bg-white"
                    >
                      <option value="gemini-flash-lite-latest">Gemini Flash Lite</option>
                    </select>
                  </div>
                </div>

                <div className="pt-3 border-t border-gray-100 flex items-center justify-end gap-2">
                  <Button
                    type="button"
                    variant="outline"
                    onClick={() => setShowNewModal(false)}
                    className="text-xs"
                  >
                    Cancel
                  </Button>
                  <Button
                    type="submit"
                    disabled={runMutation.isPending}
                    className="gap-1.5 text-xs bg-primary text-white"
                  >
                    {runMutation.isPending ? (
                      <>
                        <Clock className="h-3.5 w-3.5 animate-spin" />
                        Launching...
                      </>
                    ) : (
                      <>
                        <Play className="h-3.5 w-3.5 fill-current" />
                        Run Experiment
                      </>
                    )}
                  </Button>
                </div>
              </CardContent>
            </form>
          </Card>
        </div>
      )}

      {/* Main Experiments View */}
      {isLoading ? (
        <div className="flex flex-col items-center justify-center py-20 text-sm text-gray-500 gap-2">
          <Clock className="h-6 w-6 animate-spin text-primary" />
          Loading experiments from PostgreSQL...
        </div>
      ) : error ? (
        <div className="p-4 rounded-xl bg-red-50 border border-red-200 text-red-700 text-sm">
          Failed to load experiments: {(error as Error).message}
        </div>
      ) : !experiments || experiments.length === 0 ? (
        /* Empty State */
        <Card className="border-dashed border-2 bg-gray-50/50">
          <CardContent className="flex flex-col items-center justify-center py-16 text-center space-y-4">
            <div className="h-12 w-12 rounded-full bg-primary/10 text-primary flex items-center justify-center">
              <TestTube className="h-6 w-6" />
            </div>
            <div className="space-y-1.5 max-w-md">
              <h3 className="text-base font-semibold text-gray-900">No Experiments Recorded Yet</h3>
              <p className="text-xs text-gray-500 leading-relaxed">
                Run an experiment to compare RAG configurations (chunk sizes, retrieval depth, and reranking algorithms). Runs are executed and persisted directly to PostgreSQL.
              </p>
            </div>
            <Button
              onClick={() => setShowNewModal(true)}
              className="gap-2 text-xs bg-primary text-white"
            >
              <Plus className="h-3.5 w-3.5" />
              Start First Experiment
            </Button>
          </CardContent>
        </Card>
      ) : (
        <div className="space-y-6">
          {/* Side-by-Side Comparison Card (if 2+ experiments selected) */}
          {comparedRuns.length >= 2 && (
            <Card className="border-primary/40 shadow-xs bg-primary/[0.01]">
              <CardHeader className="pb-3 border-b border-gray-100 flex flex-row items-center justify-between">
                <div>
                  <CardTitle className="text-base font-bold text-gray-900 flex items-center gap-2">
                    <BarChart2 className="h-4 w-4 text-primary" />
                    Experiment Comparison ({comparedRuns.length} selected)
                  </CardTitle>
                  <CardDescription className="text-xs text-gray-500">
                    Comparing parameter variations against retrieval recall and generation faithfulness
                  </CardDescription>
                </div>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setSelectedForCompare([])}
                  className="text-xs h-7 text-gray-500"
                >
                  Clear Selection
                </Button>
              </CardHeader>
              <CardContent className="p-0 overflow-x-auto">
                <table className="w-full text-xs text-left min-w-[600px]">
                  <thead className="bg-gray-50 text-gray-600 font-semibold border-b border-gray-100">
                    <tr>
                      <th className="px-4 py-3">Metric / Parameter</th>
                      {comparedRuns.map((r) => (
                        <th key={r.id} className="px-4 py-3 font-semibold text-primary">
                          {r.name}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-100">
                    <tr>
                      <td className="px-4 py-2.5 font-medium text-gray-500">Chunk Size</td>
                      {comparedRuns.map((r) => (
                        <td key={r.id} className="px-4 py-2.5 text-gray-900 font-semibold">
                          {r.config?.chunk_size || 800} chars
                        </td>
                      ))}
                    </tr>
                    <tr>
                      <td className="px-4 py-2.5 font-medium text-gray-500">Chunk Overlap</td>
                      {comparedRuns.map((r) => (
                        <td key={r.id} className="px-4 py-2.5 text-gray-700">
                          {r.config?.chunk_overlap || 150} chars
                        </td>
                      ))}
                    </tr>
                    <tr>
                      <td className="px-4 py-2.5 font-medium text-gray-500">Reranker Algorithm</td>
                      {comparedRuns.map((r) => (
                        <td key={r.id} className="px-4 py-2.5 text-gray-700 capitalize">
                          {r.config?.reranker_type || "lightweight"}
                        </td>
                      ))}
                    </tr>
                    <tr>
                      <td className="px-4 py-2.5 font-medium text-gray-500">Retrieval / Rerank K</td>
                      {comparedRuns.map((r) => (
                        <td key={r.id} className="px-4 py-2.5 text-gray-700">
                          {r.config?.top_k_retrieval || 20} / {r.config?.top_k_rerank || 6}
                        </td>
                      ))}
                    </tr>
                    <tr className="bg-emerald-50/40">
                      <td className="px-4 py-2.5 font-semibold text-emerald-800">Recall @ 5</td>
                      {comparedRuns.map((r) => {
                        const rec = r.results?.metrics?.recall_at_5;
                        return (
                          <td key={r.id} className="px-4 py-2.5 font-bold text-emerald-900">
                            {rec !== undefined ? `${(rec * 100).toFixed(1)}%` : "—"}
                          </td>
                        );
                      })}
                    </tr>
                    <tr className="bg-emerald-50/40">
                      <td className="px-4 py-2.5 font-semibold text-emerald-800">Faithfulness</td>
                      {comparedRuns.map((r) => {
                        const faith = r.results?.metrics?.faithfulness;
                        return (
                          <td key={r.id} className="px-4 py-2.5 font-bold text-emerald-900">
                            {faith !== undefined ? `${(faith * 100).toFixed(1)}%` : "—"}
                          </td>
                        );
                      })}
                    </tr>
                    <tr>
                      <td className="px-4 py-2.5 font-medium text-gray-500">Answer Relevance</td>
                      {comparedRuns.map((r) => {
                        const rel = r.results?.metrics?.answer_relevance;
                        return (
                          <td key={r.id} className="px-4 py-2.5 text-gray-800 font-semibold">
                            {rel !== undefined ? `${(rel * 100).toFixed(1)}%` : "—"}
                          </td>
                        );
                      })}
                    </tr>
                    <tr>
                      <td className="px-4 py-2.5 font-medium text-gray-500">Latency</td>
                      {comparedRuns.map((r) => (
                        <td key={r.id} className="px-4 py-2.5 text-gray-700">
                          {r.results?.metrics?.total_latency_seconds ? `${r.results.metrics.total_latency_seconds}s` : "—"}
                        </td>
                      ))}
                    </tr>
                  </tbody>
                </table>
              </CardContent>
            </Card>
          )}

          {/* Experiments Table Card */}
          <Card className="shadow-xs">
            <CardHeader className="pb-3 border-b border-gray-100 flex flex-row items-center justify-between">
              <div>
                <CardTitle className="text-base font-bold text-gray-900">
                  Executed Experiments ({experiments.length})
                </CardTitle>
                <CardDescription className="text-xs text-gray-500">
                  Select 2 or more completed runs to compare configuration parameters and metrics
                </CardDescription>
              </div>
            </CardHeader>

            <CardContent className="p-0 overflow-x-auto">
              <table className="w-full text-xs text-left min-w-[700px]">
                <thead className="bg-gray-50/80 border-b border-gray-100 text-gray-600 font-semibold">
                  <tr>
                    <th className="px-4 py-3 w-10">Compare</th>
                    <th className="px-4 py-3">Experiment Name</th>
                    <th className="px-4 py-3">Status</th>
                    <th className="px-4 py-3">Created</th>
                    <th className="px-4 py-3">Chunk Size</th>
                    <th className="px-4 py-3">Reranker</th>
                    <th className="px-4 py-3">Recall@5</th>
                    <th className="px-4 py-3">Faithfulness</th>
                    <th className="px-4 py-3">Latency</th>
                    <th className="px-4 py-3 text-right">Actions</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-100">
                  {experiments.map((e) => {
                    const isChecked = selectedForCompare.includes(e.id);
                    const rec = e.results?.metrics?.recall_at_5;
                    const faith = e.results?.metrics?.faithfulness;
                    const latency = e.results?.metrics?.total_latency_seconds;

                    return (
                      <tr key={e.id} className="hover:bg-gray-50/60 transition-colors">
                        <td className="px-4 py-3">
                          <input
                            type="checkbox"
                            checked={isChecked}
                            onChange={() => toggleCompare(e.id)}
                            disabled={e.status !== "completed"}
                            className="rounded border-gray-300 text-primary focus:ring-primary h-3.5 w-3.5"
                          />
                        </td>
                        <td className="px-4 py-3 font-semibold text-gray-900 max-w-xs">
                          <div className="truncate">{e.name}</div>
                          {e.status === "failed" && (e.error_message || e.results?.error) && (
                            <div className="text-[11px] font-normal text-red-600 truncate mt-0.5" title={e.error_message || e.results?.error}>
                              Reason: {e.error_message || e.results?.error}
                            </div>
                          )}
                        </td>
                        <td className="px-4 py-3">{getStatusBadge(e.status)}</td>
                        <td className="px-4 py-3 text-gray-500 whitespace-nowrap">
                          {formatDateTime(e.created_at)}
                        </td>
                        <td className="px-4 py-3 text-gray-700">
                          {e.config?.chunk_size || 800} chars
                        </td>
                        <td className="px-4 py-3 text-gray-700 capitalize">
                          {e.config?.reranker_type || "lightweight"}
                        </td>
                        <td className="px-4 py-3 font-semibold text-gray-900">
                          {rec !== undefined ? `${(rec * 100).toFixed(1)}%` : "—"}
                        </td>
                        <td className="px-4 py-3 font-semibold text-gray-900">
                          {faith !== undefined ? `${(faith * 100).toFixed(1)}%` : "—"}
                        </td>
                        <td className="px-4 py-3 text-gray-600">
                          {latency ? `${latency}s` : "—"}
                        </td>
                        <td className="px-4 py-3 text-right">
                          <button
                            type="button"
                            onClick={() => {
                              if (confirm("Delete this experiment run?")) {
                                deleteMutation.mutate(e.id);
                              }
                            }}
                            className="text-gray-400 hover:text-red-600 transition-colors p-1"
                            title="Delete experiment"
                          >
                            <Trash2 className="h-3.5 w-3.5" />
                          </button>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}