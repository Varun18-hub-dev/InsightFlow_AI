"use client";

import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { apiClient } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import {
  Activity,
  Play,
  CheckCircle2,
  XCircle,
  Clock,
  ExternalLink,
  ChevronRight,
  ChevronDown,
  Layers,
  Award,
  Sparkles,
  BarChart2,
  FileCheck,
  Target,
  RefreshCw,
  HelpCircle
} from "lucide-react";
import toast from "react-hot-toast";
import { EvaluationRun } from "@/types";

export default function EvaluationPage() {
  const qc = useQueryClient();
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [expandedQuestions, setExpandedQuestions] = useState<Record<number, boolean>>({});

  const { data: runs, isLoading, refetch } = useQuery<EvaluationRun[]>({
    queryKey: ["evaluations"],
    queryFn: apiClient.evaluations.list,
    refetchInterval: (query) => {
      // Auto-poll if any run is pending or running
      const hasActive = query.state.data?.some((r) => r.status === "running" || r.status === "pending");
      return hasActive ? 3000 : false;
    }
  });

  const runMutation = useMutation({
    mutationFn: () => apiClient.evaluations.run(`rag_eval_${new Date().toISOString().slice(0, 19).replace(/[:T]/g, "_")}`),
    onSuccess: (newRun) => {
      toast.success("Evaluation run started in background");
      qc.invalidateQueries({ queryKey: ["evaluations"] });
      setSelectedRunId(newRun.id);
    },
    onError: (err: any) => {
      toast.error(err.response?.data?.detail || "Failed to start evaluation");
    }
  });

  const activeRun = runs?.find((r) => r.id === selectedRunId) || runs?.[0] || null;

  const toggleQuestion = (idx: number) => {
    setExpandedQuestions((prev) => ({ ...prev, [idx]: !prev[idx] }));
  };

  const getStatusBadge = (status: string) => {
    switch (status) {
      case "completed":
        return (
          <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-xs font-medium bg-emerald-50 text-emerald-700 border border-emerald-200">
            <CheckCircle2 className="h-3 w-3" />
            Completed
          </span>
        );
      case "running":
        return (
          <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-50 text-blue-700 border border-blue-200 animate-pulse">
            <Clock className="h-3 w-3 animate-spin" />
            Running
          </span>
        );
      case "failed":
        return (
          <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-xs font-medium bg-red-50 text-red-700 border border-red-200">
            <XCircle className="h-3 w-3" />
            Failed
          </span>
        );
      default:
        return (
          <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-xs font-medium bg-amber-50 text-amber-700 border border-amber-200">
            <Clock className="h-3 w-3" />
            Pending
          </span>
        );
    }
  };

  const metrics = activeRun?.results?.metrics || null;
  const questionsList = activeRun?.results?.per_question || [];

  return (
    <div className="space-y-6 max-w-7xl mx-auto pb-10">
      {/* Header */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 border-b border-gray-200 pb-5">
        <div>
          <div className="flex items-center gap-2">
            <Activity className="h-6 w-6 text-primary" />
            <h1 className="text-2xl font-bold text-gray-900 tracking-tight">RAG Evaluation Benchmarks</h1>
          </div>
          <p className="text-sm text-gray-500 mt-1">
            Empirical quality benchmarks measuring retrieval recall, hit rate, and answer faithfulness without synthetic approximations.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => refetch()}
            className="h-9 gap-1.5 text-xs text-gray-600"
          >
            <RefreshCw className="h-3.5 w-3.5" />
            Refresh
          </Button>

          <Button
            onClick={() => runMutation.mutate()}
            disabled={runMutation.isPending}
            className="h-9 gap-2 text-xs bg-primary hover:bg-primary/90 text-white shadow-xs"
          >
            {runMutation.isPending ? (
              <>
                <Clock className="h-3.5 w-3.5 animate-spin" />
                Triggering...
              </>
            ) : (
              <>
                <Play className="h-3.5 w-3.5 fill-current" />
                Run Benchmark
              </>
            )}
          </Button>
        </div>
      </div>

      {isLoading ? (
        <div className="flex flex-col items-center justify-center py-20 text-sm text-gray-500 gap-2">
          <Clock className="h-6 w-6 animate-spin text-primary" />
          Loading evaluation benchmarks...
        </div>
      ) : !runs || runs.length === 0 ? (
        /* Empty State */
        <Card className="border-dashed border-2 bg-gray-50/50">
          <CardContent className="flex flex-col items-center justify-center py-16 text-center space-y-4">
            <div className="h-12 w-12 rounded-full bg-primary/10 text-primary flex items-center justify-center">
              <Award className="h-6 w-6" />
            </div>
            <div className="space-y-1.5 max-w-md">
              <h3 className="text-base font-semibold text-gray-900">No Evaluation Runs Executed Yet</h3>
              <p className="text-xs text-gray-500 leading-relaxed">
                Click "Run Benchmark" above to execute real evaluation questions through the hybrid retrieval and reranking pipeline. Results are validated against expected ground truth citations.
              </p>
            </div>
            <Button
              onClick={() => runMutation.mutate()}
              disabled={runMutation.isPending}
              className="gap-2 text-xs bg-primary text-white"
            >
              <Play className="h-3.5 w-3.5 fill-current" />
              Start First Benchmark Run
            </Button>
          </CardContent>
        </Card>
      ) : (
        /* Main Layout: Run Selector & Active Run Details */
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6 items-start">
          {/* Left Column: List of Past Runs */}
          <div className="lg:col-span-1 space-y-3">
            <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wider px-1">
              Historical Runs ({runs.length})
            </h3>
            <div className="space-y-2">
              {runs.map((r) => {
                const isSelected = activeRun?.id === r.id;
                return (
                  <button
                    key={r.id}
                    onClick={() => setSelectedRunId(r.id)}
                    className={`w-full text-left p-3 rounded-xl border transition-all flex flex-col gap-1.5 ${
                      isSelected
                        ? "bg-white border-primary shadow-xs ring-1 ring-primary/20"
                        : "bg-white border-gray-200 hover:border-gray-300 hover:bg-gray-50/50"
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <span className="font-semibold text-xs text-gray-900 truncate">
                        {r.name}
                      </span>
                      {getStatusBadge(r.status)}
                    </div>
                    <div className="flex items-center justify-between text-[11px] text-gray-400">
                      <span>{new Date(r.started_at).toLocaleDateString()}</span>
                      <span>{new Date(r.started_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
                    </div>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Right Column: Run Details & Metrics */}
          <div className="lg:col-span-3 space-y-6">
            {activeRun && (
              <>
                {/* Active Run Overview Card */}
                <Card className="shadow-xs">
                  <CardHeader className="pb-3 border-b border-gray-100">
                    <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-2">
                      <div>
                        <div className="flex items-center gap-2">
                          <CardTitle className="text-lg font-bold text-gray-900">{activeRun.name}</CardTitle>
                          {getStatusBadge(activeRun.status)}
                        </div>
                        <CardDescription className="text-xs text-gray-500 mt-1">
                          Started at {new Date(activeRun.started_at).toLocaleString()}
                          {activeRun.completed_at && ` • Completed at ${new Date(activeRun.completed_at).toLocaleString()}`}
                        </CardDescription>
                      </div>

                      {activeRun.mlflow_run_id && (
                        <div className="flex items-center gap-1.5 text-xs text-primary font-medium bg-primary/5 px-2.5 py-1 rounded-md border border-primary/10">
                          <span>MLflow ID: {activeRun.mlflow_run_id.slice(0, 10)}...</span>
                        </div>
                      )}
                    </div>
                  </CardHeader>

                  <CardContent className="pt-4 space-y-6">
                    {/* Key Metrics Grid */}
                    {metrics ? (
                      <div>
                        <h4 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-3">
                          Pipeline Benchmark Metrics
                        </h4>
                        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3.5">
                          <div className="p-3.5 rounded-xl bg-gray-50 border border-gray-200/80">
                            <div className="flex items-center justify-between text-gray-500 text-xs">
                              <span>Recall @ 5</span>
                              <Target className="h-4 w-4 text-primary" />
                            </div>
                            <div className="text-xl font-bold text-gray-900 mt-1.5">
                              {metrics.recall_at_5 !== undefined ? `${(metrics.recall_at_5 * 100).toFixed(1)}%` : "N/A"}
                            </div>
                            <div className="text-[11px] text-gray-400 mt-0.5">Top-5 relevant retrieval</div>
                          </div>

                          <div className="p-3.5 rounded-xl bg-gray-50 border border-gray-200/80">
                            <div className="flex items-center justify-between text-gray-500 text-xs">
                              <span>Faithfulness</span>
                              <FileCheck className="h-4 w-4 text-emerald-600" />
                            </div>
                            <div className="text-xl font-bold text-gray-900 mt-1.5">
                              {metrics.faithfulness !== undefined ? `${(metrics.faithfulness * 100).toFixed(1)}%` : "N/A"}
                            </div>
                            <div className="text-[11px] text-gray-400 mt-0.5">Grounded in context</div>
                          </div>

                          <div className="p-3.5 rounded-xl bg-gray-50 border border-gray-200/80">
                            <div className="flex items-center justify-between text-gray-500 text-xs">
                              <span>Answer Relevance</span>
                              <Sparkles className="h-4 w-4 text-indigo-600" />
                            </div>
                            <div className="text-xl font-bold text-gray-900 mt-1.5">
                              {metrics.answer_relevance !== undefined ? `${(metrics.answer_relevance * 100).toFixed(1)}%` : "N/A"}
                            </div>
                            <div className="text-[11px] text-gray-400 mt-0.5">Query alignment</div>
                          </div>

                          <div className="p-3.5 rounded-xl bg-gray-50 border border-gray-200/80">
                            <div className="flex items-center justify-between text-gray-500 text-xs">
                              <span>Total Latency</span>
                              <Clock className="h-4 w-4 text-amber-600" />
                            </div>
                            <div className="text-xl font-bold text-gray-900 mt-1.5">
                              {metrics.total_latency_seconds ? `${metrics.total_latency_seconds}s` : "N/A"}
                            </div>
                            <div className="text-[11px] text-gray-400 mt-0.5">Across {metrics.questions_evaluated || 0} questions</div>
                          </div>
                        </div>
                      </div>
                    ) : activeRun.status === "running" ? (
                      <div className="p-8 text-center bg-gray-50 rounded-xl border border-gray-200 space-y-2">
                        <Clock className="h-6 w-6 animate-spin text-primary mx-auto" />
                        <div className="text-sm font-semibold text-gray-800">Evaluation Benchmark in Progress</div>
                        <div className="text-xs text-gray-500">Executing test queries through Hybrid Retriever & Gemini Flash Lite...</div>
                      </div>
                    ) : (
                      <div className="p-4 text-xs text-gray-500 bg-gray-50 rounded-lg border border-gray-200">
                        {activeRun.results?.error || "No metrics recorded for this run."}
                      </div>
                    )}

                    {/* Evaluated Questions Breakdown */}
                    {questionsList.length > 0 && (
                      <div className="space-y-3 pt-2">
                        <h4 className="text-xs font-semibold text-gray-500 uppercase tracking-wider">
                          Question-by-Question Diagnostics ({questionsList.length})
                        </h4>
                        <div className="space-y-2.5">
                          {questionsList.map((qItem: any, idx: number) => {
                            const isExpanded = !!expandedQuestions[idx];
                            return (
                              <div
                                key={idx}
                                className="border border-gray-200 rounded-xl overflow-hidden bg-white shadow-2xs"
                              >
                                <button
                                  type="button"
                                  onClick={() => toggleQuestion(idx)}
                                  className="w-full text-left p-3.5 flex items-center justify-between hover:bg-gray-50/70 transition-colors"
                                >
                                  <div className="flex items-center gap-2.5 pr-4">
                                    <span className="font-semibold text-xs text-primary bg-primary/10 px-2 py-0.5 rounded">
                                      Q{idx + 1}
                                    </span>
                                    <span className="font-medium text-xs text-gray-900">
                                      {qItem.question}
                                    </span>
                                  </div>
                                  <div className="flex items-center gap-2 shrink-0">
                                    {qItem.confidence !== undefined && (
                                      <span className="text-[11px] text-gray-400 bg-gray-100 px-2 py-0.5 rounded">
                                        Conf: {Math.round(qItem.confidence * 100)}%
                                      </span>
                                    )}
                                    {isExpanded ? (
                                      <ChevronDown className="h-4 w-4 text-gray-400" />
                                    ) : (
                                      <ChevronRight className="h-4 w-4 text-gray-400" />
                                    )}
                                  </div>
                                </button>

                                {isExpanded && (
                                  <div className="px-4 pb-4 pt-1 bg-gray-50/50 border-t border-gray-100 text-xs space-y-2">
                                    <div>
                                      <div className="font-semibold text-gray-600 text-[11px]">RAG Generated Answer:</div>
                                      <p className="mt-1 text-gray-800 whitespace-pre-wrap bg-white p-2.5 rounded-lg border border-gray-200">
                                        {qItem.answer || "No response generated."}
                                      </p>
                                    </div>

                                    {qItem.sources && qItem.sources.length > 0 && (
                                      <div>
                                        <div className="font-semibold text-gray-600 text-[11px]">Retrieved Sources:</div>
                                        <div className="flex flex-wrap gap-1.5 mt-1">
                                          {qItem.sources.map((s: string, sIdx: number) => (
                                            <span
                                              key={sIdx}
                                              className="bg-white border border-gray-200 rounded px-2 py-0.5 text-[11px] text-gray-600"
                                            >
                                              {s}
                                            </span>
                                          ))}
                                        </div>
                                      </div>
                                    )}
                                  </div>
                                )}
                              </div>
                            );
                          })}
                        </div>
                      </div>
                    )}
                  </CardContent>
                </Card>
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
}