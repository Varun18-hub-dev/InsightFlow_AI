"use client";

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { apiClient } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import {
  TestTube,
  Layers,
  BarChart3,
  ServerOff,
  Cpu,
  CheckCircle,
  HelpCircle,
  RefreshCw,
  ExternalLink,
  ChevronRight,
  Sliders,
  Award,
  Clock
} from "lucide-react";
import { Experiment } from "@/types";

export default function ExperimentsPage() {
  const [selectedExperiment, setSelectedExperiment] = useState<string | null>(null);

  // Fetch experiments list
  const {
    data: experiments,
    isLoading: isLoadingExperiments,
    isError,
    refetch: refetchExperiments
  } = useQuery<Experiment[]>({
    queryKey: ["experiments"],
    queryFn: apiClient.experiments.list
  });

  const activeExpName = selectedExperiment || experiments?.[0]?.name || null;

  // Fetch runs for the selected experiment
  const {
    data: runs,
    isLoading: isLoadingRuns,
    refetch: refetchRuns
  } = useQuery<any[]>({
    queryKey: ["experiment-runs", activeExpName],
    queryFn: () => (activeExpName ? apiClient.experiments.runs(activeExpName) : Promise.resolve([])),
    enabled: !!activeExpName
  });

  const handleRefresh = () => {
    refetchExperiments();
    if (activeExpName) refetchRuns();
  };

  const isMlflowOffline = !isLoadingExperiments && (!experiments || experiments.length === 0);

  return (
    <div className="space-y-6 max-w-7xl mx-auto pb-10">
      {/* Page Header */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 border-b border-gray-200 pb-5">
        <div>
          <div className="flex items-center gap-2">
            <TestTube className="h-6 w-6 text-primary" />
            <h1 className="text-2xl font-bold text-gray-900 tracking-tight">
              MLflow RAG Experiments & Hyperparameters
            </h1>
          </div>
          <p className="text-sm text-gray-500 mt-1">
            Track, compare, and benchmark parameter variations (chunk sizes, rerankers, top-k) across RAG pipeline runs.
          </p>
        </div>

        <Button
          variant="outline"
          size="sm"
          onClick={handleRefresh}
          className="h-9 gap-1.5 text-xs text-gray-600"
        >
          <RefreshCw className="h-3.5 w-3.5" />
          Refresh Tracking
        </Button>
      </div>

      {isLoadingExperiments ? (
        <div className="flex flex-col items-center justify-center py-20 text-sm text-gray-500 gap-2">
          <Clock className="h-6 w-6 animate-spin text-primary" />
          Connecting to MLflow Tracking Server...
        </div>
      ) : isMlflowOffline ? (
        /* MLflow Offline / Unreachable Notice */
        <div className="space-y-6">
          <div className="p-4 rounded-xl bg-amber-50/80 border border-amber-200 text-amber-900 flex items-start gap-3">
            <ServerOff className="h-5 w-5 text-amber-600 shrink-0 mt-0.5" />
            <div className="text-xs space-y-1">
              <span className="font-semibold text-sm">MLflow Tracking Server Unavailable</span>
              <p className="text-amber-700 leading-relaxed">
                The MLflow tracking server is not reachable from this environment. The core RAG pipeline and evaluation continue functioning normally with direct PostgreSQL persistence.
              </p>
            </div>
          </div>

          <Card className="shadow-xs border-gray-200">
            <CardHeader>
              <CardTitle className="text-base font-semibold text-gray-900 flex items-center gap-2">
                <Sliders className="h-4 w-4 text-primary" />
                How Experiment Tracking Operates
              </CardTitle>
              <CardDescription className="text-xs text-gray-500">
                InsightFlow AI integrates with MLflow to evaluate parameter tuning across production and staging runs.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4 text-xs text-gray-600">
              <p className="leading-relaxed">
                When MLflow is active, benchmark evaluations logged from the <strong>Evaluation</strong> tab automatically push hyperparameter runs to the MLflow tracking registry:
              </p>
              <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-3">
                <div className="p-3 bg-gray-50 rounded-lg border border-gray-100">
                  <div className="font-semibold text-gray-800">Chunk Size & Overlap</div>
                  <div className="text-gray-400 mt-1">E.g., 500 / 800 / 1200 characters</div>
                </div>
                <div className="p-3 bg-gray-50 rounded-lg border border-gray-100">
                  <div className="font-semibold text-gray-800">Reranker Algorithms</div>
                  <div className="text-gray-400 mt-1">Lightweight vs Cross-Encoder MiniLM</div>
                </div>
                <div className="p-3 bg-gray-50 rounded-lg border border-gray-100">
                  <div className="font-semibold text-gray-800">Top-K Selection</div>
                  <div className="text-gray-400 mt-1">Retrieval (20) & Reranked (6)</div>
                </div>
                <div className="p-3 bg-gray-50 rounded-lg border border-gray-100">
                  <div className="font-semibold text-gray-800">Generation Models</div>
                  <div className="text-gray-400 mt-1">Gemini Flash Lite (gemini-flash-lite-latest)</div>
                </div>
              </div>

              <div className="p-3.5 rounded-lg bg-gray-50 border border-gray-200 font-mono text-[11px] text-gray-700">
                <strong>To enable MLflow locally:</strong>
                <pre className="mt-1 text-gray-600">docker compose up mlflow</pre>
              </div>
            </CardContent>
          </Card>
        </div>
      ) : (
        /* MLflow Active State: Experiment and Runs UI */
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6 items-start">
          {/* Left Column: Experiments List */}
          <div className="lg:col-span-1 space-y-3">
            <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wider px-1">
              Registered Experiments ({experiments?.length || 0})
            </h3>
            <div className="space-y-2">
              {experiments?.map((exp) => {
                const isSelected = activeExpName === exp.name;
                return (
                  <button
                    key={exp.experiment_id}
                    onClick={() => setSelectedExperiment(exp.name)}
                    className={`w-full text-left p-3.5 rounded-xl border transition-all flex flex-col gap-1 ${
                      isSelected
                        ? "bg-white border-primary shadow-xs ring-1 ring-primary/20"
                        : "bg-white border-gray-200 hover:border-gray-300 hover:bg-gray-50/50"
                    }`}
                  >
                    <div className="font-semibold text-xs text-gray-900 truncate">
                      {exp.name}
                    </div>
                    <div className="text-[11px] text-gray-400 flex items-center justify-between">
                      <span>ID: {exp.experiment_id}</span>
                      <span className="capitalize">{exp.lifecycle_stage || "active"}</span>
                    </div>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Right Column: Runs Comparison Table */}
          <div className="lg:col-span-3 space-y-6">
            <Card className="shadow-xs">
              <CardHeader className="pb-3 border-b border-gray-100">
                <div className="flex items-center justify-between">
                  <div>
                    <CardTitle className="text-base font-bold text-gray-900">
                      Experiment: {activeExpName}
                    </CardTitle>
                    <CardDescription className="text-xs text-gray-500 mt-0.5">
                      Comparing parameter configurations against evaluation performance
                    </CardDescription>
                  </div>
                  <span className="text-xs text-gray-500 bg-gray-100 px-2.5 py-1 rounded-md font-medium">
                    {runs ? `${runs.length} runs` : "0 runs"}
                  </span>
                </div>
              </CardHeader>

              <CardContent className="p-0 overflow-x-auto">
                {isLoadingRuns ? (
                  <div className="p-12 text-center text-xs text-gray-500 flex items-center justify-center gap-2">
                    <Clock className="h-4 w-4 animate-spin text-primary" />
                    Loading runs for {activeExpName}...
                  </div>
                ) : !runs || runs.length === 0 ? (
                  <div className="p-12 text-center text-xs text-gray-400">
                    No runs recorded yet in this experiment. Run an evaluation from the Evaluation tab to log metrics.
                  </div>
                ) : (
                  <table className="w-full text-xs text-left">
                    <thead className="bg-gray-50/80 border-b border-gray-100 text-gray-600 font-medium">
                      <tr>
                        <th className="px-4 py-3">Run ID</th>
                        <th className="px-4 py-3">Chunk Size</th>
                        <th className="px-4 py-3">Reranker</th>
                        <th className="px-4 py-3">Recall@5</th>
                        <th className="px-4 py-3">Faithfulness</th>
                        <th className="px-4 py-3">Latency</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-gray-100">
                      {runs.map((r, idx) => {
                        const runId = r["run_id"] || r["info.run_id"] || `run_${idx}`;
                        const chunkSize = r["params.chunk_size"] || "800";
                        const reranker = r["params.reranker"] || "lightweight";
                        const recall = r["metrics.recall_at_5"] !== undefined ? `${(r["metrics.recall_at_5"] * 100).toFixed(1)}%` : "—";
                        const faithfulness = r["metrics.faithfulness"] !== undefined ? `${(r["metrics.faithfulness"] * 100).toFixed(1)}%` : "—";
                        const latency = r["metrics.total_latency_seconds"] ? `${r["metrics.total_latency_seconds"]}s` : "—";

                        return (
                          <tr key={runId} className="hover:bg-gray-50/50 transition-colors">
                            <td className="px-4 py-3 font-mono text-[11px] text-primary">
                              {runId.slice(0, 8)}...
                            </td>
                            <td className="px-4 py-3 text-gray-800">{chunkSize} chars</td>
                            <td className="px-4 py-3 text-gray-800">{reranker}</td>
                            <td className="px-4 py-3 font-semibold text-gray-900">{recall}</td>
                            <td className="px-4 py-3 text-gray-800">{faithfulness}</td>
                            <td className="px-4 py-3 text-gray-600">{latency}</td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                )}
              </CardContent>
            </Card>
          </div>
        </div>
      )}
    </div>
  );
}