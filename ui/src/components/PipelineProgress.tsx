import React from 'react';
import { ProgressUpdate, LogUpdate } from '../hooks/useWebSocket';

interface PipelineProgressProps {
  project: any;
  progress: Record<string, ProgressUpdate>;
  logs: LogUpdate[];
  onCancel: () => void;
}

const STAGES = [
  { id: 'import', label: 'Video Imported' },
  { id: 'scene_detection', label: 'Scene Detection' },
  { id: 'highlight_detection', label: 'Highlight Detection' },
  { id: 'transcript', label: 'Transcript Processing' },
  { id: 'clip_planning', label: 'Clip Planning' },
  { id: 'rendering', label: 'Rendering' },
  { id: 'encoding', label: 'Encoding' },
  { id: 'thumbnail', label: 'Thumbnail Creation' },
  { id: 'upload', label: 'Uploading' },
];

export function PipelineProgress({ project, progress, logs, onCancel }: PipelineProgressProps) {
  const completed = STAGES.reduce((sum, stage) => sum + (progress[stage.id]?.progress || 0), 0);
  const overall = completed / STAGES.length;

  return (
    <div className="h-full min-h-0 overflow-y-auto px-6 py-8">
      <div className="mx-auto max-w-6xl">
        <div className="grid gap-6 xl:grid-cols-[0.92fr_1.08fr]">
          <section className="rounded-[30px] border border-lol-gold/10 bg-[radial-gradient(circle_at_top_left,rgba(244,211,94,0.12),transparent_26%),linear-gradient(180deg,rgba(9,20,40,0.97),rgba(1,10,19,0.96))] p-6 shadow-[0_30px_80px_rgba(0,0,0,0.35)]">
            <div className="flex items-center justify-between gap-4">
              <div>
                <p className="text-xs uppercase tracking-[0.35em] text-lol-blue/70">Pipeline Live</p>
                <h2 className="mt-2 text-2xl font-semibold text-lol-gold-light">
                  Processing {project.name}
                </h2>
              </div>

              <button onClick={onCancel} className="btn-danger text-xs uppercase tracking-[0.2em]">
                Cancel
              </button>
            </div>

            <div className="mt-6 rounded-[24px] border border-white/5 bg-black/20 p-5">
              <div className="flex items-end justify-between gap-4">
                <div>
                  <p className="text-xs uppercase tracking-[0.25em] text-gray-500">Overall progress</p>
                  <div className="mt-2 text-4xl font-semibold text-lol-gold-light">
                    {Math.round(overall * 100)}%
                  </div>
                </div>
                <div className="text-sm text-gray-400">
                  {project.status === 'rendering' ? 'Final packaging in progress' : 'Analysis and planning in progress'}
                </div>
              </div>

              <div className="mt-5 h-3 overflow-hidden rounded-full bg-lol-dark-light">
                <div
                  className="h-full rounded-full bg-[linear-gradient(90deg,#0AC8B9,#F4D35E)] transition-all duration-500"
                  style={{ width: `${Math.max(4, overall * 100)}%` }}
                />
              </div>
            </div>

            <div className="mt-6 space-y-3">
              {STAGES.map((stage, index) => {
                const stageProgress = progress[stage.id];
                const pct = stageProgress?.progress || 0;
                const isDone = pct >= 1;
                const isActive = pct > 0 && pct < 1;

                return (
                  <div
                    key={stage.id}
                    className={`rounded-2xl border px-4 py-4 transition-all ${
                      isDone
                        ? 'border-lol-blue/15 bg-lol-blue/10'
                        : isActive
                        ? 'border-lol-gold/20 bg-lol-gold/10'
                        : 'border-white/5 bg-black/20'
                    }`}
                  >
                    <div className="flex items-center justify-between gap-4">
                      <div className="flex items-center gap-4">
                        <div className={`flex h-10 w-10 items-center justify-center rounded-2xl text-sm font-semibold ${
                          isDone ? 'bg-lol-blue/20 text-lol-blue' : isActive ? 'bg-lol-gold/20 text-lol-gold-light' : 'bg-lol-dark-light text-gray-500'
                        }`}>
                          {isDone ? 'OK' : `${index + 1}`}
                        </div>
                        <div>
                          <div className="text-sm font-medium text-white">{stage.label}</div>
                          <div className="mt-1 text-xs text-gray-500">
                            {stageProgress?.message || (isDone ? 'Completed' : 'Waiting')}
                          </div>
                        </div>
                      </div>
                      <div className="text-sm text-gray-400">{Math.round(pct * 100)}%</div>
                    </div>

                    <div className="mt-3 h-2 overflow-hidden rounded-full bg-lol-dark">
                      <div
                        className={`h-full rounded-full transition-all duration-500 ${isDone ? 'bg-lol-blue' : 'bg-lol-gold'}`}
                        style={{ width: `${Math.max(pct > 0 ? 4 : 0, pct * 100)}%` }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>

            <div className="mt-6 grid gap-3 md:grid-cols-3">
              <StatCard label="Highlights" value={String(project.highlights?.length || 0)} />
              <StatCard label="Clips" value={String(project.edit_plan?.highlight_count || 0)} />
              <StatCard label="Runtime" value={`${Math.round(project.edit_plan?.total_duration || 0)}s`} />
            </div>
          </section>

          <section className="rounded-[30px] border border-lol-blue/10 bg-[linear-gradient(180deg,rgba(9,20,40,0.97),rgba(1,10,19,0.95))] p-6 shadow-[0_30px_80px_rgba(0,0,0,0.35)]">
            <div className="flex items-center justify-between gap-4">
              <div>
                <p className="text-xs uppercase tracking-[0.35em] text-lol-blue/70">Processing Log</p>
                <h3 className="mt-2 text-xl font-semibold text-lol-gold-light">Live diagnostics</h3>
              </div>
            </div>

            <div className="mt-5 h-[560px] overflow-y-auto rounded-[24px] border border-white/5 bg-black/25 p-4 font-mono text-xs">
              {logs.length === 0 ? (
                <div className="mt-10 text-center text-gray-600">Waiting for pipeline activity...</div>
              ) : (
                <div className="space-y-2">
                  {logs.map((log, index) => (
                    <div
                      key={`${log.timestamp}-${index}`}
                      className={`rounded-xl px-3 py-2 ${
                        log.level === 'error'
                          ? 'bg-lol-red/10 text-lol-red'
                          : log.level === 'warning'
                          ? 'bg-yellow-500/10 text-yellow-400'
                          : 'bg-white/[0.03] text-gray-300'
                      }`}
                    >
                      <span className="mr-3 text-gray-500">
                        {new Date(log.timestamp * 1000).toLocaleTimeString()}
                      </span>
                      {log.message}
                    </div>
                  ))}
                </div>
              )}
            </div>
          </section>
        </div>
      </div>
    </div>
  );
}

function StatCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-2xl border border-white/5 bg-black/20 p-4">
      <div className="text-xs uppercase tracking-[0.22em] text-gray-500">{label}</div>
      <div className="mt-2 text-xl font-semibold text-lol-gold-light">{value}</div>
    </div>
  );
}
