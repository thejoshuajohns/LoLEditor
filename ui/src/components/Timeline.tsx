import React, { useState } from 'react';
import { api } from '../hooks/useApi';

interface TimelineProps {
  project: any;
  onRender: () => void;
  onRefresh: () => void;
}

function formatTime(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, '0')}`;
}

function scoreColor(score: number): string {
  if (score >= 0.8) return 'from-lol-red to-orange-500';
  if (score >= 0.6) return 'from-orange-500 to-lol-gold';
  if (score >= 0.4) return 'from-lol-gold to-lol-blue';
  return 'from-lol-blue to-cyan-400';
}

function eventTypeIcon(type: string): string {
  const icons: Record<string, string> = {
    pentakill: 'PK',
    multi_kill: 'MK',
    kill: 'K',
    teamfight: 'TF',
    objective: 'OBJ',
    burst_combo: 'BC',
    ace: 'ACE',
    death: 'D',
    scoreboard_spike: 'SB',
    scene_change: 'SC',
    audio_excitement: 'AU',
    transcript_hype: 'TX',
  };
  return icons[type] || 'HL';
}

export function Timeline({ project, onRender, onRefresh }: TimelineProps) {
  const [selectedClip, setSelectedClip] = useState<number | null>(0);

  const editPlan = project.edit_plan;
  const clips = editPlan?.clips || [];
  const duration = project.duration_seconds || editPlan?.source_duration || 1;
  const content = project.content_package || {};
  const overlayItems = project.overlay_assets?.items || [];

  const handleDeleteClip = async (index: number) => {
    await api.updateClips(project.id, [{ index, delete: true }]);
    await onRefresh();
    setSelectedClip(null);
  };

  const handleExtendClip = async (index: number, startDelta: number, endDelta: number) => {
    const clip = clips[index];
    if (!clip) return;
    await api.updateClips(project.id, [{
      index,
      start: Math.max(0, clip.start + startDelta),
      end: Math.min(duration, clip.end + endDelta),
    }]);
    await onRefresh();
  };

  const handleReorder = async (index: number, direction: -1 | 1) => {
    const nextIndex = index + direction;
    if (nextIndex < 0 || nextIndex >= clips.length) return;
    const order = clips.map((_: any, idx: number) => idx);
    [order[index], order[nextIndex]] = [order[nextIndex], order[index]];
    await api.reorderClips(project.id, order);
    await onRefresh();
    setSelectedClip(nextIndex);
  };

  return (
    <div className="h-full min-h-0 overflow-y-auto px-6 py-8">
      <div className="mx-auto max-w-7xl">
        <div className="rounded-[30px] border border-lol-gold/10 bg-[radial-gradient(circle_at_top_left,rgba(244,211,94,0.12),transparent_26%),linear-gradient(180deg,rgba(9,20,40,0.97),rgba(1,10,19,0.96))] p-6 shadow-[0_30px_80px_rgba(0,0,0,0.35)]">
          <div className="flex flex-wrap items-start justify-between gap-4">
            <div>
              <p className="text-xs uppercase tracking-[0.35em] text-lol-blue/70">Timeline Editor</p>
              <h2 className="mt-2 text-3xl font-semibold text-lol-gold-light">
                Refine the final montage before render
              </h2>
              <p className="mt-3 max-w-3xl text-sm leading-6 text-gray-400">
                Delete weaker clips, extend the setup or aftermath, and reorder beats if you want a
                different narrative flow. Generated YouTube metadata and overlay assets are previewed beside the timeline.
              </p>
            </div>

            <div className="flex flex-wrap gap-3">
              <button onClick={onRefresh} className="btn-secondary">
                Refresh
              </button>
              <button onClick={onRender} className="btn-primary">
                Render Highlight Package
              </button>
            </div>
          </div>

          <div className="mt-7 rounded-[24px] border border-white/5 bg-black/20 p-5">
            <div className="mb-3 flex justify-between text-[11px] uppercase tracking-[0.2em] text-gray-500">
              <span>Match timeline</span>
              <span>{clips.length} clips selected</span>
            </div>

            <div className="flex justify-between text-[10px] text-gray-600">
              {Array.from({ length: 9 }, (_, index) => (
                <span key={index}>{formatTime((duration / 8) * index)}</span>
              ))}
            </div>

            <div className="relative mt-3 h-24 overflow-hidden rounded-[22px] border border-white/5 bg-[linear-gradient(180deg,rgba(10,20,40,0.9),rgba(1,10,19,0.95))]">
              {project.vision_windows?.map((window: any, index: number) => {
                const left = (window.start / duration) * 100;
                const width = ((window.end - window.start) / duration) * 100;
                const opacity = Math.max(0.08, Math.min(0.7, window.score));
                return (
                  <div
                    key={`${window.start}-${index}`}
                    className="absolute inset-y-0 rounded-full"
                    style={{
                      left: `${left}%`,
                      width: `${Math.max(0.25, width)}%`,
                      background: `linear-gradient(180deg, rgba(10, 200, 185, ${opacity}), rgba(10, 200, 185, ${opacity * 0.35}))`,
                    }}
                  />
                );
              })}

              {clips.map((clip: any, index: number) => {
                const left = (clip.start / duration) * 100;
                const width = ((clip.end - clip.start) / duration) * 100;
                const active = selectedClip === index;
                return (
                  <button
                    key={`${clip.start}-${clip.end}-${index}`}
                    onClick={() => setSelectedClip(index)}
                    className={`absolute top-4 h-16 rounded-2xl border text-left transition-all ${
                      active
                        ? 'border-lol-gold bg-lol-gold/25 shadow-[0_0_0_1px_rgba(244,211,94,0.25)]'
                        : 'border-lol-gold/30 bg-lol-gold/10 hover:bg-lol-gold/15'
                    }`}
                    style={{ left: `${left}%`, width: `${Math.max(1, width)}%` }}
                    title={`${clip.label} (${formatTime(clip.start)} - ${formatTime(clip.end)})`}
                  >
                    <div className="px-3 py-2">
                      <div className="truncate text-xs font-medium text-lol-gold-light">{clip.label}</div>
                      <div className="mt-1 text-[11px] text-gray-300">{formatTime(clip.duration)}</div>
                    </div>
                  </button>
                );
              })}
            </div>
          </div>

          <div className="mt-6 grid gap-6 xl:grid-cols-[0.9fr_1.1fr_0.8fr]">
            <div className="rounded-[24px] border border-white/5 bg-black/20 p-4">
              <p className="text-xs uppercase tracking-[0.25em] text-gray-500">Clips</p>
              <div className="mt-4 space-y-2">
                {clips.map((clip: any, index: number) => (
                  <button
                    key={`${clip.start}-${clip.end}-${index}`}
                    onClick={() => setSelectedClip(index)}
                    className={`w-full rounded-2xl border px-4 py-3 text-left transition-all ${
                      selectedClip === index
                        ? 'border-lol-gold/40 bg-lol-gold/10'
                        : 'border-white/5 bg-black/20 hover:border-lol-blue/20 hover:bg-lol-blue/5'
                    }`}
                  >
                    <div className="flex items-start gap-3">
                      <div className={`flex h-10 w-10 items-center justify-center rounded-2xl bg-gradient-to-br ${scoreColor(clip.score)} text-[11px] font-semibold text-black`}>
                        {eventTypeIcon(clip.event_type)}
                      </div>
                      <div className="min-w-0 flex-1">
                        <div className="truncate text-sm font-medium text-white">{clip.label}</div>
                        <div className="mt-1 text-xs text-gray-500">
                          {formatTime(clip.start)} - {formatTime(clip.end)} • {(clip.score * 100).toFixed(0)}%
                        </div>
                      </div>
                    </div>
                  </button>
                ))}
              </div>
            </div>

            <div className="rounded-[24px] border border-white/5 bg-black/20 p-5">
              {selectedClip !== null && clips[selectedClip] ? (
                <ClipDetail
                  clip={clips[selectedClip]}
                  index={selectedClip}
                  totalClips={clips.length}
                  onDelete={() => handleDeleteClip(selectedClip)}
                  onMoveUp={() => handleReorder(selectedClip, -1)}
                  onMoveDown={() => handleReorder(selectedClip, 1)}
                  onExtendStart={(seconds) => handleExtendClip(selectedClip, -seconds, 0)}
                  onExtendEnd={(seconds) => handleExtendClip(selectedClip, 0, seconds)}
                />
              ) : (
                <div className="flex h-full items-center justify-center text-gray-600">
                  Select a clip to inspect or edit it.
                </div>
              )}
            </div>

            <div className="space-y-5">
              <div className="rounded-[24px] border border-lol-blue/15 bg-black/20 p-5">
                <p className="text-xs uppercase tracking-[0.25em] text-lol-blue/70">Generated Metadata</p>
                <div className="mt-4 rounded-2xl border border-white/5 bg-black/20 p-4">
                  <div className="text-xs uppercase tracking-[0.2em] text-gray-500">Title</div>
                  <div className="mt-2 text-base font-semibold text-lol-gold-light">
                    {content.title || 'Will be generated at analysis time'}
                  </div>
                </div>
                <div className="mt-4 rounded-2xl border border-white/5 bg-black/20 p-4">
                  <div className="text-xs uppercase tracking-[0.2em] text-gray-500">Tags</div>
                  <div className="mt-3 flex flex-wrap gap-2">
                    {(content.tags || []).slice(0, 8).map((tag: string) => (
                      <span key={tag} className="rounded-full border border-lol-blue/20 bg-lol-blue/10 px-3 py-1 text-xs text-lol-blue">
                        {tag}
                      </span>
                    ))}
                  </div>
                </div>
              </div>

              <div className="rounded-[24px] border border-lol-gold/10 bg-black/20 p-5">
                <p className="text-xs uppercase tracking-[0.25em] text-lol-gold/70">Overlay Assets</p>
                <div className="mt-4 space-y-2">
                  {overlayItems.length === 0 ? (
                    <div className="rounded-2xl border border-white/5 bg-black/20 px-4 py-3 text-sm text-gray-500">
                      No overlay assets generated yet.
                    </div>
                  ) : (
                    overlayItems.map((item: any) => (
                      <div key={`${item.type}-${item.path || item.count}`} className="rounded-2xl border border-white/5 bg-black/20 px-4 py-3">
                        <div className="text-sm font-medium text-white">{item.type.replace(/_/g, ' ')}</div>
                        <div className="mt-1 truncate text-xs text-gray-500">{item.path || `${item.count} frames`}</div>
                      </div>
                    ))
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function ClipDetail({
  clip,
  index,
  totalClips,
  onDelete,
  onMoveUp,
  onMoveDown,
  onExtendStart,
  onExtendEnd,
}: {
  clip: any;
  index: number;
  totalClips: number;
  onDelete: () => void;
  onMoveUp: () => void;
  onMoveDown: () => void;
  onExtendStart: (seconds: number) => void;
  onExtendEnd: (seconds: number) => void;
}) {
  return (
    <div>
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <p className="text-xs uppercase tracking-[0.25em] text-gray-500">Selected Clip</p>
          <h3 className="mt-2 text-2xl font-semibold text-lol-gold-light">{clip.label}</h3>
          <p className="mt-1 text-sm text-gray-500">
            Clip {index + 1} of {totalClips} • {clip.event_type}
          </p>
        </div>
        <button onClick={onDelete} className="btn-danger text-xs uppercase tracking-[0.2em]">
          Delete
        </button>
      </div>

      <div className="mt-5 grid gap-3 md:grid-cols-4">
        <Metric label="Start" value={formatTime(clip.start)} />
        <Metric label="End" value={formatTime(clip.end)} />
        <Metric label="Duration" value={`${clip.duration.toFixed(1)}s`} />
        <Metric label="Score" value={`${(clip.score * 100).toFixed(0)}%`} />
      </div>

      <div className="mt-5 rounded-2xl border border-white/5 bg-black/20 p-4">
        <div className="flex items-center justify-between gap-4">
          <span className="text-sm text-gray-300">Detection confidence</span>
          <span className="text-sm text-lol-gold-light">{(clip.score * 100).toFixed(0)}%</span>
        </div>
        <div className="mt-3 h-3 overflow-hidden rounded-full bg-lol-dark">
          <div
            className={`h-full rounded-full bg-gradient-to-r ${scoreColor(clip.score)}`}
            style={{ width: `${Math.max(8, clip.score * 100)}%` }}
          />
        </div>
      </div>

      <div className="mt-5 rounded-2xl border border-white/5 bg-black/20 p-4">
        <p className="text-xs uppercase tracking-[0.25em] text-gray-500">Boundary controls</p>
        <div className="mt-4 flex flex-wrap gap-2">
          <button onClick={() => onExtendStart(5)} className="btn-secondary text-xs">Extend Start -5s</button>
          <button onClick={() => onExtendStart(2)} className="btn-secondary text-xs">Extend Start -2s</button>
          <button onClick={() => onExtendEnd(2)} className="btn-secondary text-xs">Extend End +2s</button>
          <button onClick={() => onExtendEnd(5)} className="btn-secondary text-xs">Extend End +5s</button>
        </div>
      </div>

      <div className="mt-5 rounded-2xl border border-white/5 bg-black/20 p-4">
        <p className="text-xs uppercase tracking-[0.25em] text-gray-500">Narrative order</p>
        <div className="mt-4 flex flex-wrap gap-2">
          <button onClick={onMoveUp} className="btn-secondary text-xs">Move Earlier</button>
          <button onClick={onMoveDown} className="btn-secondary text-xs">Move Later</button>
        </div>
      </div>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-2xl border border-white/5 bg-black/20 p-4">
      <div className="text-xs uppercase tracking-[0.2em] text-gray-500">{label}</div>
      <div className="mt-2 text-lg font-semibold text-lol-gold-light">{value}</div>
    </div>
  );
}
