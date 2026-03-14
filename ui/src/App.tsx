import React, { useEffect, useState } from 'react';
import { DropZone } from './components/DropZone';
import { MetadataPanel } from './components/MetadataPanel';
import { PipelineProgress } from './components/PipelineProgress';
import { Timeline } from './components/Timeline';
import { Header } from './components/Header';
import { useWebSocket } from './hooks/useWebSocket';
import { api } from './hooks/useApi';

type AppView = 'import' | 'configure' | 'processing' | 'timeline' | 'complete';

interface Project {
  id: string;
  name: string;
  input_path: string;
  output_dir: string;
  status: string;
  highlights: any[];
  edit_plan: any;
  content_package?: any;
  overlay_assets?: any;
  artifacts?: any;
  duration_seconds?: number;
  output_path?: string;
  thumbnail_path?: string;
}

export default function App() {
  const [view, setView] = useState<AppView>('import');
  const [project, setProject] = useState<Project | null>(null);
  const [serverOnline, setServerOnline] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const { connected, progress, logs, cancel } = useWebSocket(project?.id || null);

  useEffect(() => {
    const checkHealth = async () => {
      try {
        await api.health();
        setServerOnline(true);
      } catch {
        setServerOnline(false);
      }
    };
    checkHealth();
    const interval = setInterval(checkHealth, 5000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    if (!project || !['processing', 'timeline', 'complete'].includes(view)) return;
    const poll = setInterval(async () => {
      try {
        const updated = await api.getProject(project.id);
        setProject(updated);

        if (updated.status === 'analyzed' && view === 'processing') {
          setView('timeline');
        } else if (updated.status === 'rendered' && view !== 'complete') {
          setView('complete');
        } else if (updated.status === 'uploaded') {
          setView('complete');
        } else if (updated.status === 'error') {
          setError('Pipeline failed. Check the processing log for details.');
        }
      } catch {
        // Ignore polling errors while the user keeps working.
      }
    }, 2000);
    return () => clearInterval(poll);
  }, [project?.id, view]);

  const handleFileImport = async (filePath: string) => {
    setError(null);
    try {
      const name = filePath.split('/').pop()?.replace(/\.[^.]+$/, '') || 'Untitled';
      const created = await api.createProject({ name, input_path: filePath });
      setProject(created);
      setView('configure');
    } catch (err: any) {
      setError(err.message);
    }
  };

  const handleStartProcessing = async (payload: { champion: any; content: any }) => {
    if (!project) return;
    setError(null);
    try {
      await api.updateChampion(project.id, payload.champion);
      await api.updateContent(project.id, payload.content);
      await api.startAnalysis(project.id);
      setView('processing');
    } catch (err: any) {
      setError(err.message);
    }
  };

  const handleRender = async () => {
    if (!project) return;
    setError(null);
    try {
      await api.startRender(project.id);
      setView('processing');
    } catch (err: any) {
      setError(err.message);
    }
  };

  const handleNewProject = () => {
    setProject(null);
    setView('import');
    setError(null);
  };

  return (
    <div className="relative min-h-screen overflow-hidden bg-app-shell text-white">
      <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_top_left,rgba(244,211,94,0.12),transparent_28%),radial-gradient(circle_at_top_right,rgba(10,200,185,0.12),transparent_30%),linear-gradient(180deg,#08111f_0%,#04080f_45%,#02060d_100%)]" />
      <div className="relative flex h-screen min-h-0 flex-col overflow-hidden">
        <div className="titlebar-drag h-8 bg-black/20 backdrop-blur-md" />

        <Header
          project={project}
          view={view}
          serverOnline={serverOnline}
          wsConnected={connected}
          onNewProject={handleNewProject}
        />

        {error ? (
          <div className="mx-5 mt-3 rounded-2xl border border-lol-red/30 bg-lol-red/10 px-4 py-3 text-sm text-lol-red">
            <div className="flex items-center justify-between gap-4">
              <span>{error}</span>
              <button onClick={() => setError(null)} className="text-lol-red/70 hover:text-lol-red">
                Dismiss
              </button>
            </div>
          </div>
        ) : null}

        <main className="relative min-h-0 flex-1">
          {view === 'import' && (
            <DropZone onFileSelect={handleFileImport} serverOnline={serverOnline} />
          )}

          {view === 'configure' && project && (
            <MetadataPanel
              project={project}
              onStart={handleStartProcessing}
              onBack={() => setView('import')}
            />
          )}

          {view === 'processing' && project && (
            <PipelineProgress
              project={project}
              progress={progress}
              logs={logs}
              onCancel={cancel}
            />
          )}

          {view === 'timeline' && project && (
            <Timeline
              project={project}
              onRender={handleRender}
              onRefresh={async () => {
                const updated = await api.getProject(project.id);
                setProject(updated);
              }}
            />
          )}

          {view === 'complete' && project && (
            <CompleteView
              project={project}
              onEditTimeline={() => setView('timeline')}
              onNewProject={handleNewProject}
            />
          )}
        </main>
      </div>
    </div>
  );
}

function CompleteView({
  project,
  onEditTimeline,
  onNewProject,
}: {
  project: Project;
  onEditTimeline: () => void;
  onNewProject: () => void;
}) {
  const content = project.content_package || {};
  const chapters: any[] = content.chapters || [];
  const artifacts = project.artifacts || {};

  return (
    <div className="h-full min-h-0 overflow-y-auto px-6 py-8">
      <div className="mx-auto max-w-6xl">
        <div className="rounded-[32px] border border-lol-gold/10 bg-[radial-gradient(circle_at_top_left,rgba(10,200,185,0.16),transparent_28%),linear-gradient(180deg,rgba(9,20,40,0.98),rgba(1,10,19,0.96))] p-8 shadow-[0_30px_80px_rgba(0,0,0,0.35)]">
          <div className="flex flex-wrap items-start justify-between gap-6">
            <div>
              <p className="text-xs uppercase tracking-[0.35em] text-lol-blue/70">Package Complete</p>
              <h2 className="mt-3 text-4xl font-semibold text-lol-gold-light">Creator-ready highlight drop</h2>
              <p className="mt-3 max-w-3xl text-sm leading-6 text-gray-400">
                Your project now includes a final montage, thumbnail, overlay assets, generated title ideas,
                tags, and chapters. You can still adjust the timeline or start a new package.
              </p>
            </div>

            <div className="flex flex-wrap gap-3">
              <button onClick={onEditTimeline} className="btn-secondary">
                Edit Timeline
              </button>
              <button onClick={onNewProject} className="btn-primary">
                New Project
              </button>
            </div>
          </div>

          <div className="mt-8 grid gap-5 xl:grid-cols-[1.1fr_0.9fr]">
            <div className="rounded-[26px] border border-white/5 bg-black/20 p-5">
              <p className="text-xs uppercase tracking-[0.3em] text-gray-500">Generated Title</p>
              <h3 className="mt-3 text-2xl font-semibold text-lol-gold-light">
                {content.title || 'No title generated'}
              </h3>

              <div className="mt-6 grid gap-4 md:grid-cols-2">
                <div className="rounded-2xl border border-white/5 bg-black/20 p-4">
                  <p className="text-xs uppercase tracking-[0.25em] text-gray-500">Artifacts</p>
                  <div className="mt-3 space-y-2 text-sm text-gray-300">
                    <ArtifactRow label="Final video" value={artifacts.final_video || project.output_path || 'Pending'} />
                    <ArtifactRow label="Thumbnail" value={artifacts.thumbnail || project.thumbnail_path || 'Pending'} />
                    <ArtifactRow label="Workspace" value={project.output_dir} />
                  </div>
                </div>

                <div className="rounded-2xl border border-white/5 bg-black/20 p-4">
                  <p className="text-xs uppercase tracking-[0.25em] text-gray-500">Detection Summary</p>
                  <div className="mt-3 space-y-2 text-sm text-gray-300">
                    <ArtifactRow label="Highlights" value={String(project.highlights?.length || 0)} />
                    <ArtifactRow label="Clips" value={String(project.edit_plan?.highlight_count || 0)} />
                    <ArtifactRow label="Total runtime" value={`${Math.round(project.edit_plan?.total_duration || 0)}s`} />
                  </div>
                </div>
              </div>

              <div className="mt-6 rounded-2xl border border-white/5 bg-black/20 p-4">
                <p className="text-xs uppercase tracking-[0.25em] text-gray-500">Description Draft</p>
                <pre className="mt-3 whitespace-pre-wrap text-sm leading-6 text-gray-300">
                  {content.description || 'No description generated yet.'}
                </pre>
              </div>
            </div>

            <div className="space-y-5">
              <div className="rounded-[26px] border border-lol-blue/15 bg-black/20 p-5">
                <p className="text-xs uppercase tracking-[0.3em] text-lol-blue/70">Title Candidates</p>
                <div className="mt-4 space-y-3">
                  {(content.title_candidates || []).map((candidate: string, index: number) => (
                    <div key={candidate + index} className="rounded-2xl border border-white/5 bg-black/20 px-4 py-3 text-sm text-gray-200">
                      {candidate}
                    </div>
                  ))}
                </div>
              </div>

              <div className="rounded-[26px] border border-lol-gold/10 bg-black/20 p-5">
                <p className="text-xs uppercase tracking-[0.3em] text-lol-gold/70">Chapters and Tags</p>
                <div className="mt-4 rounded-2xl border border-white/5 bg-black/20 p-4">
                  <div className="flex flex-wrap gap-2">
                    {(content.tags || []).map((tag: string) => (
                      <span key={tag} className="rounded-full border border-lol-blue/20 bg-lol-blue/10 px-3 py-1 text-xs text-lol-blue">
                        {tag}
                      </span>
                    ))}
                  </div>
                </div>

                <div className="mt-4 rounded-2xl border border-white/5 bg-black/20 p-4">
                  <div className="space-y-2">
                    {chapters.length === 0 ? (
                      <div className="text-sm text-gray-500">No chapters generated.</div>
                    ) : (
                      chapters.map((chapter) => (
                        <div key={`${chapter.timestamp}-${chapter.label}`} className="flex items-center justify-between gap-4 text-sm text-gray-300">
                          <span className="font-mono text-lol-gold-light">{chapter.timestamp}</span>
                          <span className="flex-1">{chapter.label}</span>
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
    </div>
  );
}

function ArtifactRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between gap-4">
      <span className="text-gray-500">{label}</span>
      <span className="truncate text-right">{value}</span>
    </div>
  );
}
