import React, { useCallback, useState } from 'react';

interface DropZoneProps {
  onFileSelect: (filePath: string) => void;
  serverOnline: boolean;
}

declare global {
  interface Window {
    electronAPI?: {
      openFile: () => Promise<string | null>;
      openImage: () => Promise<string | null>;
      saveFile: (defaultPath: string) => Promise<string | null>;
      isElectron: boolean;
    };
  }
}

export function DropZone({ onFileSelect, serverOnline }: DropZoneProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [manualPath, setManualPath] = useState('');

  const handleDragOver = useCallback((event: React.DragEvent) => {
    event.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback(() => {
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((event: React.DragEvent) => {
    event.preventDefault();
    setIsDragging(false);
    const files = event.dataTransfer.files;
    if (files.length === 0) return;
    const file = files[0];
    const filePath = (file as any).path || file.name;
    onFileSelect(filePath);
  }, [onFileSelect]);

  const handleBrowse = async () => {
    if (!window.electronAPI?.isElectron) return;
    const path = await window.electronAPI.openFile();
    if (path) onFileSelect(path);
  };

  const handleManualSubmit = (event: React.FormEvent) => {
    event.preventDefault();
    if (manualPath.trim()) onFileSelect(manualPath.trim());
  };

  return (
    <div className="h-full min-h-0 overflow-y-auto px-8 py-10">
      <div className="flex min-h-full items-center justify-center">
        <div className="w-full max-w-6xl">
          <div className="grid gap-6 xl:grid-cols-[1.08fr_0.92fr]">
          <section
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onClick={handleBrowse}
            className={`relative cursor-pointer overflow-hidden rounded-[34px] border p-10 transition-all duration-300 ${
              isDragging
                ? 'border-lol-gold bg-lol-gold/10 shadow-[0_0_0_1px_rgba(244,211,94,0.25),0_30px_80px_rgba(0,0,0,0.35)]'
                : 'border-lol-gold/10 bg-[radial-gradient(circle_at_top_left,rgba(244,211,94,0.12),transparent_28%),radial-gradient(circle_at_bottom_right,rgba(10,200,185,0.12),transparent_32%),linear-gradient(160deg,rgba(9,20,40,0.98),rgba(1,10,19,0.95))] shadow-[0_30px_80px_rgba(0,0,0,0.35)]'
            }`}
          >
            <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-lol-gold/60 to-transparent" />
            <div className="max-w-3xl">
              <p className="text-xs uppercase tracking-[0.38em] text-lol-blue/70">League Highlight Studio</p>
              <h2 className="mt-4 text-5xl font-semibold leading-tight text-lol-gold-light">
                Drag in your OBS recording and turn it into a creator-ready montage
              </h2>
              <p className="mt-5 max-w-2xl text-base leading-7 text-gray-400">
                LoLEditor analyzes scene shifts, OCR cues, transcript callouts, objective moments,
                scoreboard swings, and audio energy to build a smart highlight package automatically.
              </p>
            </div>

            <div className="mt-10 grid gap-5 md:grid-cols-3">
              {[
                { label: 'Multi-signal AI', text: 'Scene changes, OCR, transcript, scoreboard, objectives, and audio combined.' },
                { label: 'Smart editing', text: 'Dynamic clip lengths, setup windows, aftermath, and timeline cleanup.' },
                { label: 'Publishing bundle', text: 'Title ideas, chapters, thumbnail headline, overlays, and YouTube-ready output.' },
              ].map((card) => (
                <div key={card.label} className="rounded-[24px] border border-white/5 bg-black/20 p-5">
                  <div className="text-xs uppercase tracking-[0.25em] text-lol-gold/70">{card.label}</div>
                  <div className="mt-3 text-sm leading-6 text-gray-400">{card.text}</div>
                </div>
              ))}
            </div>

            <div className={`mt-10 rounded-[28px] border border-dashed p-10 text-center transition-all ${
              isDragging ? 'border-lol-gold bg-lol-gold/5' : 'border-lol-blue/25 bg-lol-blue/5'
            }`}>
              <div className="text-[72px] leading-none text-lol-gold-light">{isDragging ? 'DROP' : 'OBS'}</div>
              <div className="mt-5 text-xl font-semibold text-white">
                {isDragging ? 'Release to import your match recording' : 'Drop a recording here or click to browse'}
              </div>
              <div className="mt-2 text-sm text-gray-500">MP4, MOV, MKV, AVI, FLV, WMV, WebM</div>
            </div>
          </section>

            <aside className="space-y-6">
              <div className="rounded-[30px] border border-lol-blue/10 bg-[linear-gradient(180deg,rgba(9,20,40,0.97),rgba(1,10,19,0.95))] p-6 shadow-[0_30px_80px_rgba(0,0,0,0.35)]">
                <p className="text-xs uppercase tracking-[0.35em] text-lol-blue/70">Quick Import</p>
                <h3 className="mt-3 text-2xl font-semibold text-lol-gold-light">Paste a file path</h3>
                <form onSubmit={handleManualSubmit} className="mt-5">
                  <textarea
                    value={manualPath}
                    onChange={(event) => setManualPath(event.target.value)}
                    placeholder="/path/to/league-recording.mp4"
                    rows={5}
                    className="input-field resize-none"
                  />
                  <button type="submit" className="btn-primary mt-4 w-full" disabled={!manualPath.trim() || !serverOnline}>
                    Import Recording
                  </button>
                </form>
              </div>

              <div className="rounded-[30px] border border-lol-gold/10 bg-[linear-gradient(180deg,rgba(18,14,6,0.95),rgba(8,6,2,0.94))] p-6">
                <p className="text-xs uppercase tracking-[0.35em] text-lol-gold/70">Readiness</p>
                <div className="mt-4 rounded-[24px] border border-white/5 bg-black/20 p-5">
                  <div className="flex items-center justify-between text-sm text-gray-300">
                    <span>Desktop API</span>
                    <span className={serverOnline ? 'text-lol-blue' : 'text-lol-red'}>
                      {serverOnline ? 'Online' : 'Offline'}
                    </span>
                  </div>
                  <div className="mt-3 text-sm leading-6 text-gray-500">
                    {serverOnline
                      ? 'The local processing server is ready. Import a recording to create a new project workspace.'
                      : 'Run `lol-video-editor serve` to bring the desktop API online.'}
                  </div>
                </div>
              </div>
            </aside>
          </div>
        </div>
      </div>
    </div>
  );
}
