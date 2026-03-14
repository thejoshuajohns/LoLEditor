import React from 'react';

interface HeaderProps {
  project: any;
  view: string;
  serverOnline: boolean;
  wsConnected: boolean;
  onNewProject: () => void;
}

export function Header({ project, view, serverOnline, wsConnected, onNewProject }: HeaderProps) {
  return (
    <header className="titlebar-nodrag mx-4 mt-2 rounded-[24px] border border-white/5 bg-[linear-gradient(180deg,rgba(9,20,40,0.88),rgba(1,10,19,0.82))] px-6 py-4 backdrop-blur-xl">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-5">
          <div>
            <div className="text-[11px] uppercase tracking-[0.42em] text-lol-blue/70">Desktop Studio</div>
            <h1 className="mt-1 text-xl font-semibold text-lol-gold-light">
              LoLEditor <span className="text-white/70">Creator Suite</span>
            </h1>
          </div>

          {project ? (
            <div className="hidden h-10 w-px bg-white/10 md:block" />
          ) : null}

          {project ? (
            <div>
              <div className="text-sm text-gray-300">{project.name}</div>
              <div className="mt-1 flex items-center gap-3 text-xs text-gray-500">
                <span className="rounded-full border border-white/5 bg-black/20 px-2.5 py-1 uppercase tracking-[0.2em]">
                  {view}
                </span>
                <span>{project.status}</span>
              </div>
            </div>
          ) : null}
        </div>

        <div className="flex items-center gap-3">
          <StatusPill label="API" active={serverOnline} activeColor="bg-lol-blue" inactiveColor="bg-lol-red" />
          {project ? (
            <StatusPill label="Live" active={wsConnected} activeColor="bg-lol-gold" inactiveColor="bg-gray-600" />
          ) : null}
          {project ? (
            <button onClick={onNewProject} className="btn-secondary text-xs uppercase tracking-[0.22em]">
              New Project
            </button>
          ) : null}
        </div>
      </div>
    </header>
  );
}

function StatusPill({
  label,
  active,
  activeColor,
  inactiveColor,
}: {
  label: string;
  active: boolean;
  activeColor: string;
  inactiveColor: string;
}) {
  return (
    <div className="flex items-center gap-2 rounded-full border border-white/5 bg-black/20 px-3 py-1.5 text-xs uppercase tracking-[0.22em] text-gray-400">
      <span className={`h-2.5 w-2.5 rounded-full ${active ? activeColor : inactiveColor}`} />
      {label}
    </div>
  );
}
