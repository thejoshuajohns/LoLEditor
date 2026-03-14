import React, { useState } from 'react';

interface MetadataPanelProps {
  project: any;
  onStart: (payload: { champion: any; content: any }) => void;
  onBack: () => void;
}

const LANES = ['Top', 'Jungle', 'Mid', 'Bot', 'Support'];
const ROLES = ['Carry', 'Assassin', 'Mage', 'Tank', 'Bruiser', 'Support'];
const RANKS = [
  'Iron', 'Bronze', 'Silver', 'Gold', 'Platinum',
  'Emerald', 'Diamond', 'Master', 'Grandmaster', 'Challenger',
];

export function MetadataPanel({ project, onStart, onBack }: MetadataPanelProps) {
  const [champion, setChampion] = useState({
    champion_name: '',
    champion_png: '',
    lane: '',
    role: '',
    player_name: '',
    rank: '',
    patch: '',
    kills: '',
    deaths: '',
    assists: '',
  });

  const [content, setContent] = useState({
    title: '',
    description: '',
    tags: '',
    thumbnail_headline: '',
    auto_generate_title: true,
    auto_generate_description: true,
    auto_generate_tags: true,
    auto_generate_chapters: true,
  });

  const handleChampionPng = async () => {
    if (!window.electronAPI?.isElectron) return;
    const path = await window.electronAPI.openImage();
    if (path) {
      setChampion((prev) => ({ ...prev, champion_png: path }));
    }
  };

  const handleStart = () => {
    onStart({
      champion: {
        ...champion,
        kills: champion.kills === '' ? null : Number(champion.kills),
        deaths: champion.deaths === '' ? null : Number(champion.deaths),
        assists: champion.assists === '' ? null : Number(champion.assists),
      },
      content: {
        ...content,
        tags: content.tags
          .split(',')
          .map((tag) => tag.trim())
          .filter(Boolean),
      },
    });
  };

  return (
    <div className="h-full min-h-0 overflow-y-auto px-6 py-8">
      <div className="mx-auto max-w-6xl">
        <button onClick={onBack} className="text-xs uppercase tracking-[0.3em] text-gray-500 hover:text-lol-gold-light">
          Back to import
        </button>

        <div className="mt-5 grid gap-6 xl:grid-cols-[1.1fr_0.9fr]">
          <section className="relative overflow-hidden rounded-[28px] border border-lol-gold/10 bg-[radial-gradient(circle_at_top_left,rgba(244,211,94,0.14),transparent_32%),linear-gradient(160deg,rgba(9,20,40,0.96),rgba(1,10,19,0.96))] p-6 shadow-[0_30px_80px_rgba(0,0,0,0.35)]">
            <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-lol-gold/50 to-transparent" />
            <div className="flex items-start justify-between gap-6">
              <div>
                <p className="text-xs uppercase tracking-[0.35em] text-lol-blue/70">Creator Setup</p>
                <h2 className="mt-2 text-3xl font-semibold text-lol-gold-light">
                  Build a branded League highlight package
                </h2>
                <p className="mt-3 max-w-2xl text-sm leading-6 text-gray-400">
                  Add champion context, patch, rank, and optional KDA so the render pipeline can
                  generate overlays, title ideas, chapters, and a thumbnail headline automatically.
                </p>
              </div>

              <div className="min-w-[220px] rounded-2xl border border-lol-blue/20 bg-lol-blue/10 p-4">
                <p className="text-xs uppercase tracking-[0.3em] text-lol-blue/70">Project</p>
                <div className="mt-3 space-y-2 text-sm text-gray-300">
                  <div className="flex justify-between gap-4">
                    <span className="text-gray-500">Recording</span>
                    <span className="truncate text-right">{project.input_path.split('/').pop()}</span>
                  </div>
                  <div className="flex justify-between gap-4">
                    <span className="text-gray-500">Workspace</span>
                    <span className="truncate text-right">{project.output_dir.split('/').pop()}</span>
                  </div>
                </div>
              </div>
            </div>

            <div className="mt-8 grid gap-6 md:grid-cols-2">
              <div className="rounded-2xl border border-white/5 bg-black/20 p-5">
                <p className="text-xs uppercase tracking-[0.3em] text-lol-gold/70">Champion Context</p>
                <div className="mt-4 grid gap-4">
                  <Field label="Champion Name">
                    <input
                      value={champion.champion_name}
                      onChange={(e) => setChampion((prev) => ({ ...prev, champion_name: e.target.value }))}
                      placeholder="Zed, Jinx, Lee Sin..."
                      className="input-field"
                    />
                  </Field>

                  <Field label="Champion Portrait">
                    <div className="flex gap-2">
                      <input
                        value={champion.champion_png}
                        onChange={(e) => setChampion((prev) => ({ ...prev, champion_png: e.target.value }))}
                        placeholder="Path to PNG portrait..."
                        className="input-field flex-1"
                      />
                      <button onClick={handleChampionPng} className="btn-secondary text-xs">
                        Browse
                      </button>
                    </div>
                  </Field>

                  <div className="grid gap-4 md:grid-cols-2">
                    <Field label="Player Name">
                      <input
                        value={champion.player_name}
                        onChange={(e) => setChampion((prev) => ({ ...prev, player_name: e.target.value }))}
                        placeholder="Summoner / creator"
                        className="input-field"
                      />
                    </Field>

                    <Field label="Patch">
                      <input
                        value={champion.patch}
                        onChange={(e) => setChampion((prev) => ({ ...prev, patch: e.target.value }))}
                        placeholder="14.10"
                        className="input-field"
                      />
                    </Field>
                  </div>

                  <div className="grid gap-4 md:grid-cols-3">
                    <Field label="Lane">
                      <select
                        value={champion.lane}
                        onChange={(e) => setChampion((prev) => ({ ...prev, lane: e.target.value }))}
                        className="input-field"
                      >
                        <option value="">Select...</option>
                        {LANES.map((lane) => (
                          <option key={lane} value={lane.toLowerCase()}>{lane}</option>
                        ))}
                      </select>
                    </Field>

                    <Field label="Role">
                      <select
                        value={champion.role}
                        onChange={(e) => setChampion((prev) => ({ ...prev, role: e.target.value }))}
                        className="input-field"
                      >
                        <option value="">Select...</option>
                        {ROLES.map((role) => (
                          <option key={role} value={role.toLowerCase()}>{role}</option>
                        ))}
                      </select>
                    </Field>

                    <Field label="Rank">
                      <select
                        value={champion.rank}
                        onChange={(e) => setChampion((prev) => ({ ...prev, rank: e.target.value }))}
                        className="input-field"
                      >
                        <option value="">Select...</option>
                        {RANKS.map((rank) => (
                          <option key={rank} value={rank.toLowerCase()}>{rank}</option>
                        ))}
                      </select>
                    </Field>
                  </div>

                  <div className="grid gap-4 md:grid-cols-3">
                    <Field label="Kills">
                      <input
                        type="number"
                        min="0"
                        value={champion.kills}
                        onChange={(e) => setChampion((prev) => ({ ...prev, kills: e.target.value }))}
                        placeholder="11"
                        className="input-field"
                      />
                    </Field>
                    <Field label="Deaths">
                      <input
                        type="number"
                        min="0"
                        value={champion.deaths}
                        onChange={(e) => setChampion((prev) => ({ ...prev, deaths: e.target.value }))}
                        placeholder="2"
                        className="input-field"
                      />
                    </Field>
                    <Field label="Assists">
                      <input
                        type="number"
                        min="0"
                        value={champion.assists}
                        onChange={(e) => setChampion((prev) => ({ ...prev, assists: e.target.value }))}
                        placeholder="9"
                        className="input-field"
                      />
                    </Field>
                  </div>
                </div>
              </div>

              <div className="rounded-2xl border border-white/5 bg-black/20 p-5">
                <p className="text-xs uppercase tracking-[0.3em] text-lol-gold/70">Publishing Metadata</p>
                <div className="mt-4 grid gap-4">
                  <Field
                    label="Video Title"
                    hint="Leave blank and keep auto enabled to let the app generate title ideas."
                  >
                    <input
                      value={content.title}
                      onChange={(e) => setContent((prev) => ({ ...prev, title: e.target.value }))}
                      placeholder="Auto-generated if left blank"
                      className="input-field"
                    />
                  </Field>

                  <Field label="Description">
                    <textarea
                      value={content.description}
                      onChange={(e) => setContent((prev) => ({ ...prev, description: e.target.value }))}
                      placeholder="The app can generate a description with chapters and tags..."
                      rows={5}
                      className="input-field resize-none"
                    />
                  </Field>

                  <Field label="Tags">
                    <input
                      value={content.tags}
                      onChange={(e) => setContent((prev) => ({ ...prev, tags: e.target.value }))}
                      placeholder="leagueoflegends, zed, highlights, ranked"
                      className="input-field"
                    />
                  </Field>

                  <Field label="Thumbnail Headline">
                    <input
                      value={content.thumbnail_headline}
                      onChange={(e) => setContent((prev) => ({ ...prev, thumbnail_headline: e.target.value }))}
                      placeholder="Optional manual headline"
                      className="input-field"
                    />
                  </Field>
                </div>

                <div className="mt-5 grid gap-2 rounded-2xl border border-lol-blue/10 bg-lol-blue-dark/40 p-4">
                  <Toggle
                    label="Auto-generate title"
                    checked={content.auto_generate_title}
                    onChange={(checked) => setContent((prev) => ({ ...prev, auto_generate_title: checked }))}
                  />
                  <Toggle
                    label="Auto-generate description"
                    checked={content.auto_generate_description}
                    onChange={(checked) => setContent((prev) => ({ ...prev, auto_generate_description: checked }))}
                  />
                  <Toggle
                    label="Auto-generate tags"
                    checked={content.auto_generate_tags}
                    onChange={(checked) => setContent((prev) => ({ ...prev, auto_generate_tags: checked }))}
                  />
                  <Toggle
                    label="Auto-generate chapter markers"
                    checked={content.auto_generate_chapters}
                    onChange={(checked) => setContent((prev) => ({ ...prev, auto_generate_chapters: checked }))}
                  />
                </div>
              </div>
            </div>

            <div className="mt-6 flex flex-wrap items-center justify-between gap-4 rounded-2xl border border-lol-gold/10 bg-black/20 px-5 py-4">
              <div>
                <p className="text-xs uppercase tracking-[0.3em] text-lol-gold/70">Output Promise</p>
                <p className="mt-2 text-sm text-gray-400">
                  The pipeline will produce a ranked clip timeline, branded overlays, title ideas,
                  chapters, thumbnail headline, and a YouTube-ready render package.
                </p>
              </div>
              <button onClick={handleStart} className="btn-primary px-8 py-3 text-base">
                Analyze and Build Package
              </button>
            </div>
          </section>

          <aside className="space-y-6">
            <div className="rounded-[28px] border border-lol-blue/15 bg-[linear-gradient(180deg,rgba(10,20,40,0.95),rgba(1,10,19,0.95))] p-6 shadow-[0_30px_80px_rgba(0,0,0,0.35)]">
              <p className="text-xs uppercase tracking-[0.35em] text-lol-blue/70">Pipeline Preview</p>
              <h3 className="mt-3 text-xl font-semibold text-lol-gold-light">What gets generated</h3>
              <div className="mt-5 space-y-3">
                {[
                  'Weighted highlight scoring with scene, OCR, transcript, objective, audio, and scoreboard signals',
                  'Smart clip planning with context windows, dynamic length, and post-fight retention',
                  'Champion-aware intro card, end card, portrait overlay, optional KDA overlay, and thumbnail headline',
                  'YouTube title ideas, description draft, tags, and chapters for upload-ready packaging',
                ].map((item) => (
                  <div key={item} className="rounded-2xl border border-white/5 bg-black/20 px-4 py-3 text-sm text-gray-300">
                    {item}
                  </div>
                ))}
              </div>
            </div>

            <div className="rounded-[28px] border border-lol-gold/10 bg-[linear-gradient(180deg,rgba(20,16,8,0.94),rgba(10,8,3,0.92))] p-6">
              <p className="text-xs uppercase tracking-[0.35em] text-lol-gold/70">Branding Snapshot</p>
              <div className="mt-4 rounded-[24px] border border-lol-gold/10 bg-black/25 p-5">
                <div className="flex items-center gap-4">
                  <div className="flex h-16 w-16 items-center justify-center rounded-2xl border border-lol-gold/20 bg-lol-gold/10 text-2xl text-lol-gold-light">
                    {champion.champion_name ? champion.champion_name.slice(0, 1).toUpperCase() : 'L'}
                  </div>
                  <div>
                    <div className="text-xl font-semibold text-lol-gold-light">
                      {champion.champion_name || 'Champion'}
                    </div>
                    <div className="mt-1 text-sm text-gray-400">
                      {[champion.player_name, champion.rank, champion.lane].filter(Boolean).join(' • ') || 'Player • Rank • Lane'}
                    </div>
                  </div>
                </div>
                <div className="mt-5 rounded-2xl border border-white/5 bg-black/20 p-4">
                  <div className="text-xs uppercase tracking-[0.3em] text-gray-500">Thumbnail Headline</div>
                  <div className="mt-2 text-lg font-semibold text-lol-gold-light">
                    {content.thumbnail_headline || 'Auto-generate from your top highlight'}
                  </div>
                </div>
              </div>
            </div>
          </aside>
        </div>
      </div>
    </div>
  );
}

function Field({
  label,
  hint,
  children,
}: {
  label: string;
  hint?: string;
  children: React.ReactNode;
}) {
  return (
    <label className="block">
      <div className="mb-1.5 flex items-center justify-between gap-4">
        <span className="text-xs uppercase tracking-[0.22em] text-gray-400">{label}</span>
        {hint ? <span className="text-[11px] text-gray-600">{hint}</span> : null}
      </div>
      {children}
    </label>
  );
}

function Toggle({
  label,
  checked,
  onChange,
}: {
  label: string;
  checked: boolean;
  onChange: (next: boolean) => void;
}) {
  return (
    <label className="flex items-center justify-between gap-4 rounded-xl border border-white/5 bg-black/20 px-4 py-3 text-sm text-gray-300">
      <span>{label}</span>
      <button
        type="button"
        onClick={() => onChange(!checked)}
        className={`relative h-7 w-12 rounded-full transition-colors ${checked ? 'bg-lol-blue' : 'bg-lol-dark-light'}`}
      >
        <span
          className={`absolute top-1 h-5 w-5 rounded-full bg-white transition-transform ${checked ? 'translate-x-6' : 'translate-x-1'}`}
        />
      </button>
    </label>
  );
}
