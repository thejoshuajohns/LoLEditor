const API_BASE = 'http://127.0.0.1:8420';

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(error.detail || `API error: ${res.status}`);
  }
  return res.json();
}

export const api = {
  // Health
  health: () => request<{ status: string; version: string }>('/api/health'),

  // Projects
  createProject: (data: { name: string; input_path: string; output_dir?: string }) =>
    request<any>('/api/projects', { method: 'POST', body: JSON.stringify(data) }),

  listProjects: () => request<any[]>('/api/projects'),

  getProject: (id: string) => request<any>(`/api/projects/${id}`),

  deleteProject: (id: string) =>
    request<any>(`/api/projects/${id}`, { method: 'DELETE' }),

  // Champion
  updateChampion: (id: string, data: any) =>
    request<any>(`/api/projects/${id}/champion`, { method: 'PUT', body: JSON.stringify(data) }),

  updateContent: (id: string, data: any) =>
    request<any>(`/api/projects/${id}/content`, { method: 'PUT', body: JSON.stringify(data) }),

  // Pipeline
  startAnalysis: (id: string, settings?: any) =>
    request<any>(`/api/projects/${id}/analyze`, { method: 'POST', body: JSON.stringify(settings || {}) }),

  startRender: (id: string, settings?: any) =>
    request<any>(`/api/projects/${id}/render`, { method: 'POST', body: JSON.stringify(settings || {}) }),

  startUpload: (id: string, settings: any) =>
    request<any>(`/api/projects/${id}/upload`, { method: 'POST', body: JSON.stringify(settings) }),

  // Timeline
  getTimeline: (id: string) => request<any>(`/api/projects/${id}/timeline`),

  updateClips: (id: string, updates: any[]) =>
    request<any>(`/api/projects/${id}/timeline/clips`, { method: 'PUT', body: JSON.stringify(updates) }),

  reorderClips: (id: string, order: number[]) =>
    request<any>(`/api/projects/${id}/timeline/reorder`, { method: 'POST', body: JSON.stringify(order) }),

  // Stages
  getPipelineStages: () => request<any[]>('/api/pipeline-stages'),
};
