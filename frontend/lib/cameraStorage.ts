// Simple browser storage for camera configuration (works on Vercel/static hosting)
// Stores only static camera config; volatile cell data should NOT be persisted.

export interface StoredCameraConfig {
  id: string;
  name: string;
  lat: number;
  lng: number;
  status: string;
  direction: number;
  fovRadius: number;
}

export interface StoredCoverageCircle {
  center: { lat: number; lng: number };
  radius: number;
}

export interface CellsSnapshotCamera {
  camera_id: string;
  cells: number[][];
}

export interface CellsSnapshot {
  timestamp: number;
  cameras: CellsSnapshotCamera[];
}

const CAMERAS_KEY = "cameraConfig";
const COVERAGE_KEY = "cameraCoverage";
const CELLS_SNAPSHOT_KEY = "cameraCellsLatest";

export function loadCamerasFromStorage(): StoredCameraConfig[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = window.localStorage.getItem(CAMERAS_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw) as StoredCameraConfig[];
    if (!Array.isArray(parsed)) return [];
    return parsed;
  } catch {
    return [];
  }
}

export function saveCamerasToStorage(cameras: StoredCameraConfig[]): void {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(CAMERAS_KEY, JSON.stringify(cameras));
  } catch {
    // ignore write errors (e.g., quota)
  }
}

export function loadCoverageFromStorage(): StoredCoverageCircle | null {
  if (typeof window === "undefined") return null;
  try {
    const raw = window.localStorage.getItem(COVERAGE_KEY);
    if (!raw) return null;
    return JSON.parse(raw) as StoredCoverageCircle;
  } catch {
    return null;
  }
}

export function saveCoverageToStorage(coverage: StoredCoverageCircle | null): void {
  if (typeof window === "undefined") return;
  try {
    if (coverage) {
      window.localStorage.setItem(COVERAGE_KEY, JSON.stringify(coverage));
    } else {
      window.localStorage.removeItem(COVERAGE_KEY);
    }
  } catch {
    // ignore
  }
}

// Optional: cache the latest cells snapshot to survive reloads (keep only the most recent)
export function saveLatestCellsSnapshot(snapshot: CellsSnapshot): void {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(CELLS_SNAPSHOT_KEY, JSON.stringify(snapshot));
  } catch {
    // ignore
  }
}

export function loadLatestCellsSnapshot(): CellsSnapshot | null {
  if (typeof window === "undefined") return null;
  try {
    const raw = window.localStorage.getItem(CELLS_SNAPSHOT_KEY);
    if (!raw) return null;
    return JSON.parse(raw) as CellsSnapshot;
  } catch {
    return null;
  }
}


