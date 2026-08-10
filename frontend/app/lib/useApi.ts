"use client";

import { useCallback, useEffect, useRef, useState } from "react";

export interface UseApiState<T> {
  data: T | null;
  error: string | null;
  loading: boolean;
  refreshing: boolean;
  reload: () => void;
  lastUpdated: number | null;
}

/**
 * Fetch + optional polling hook. `fetcher` receives an AbortSignal.
 * `deps` re-triggers a fresh (loading) fetch when they change.
 * `pollMs` (when > 0) does a background refresh without flashing the loader.
 */
export function useApi<T>(
  fetcher: (signal: AbortSignal) => Promise<T>,
  deps: React.DependencyList = [],
  pollMs = 0,
): UseApiState<T> {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [lastUpdated, setLastUpdated] = useState<number | null>(null);
  const [nonce, setNonce] = useState(0);
  const fetcherRef = useRef(fetcher);
  fetcherRef.current = fetcher;

  const reload = useCallback(() => setNonce((n) => n + 1), []);

  // Primary fetch: shows loading spinner. Retriggered by deps + manual reload.
  useEffect(() => {
    const ctrl = new AbortController();
    let alive = true;
    setLoading(true);
    fetcherRef
      .current(ctrl.signal)
      .then((d) => {
        if (!alive) return;
        setData(d);
        setError(null);
        setLastUpdated(Date.now());
      })
      .catch((e: unknown) => {
        if (!alive || ctrl.signal.aborted) return;
        setError(e instanceof Error ? e.message : String(e));
      })
      .finally(() => {
        if (alive) setLoading(false);
      });
    return () => {
      alive = false;
      ctrl.abort();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [nonce, ...deps]);

  // Background poll: silent refresh, no loading flash.
  useEffect(() => {
    if (!pollMs) return;
    const id = setInterval(() => {
      const ctrl = new AbortController();
      setRefreshing(true);
      fetcherRef
        .current(ctrl.signal)
        .then((d) => {
          setData(d);
          setError(null);
          setLastUpdated(Date.now());
        })
        .catch(() => {
          /* keep last good data on transient poll errors */
        })
        .finally(() => setRefreshing(false));
    }, pollMs);
    return () => clearInterval(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pollMs, nonce, ...deps]);

  return { data, error, loading, refreshing, reload, lastUpdated };
}
