"use client";

import { createContext, useContext } from "react";
import { api, Agent } from "./api";
import { useApi } from "./useApi";

interface AgentsCtx {
  agents: Agent[];
  loading: boolean;
  error: string | null;
}

const Ctx = createContext<AgentsCtx>({ agents: [], loading: true, error: null });

export function AgentsProvider({ children }: { children: React.ReactNode }) {
  const { data, loading, error } = useApi<Agent[]>(
    (signal) => api.agents(signal),
    [],
    30000,
  );
  return (
    <Ctx.Provider value={{ agents: data ?? [], loading, error }}>
      {children}
    </Ctx.Provider>
  );
}

export function useAgents() {
  return useContext(Ctx);
}
