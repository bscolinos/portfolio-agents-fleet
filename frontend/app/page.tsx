import Header from "./components/Header";
import SectionNav from "./components/SectionNav";
import Leaderboard from "./components/Leaderboard";
import EquityCurves from "./components/EquityCurves";
import MemoryPanel from "./components/Memory";
import Runs from "./components/Runs";
import Blotter from "./components/Blotter";
import Positions from "./components/Positions";
import Audit from "./components/Audit";
import { AgentsProvider } from "./lib/AgentsContext";

export default function Page() {
  return (
    <AgentsProvider>
      <main className="terminal">
        <Header />
        <SectionNav />
        <div className="content">
          <Leaderboard />
          <EquityCurves />
          <MemoryPanel />
          <Runs />
          <Blotter />
          <Positions />
          <Audit />
        </div>
        <footer className="app-footer">
          <span>
            NVIDIA GPU-accelerated optimization · SingleStore persisted memory &amp;
            trade tracking
          </span>
          <span className="dim">Portfolio Agents — customer demo</span>
        </footer>
      </main>
    </AgentsProvider>
  );
}
