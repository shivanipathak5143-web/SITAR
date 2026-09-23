import { useEffect, useMemo, useState } from 'react';
import ControlBar from './components/ControlBar';
import DashboardView from './components/DashboardView';
import MetricsView from './components/MetricsView';
import AboutView from './components/AboutView';
import { makeScenario } from './lib/simulate';
import { buildReport } from './lib/pipeline';
import './App.css';

const SATELLITE_LABELS = {
  insat3ds: 'INSAT-3DS TIR1',
  insat3dr: 'INSAT-3DR TIR1',
  goes19: 'GOES-19 ABI Ch.13',
  himawari8: 'Himawari-8 B13',
};

export default function App() {
  const [tab, setTab] = useState('dashboard');
  const [satellite, setSatellite] = useState('insat3ds');
  const [resolution, setResolution] = useState(15);
  const [method, setMethod] = useState('ai');
  const [scenario, setScenario] = useState(() => makeScenario(7));
  const [running, setRunning] = useState(false);

  const [frameIdx, setFrameIdx] = useState(0);
  const [playing, setPlaying] = useState(true);
  const [speed, setSpeed] = useState(1);

  const reportAI = useMemo(() => buildReport(scenario, resolution, 'ai'), [scenario, resolution]);
  const reportTraditional = useMemo(
    () => buildReport(scenario, resolution, 'traditional'),
    [scenario, resolution],
  );
  const report = method === 'ai' ? reportAI : reportTraditional;
  const rowCount = report.rows.length;
  const time = report.rows[Math.min(frameIdx, rowCount - 1)]?.t ?? 0;

  // Resolution changes the number/spacing of frames, so the current index
  // may no longer be valid -- snap back to the start of the sequence.
  useEffect(() => {
    setFrameIdx(0);
  }, [resolution]);

  useEffect(() => {
    if (!playing) return;
    const id = setInterval(() => {
      setFrameIdx((i) => (i + 1) % rowCount);
    }, 600 / speed);
    return () => clearInterval(id);
  }, [playing, speed, rowCount]);

  const handleRunInterpolation = () => {
    setRunning(true);
    window.setTimeout(() => {
      setScenario(makeScenario(Math.floor(Math.random() * 100000)));
      setFrameIdx(0);
      setRunning(false);
    }, 700);
  };

  return (
    <div className="app-shell">
      <header className="app-header">
        <div className="app-header__title">
          <span className="app-header__badge">PROTOTYPE</span>
          <h1>Fill in the Frames, Seamlessly</h1>
          <p>AI/ML optical-flow frame interpolation for geostationary satellite imagery</p>
        </div>
        <nav className="app-tabs">
          {['dashboard', 'metrics', 'about'].map((t) => (
            <button
              key={t}
              className={`app-tabs__btn ${tab === t ? 'app-tabs__btn--active' : ''}`}
              onClick={() => setTab(t)}
            >
              {t === 'dashboard' ? 'Dashboard' : t === 'metrics' ? 'Validation Report' : 'Pipeline & About'}
            </button>
          ))}
        </nav>
      </header>

      <main className="app-main">
        {tab === 'dashboard' && (
          <>
            <ControlBar
              satellite={satellite}
              onSatellite={setSatellite}
              resolution={resolution}
              onResolution={setResolution}
              method={method}
              onMethod={setMethod}
              playing={playing}
              onTogglePlay={() => setPlaying((p) => !p)}
              speed={speed}
              onSpeed={setSpeed}
              onRunInterpolation={handleRunInterpolation}
              running={running}
            />
            <DashboardView
              scenario={scenario}
              report={report}
              time={time}
              onScrub={(t) => {
                setPlaying(false);
                const idx = report.rows.findIndex((r) => r.t === t);
                if (idx !== -1) setFrameIdx(idx);
              }}
              resolution={resolution}
              method={method}
            />
          </>
        )}

        {tab === 'metrics' && (
          <MetricsView
            reportAI={reportAI}
            reportTraditional={reportTraditional}
            resolution={resolution}
            satelliteLabel={SATELLITE_LABELS[satellite]}
          />
        )}

        {tab === 'about' && <AboutView />}
      </main>

      <footer className="app-footer">
        Synthetic demo data — no real satellite imagery is processed in this prototype. See "Pipeline & About" for
        the intended production data flow.
      </footer>
    </div>
  );
}
