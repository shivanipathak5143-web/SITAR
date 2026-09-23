const SATELLITES = [
  { id: 'insat3ds', label: 'INSAT-3DS TIR1' },
  { id: 'insat3dr', label: 'INSAT-3DR TIR1' },
  { id: 'goes19', label: 'GOES-19 ABI Ch.13' },
  { id: 'himawari8', label: 'Himawari-8 B13' },
];

const RESOLUTIONS = [
  { id: 15, label: '30 min -> 15 min' },
  { id: 7.5, label: '30 min -> 7.5 min' },
];

const METHODS = [
  { id: 'ai', label: 'AI Model (flow-warp)' },
  { id: 'traditional', label: 'Traditional (linear blend)' },
];

export default function ControlBar({
  satellite,
  onSatellite,
  resolution,
  onResolution,
  method,
  onMethod,
  playing,
  onTogglePlay,
  speed,
  onSpeed,
  onRunInterpolation,
  running,
}) {
  return (
    <div className="control-bar">
      <div className="control-bar__group">
        <label>Dataset</label>
        <select value={satellite} onChange={(e) => onSatellite(e.target.value)}>
          {SATELLITES.map((s) => (
            <option key={s.id} value={s.id}>
              {s.label}
            </option>
          ))}
        </select>
      </div>

      <div className="control-bar__group">
        <label>Target resolution</label>
        <select value={resolution} onChange={(e) => onResolution(Number(e.target.value))}>
          {RESOLUTIONS.map((r) => (
            <option key={r.id} value={r.id}>
              {r.label}
            </option>
          ))}
        </select>
      </div>

      <div className="control-bar__group">
        <label>Interpolation method</label>
        <select value={method} onChange={(e) => onMethod(e.target.value)}>
          {METHODS.map((m) => (
            <option key={m.id} value={m.id}>
              {m.label}
            </option>
          ))}
        </select>
      </div>

      <div className="control-bar__group control-bar__group--playback">
        <label>Playback</label>
        <div className="playback">
          <button className="btn btn--icon" onClick={onTogglePlay}>
            {playing ? '⏸' : '▶'}
          </button>
          <select value={speed} onChange={(e) => onSpeed(Number(e.target.value))}>
            <option value={0.5}>0.5x</option>
            <option value={1}>1x</option>
            <option value={2}>2x</option>
            <option value={4}>4x</option>
          </select>
        </div>
      </div>

      <button className="btn btn--primary" onClick={onRunInterpolation} disabled={running}>
        {running ? 'Running model…' : 'Run interpolation'}
      </button>
    </div>
  );
}
