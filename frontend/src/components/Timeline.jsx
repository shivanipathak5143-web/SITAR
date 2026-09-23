import { DURATION_MIN } from '../lib/simulate';

export default function Timeline({ rows, currentT, onScrub }) {
  return (
    <div className="timeline">
      <div className="timeline__track">
        {rows.map((r) => {
          const pct = (r.t / DURATION_MIN) * 100;
          const isCurrent = Math.abs(r.t - currentT) < 1e-6;
          return (
            <button
              key={r.t}
              className={`timeline__marker ${r.isNative ? 'timeline__marker--native' : 'timeline__marker--synthetic'} ${
                isCurrent ? 'timeline__marker--current' : ''
              }`}
              style={{ left: `${pct}%` }}
              title={`T+${r.t} min — ${r.isNative ? 'captured frame' : 'AI-generated frame'}`}
              onClick={() => onScrub(r.t)}
            />
          );
        })}
        <div className="timeline__playhead" style={{ left: `${(currentT / DURATION_MIN) * 100}%` }} />
      </div>
      <div className="timeline__legend">
        <span><i className="dot dot--native" /> Captured (real)</span>
        <span><i className="dot dot--synthetic" /> Interpolated (synthetic)</span>
      </div>
    </div>
  );
}
