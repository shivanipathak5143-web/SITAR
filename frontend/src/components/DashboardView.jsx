import { useMemo, useState } from 'react';
import SatelliteCanvas from './SatelliteCanvas';
import FlowOverlay from './FlowOverlay';
import Timeline from './Timeline';
import { bracket, estimateFlowVectors, fieldAt } from '../lib/simulate';

function findRow(rows, t) {
  return rows.find((r) => Math.abs(r.t - t) < 1e-6) ?? rows[0];
}

export default function DashboardView({ scenario, report, time, onScrub, resolution, method }) {
  const [showFlow, setShowFlow] = useState(true);

  const row = useMemo(() => findRow(report.rows, time), [report.rows, time]);
  const nativeField = useMemo(() => {
    const { tA } = bracket(time);
    return fieldAt(scenario, tA);
  }, [scenario, time]);

  const vectors = useMemo(() => {
    const { tA, tB } = bracket(time);
    return estimateFlowVectors(scenario, tA, tB);
  }, [scenario, time]);

  return (
    <div className="dashboard">
      <div className="dashboard__panels">
        <SatelliteCanvas
          field={row.truth}
          label="Ground truth (reference)"
          badge="fine-grained"
          badgeTone="truth"
          time={row.t}
        />
        <SatelliteCanvas
          field={nativeField}
          label="Original captured (30 min cadence)"
          badge="real"
          badgeTone="native"
          time={bracket(time).tA}
        />
        <div className="sat-canvas sat-canvas--overlaid">
          <SatelliteCanvas
            field={row.reconstructed}
            label={`Interpolated output (${resolution} min, ${method === 'ai' ? 'AI model' : 'traditional'})`}
            badge={row.isNative ? 'real' : 'synthetic'}
            badgeTone={row.isNative ? 'native' : 'synthetic'}
            time={row.t}
          />
          {showFlow && !row.isNative && (
            <div className="flow-overlay-wrap">
              <FlowOverlay vectors={vectors} />
            </div>
          )}
        </div>
      </div>

      <label className="flow-toggle">
        <input type="checkbox" checked={showFlow} onChange={(e) => setShowFlow(e.target.checked)} />
        Show estimated motion vectors
      </label>

      <Timeline rows={report.rows} currentT={time} onScrub={onScrub} />

      <div className="frame-readout">
        {row.isNative ? (
          <span className="frame-readout__native">This timestamp is a real captured frame — no reconstruction needed.</span>
        ) : (
          <div className="frame-readout__metrics">
            <span>SSIM <strong>{row.metrics.ssim.toFixed(3)}</strong></span>
            <span>PSNR <strong>{row.metrics.psnr.toFixed(1)} dB</strong></span>
            <span>MSE <strong>{row.metrics.mse.toFixed(4)}</strong></span>
            <span>FSIM <strong>{row.metrics.fsim.toFixed(3)}</strong></span>
          </div>
        )}
      </div>
    </div>
  );
}
