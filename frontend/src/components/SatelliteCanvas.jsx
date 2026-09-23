import { useEffect, useRef } from 'react';
import { GRID, fieldToImageData } from '../lib/simulate';

function formatClock(tMin) {
  const h = Math.floor(tMin / 60);
  const m = Math.round(tMin % 60);
  return `T+${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}`;
}

export default function SatelliteCanvas({ field, label, badge, badgeTone = 'native', time, size = 220 }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !field) return;
    const ctx = canvas.getContext('2d');
    const img = fieldToImageData(field, ctx);
    ctx.putImageData(img, 0, 0);
  }, [field]);

  return (
    <div className="sat-canvas">
      <div className="sat-canvas__head">
        <span className="sat-canvas__label">{label}</span>
        {badge && <span className={`badge badge--${badgeTone}`}>{badge}</span>}
      </div>
      <div className="sat-canvas__frame" style={{ width: size, height: size }}>
        <canvas
          ref={canvasRef}
          width={GRID}
          height={GRID}
          style={{ width: size, height: size, imageRendering: 'auto' }}
        />
        <span className="sat-canvas__time">{formatClock(time)}</span>
      </div>
    </div>
  );
}
