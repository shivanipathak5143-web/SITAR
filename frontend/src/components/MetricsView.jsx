import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

const METRIC_DEFS = [
  { key: 'ssim', label: 'SSIM', higherBetter: true, decimals: 3 },
  { key: 'psnr', label: 'PSNR (dB)', higherBetter: true, decimals: 1 },
  { key: 'fsim', label: 'FSIM (approx.)', higherBetter: true, decimals: 3 },
  { key: 'mse', label: 'MSE', higherBetter: false, decimals: 4 },
];

function buildSeries(reportAI, reportTraditional, key) {
  const synthAI = reportAI.rows.filter((r) => !r.isNative);
  const synthTrad = reportTraditional.rows.filter((r) => !r.isNative);
  return synthAI.map((r, i) => ({
    t: r.t,
    ai: r.metrics[key],
    traditional: synthTrad[i]?.metrics[key],
  }));
}

function StatCard({ label, ai, traditional, higherBetter, decimals }) {
  const improvement = higherBetter
    ? ((ai - traditional) / Math.abs(traditional || 1)) * 100
    : ((traditional - ai) / Math.abs(traditional || 1)) * 100;
  const better = improvement >= 0;
  return (
    <div className="stat-card">
      <div className="stat-card__label">{label}</div>
      <div className="stat-card__row">
        <div>
          <div className="stat-card__value">{ai.toFixed(decimals)}</div>
          <div className="stat-card__tag">AI model</div>
        </div>
        <div>
          <div className="stat-card__value stat-card__value--muted">{traditional.toFixed(decimals)}</div>
          <div className="stat-card__tag">Traditional</div>
        </div>
      </div>
      <div className={`stat-card__delta ${better ? 'stat-card__delta--good' : 'stat-card__delta--bad'}`}>
        {better ? '▲' : '▼'} {Math.abs(improvement).toFixed(1)}% {better ? 'better' : 'worse'} than traditional
      </div>
    </div>
  );
}

export default function MetricsView({ reportAI, reportTraditional, resolution, satelliteLabel }) {
  return (
    <div className="metrics-view">
      <div className="metrics-view__intro">
        <h2>Validation report</h2>
        <p>
          Reconstructed frames at <strong>{resolution} min</strong> resolution for <strong>{satelliteLabel}</strong>{' '}
          compared against the fine-grained ground-truth sequence, using both the AI flow-warp model and a
          traditional linear-blend baseline.
        </p>
      </div>

      <div className="stat-card-grid">
        {METRIC_DEFS.map((m) => (
          <StatCard
            key={m.key}
            label={m.label}
            ai={reportAI.averages[m.key]}
            traditional={reportTraditional.averages[m.key]}
            higherBetter={m.higherBetter}
            decimals={m.decimals}
          />
        ))}
      </div>

      <div className="chart-grid">
        {METRIC_DEFS.map((m) => (
          <div className="chart-card" key={m.key}>
            <h3>{m.label} over sequence</h3>
            <ResponsiveContainer width="100%" height={220}>
              <LineChart data={buildSeries(reportAI, reportTraditional, m.key)}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
                <XAxis dataKey="t" tick={{ fill: '#8fa3b8', fontSize: 12 }} label={{ value: 'minutes', position: 'insideBottom', offset: -3, fill: '#8fa3b8', fontSize: 11 }} />
                <YAxis tick={{ fill: '#8fa3b8', fontSize: 12 }} width={50} />
                <Tooltip contentStyle={{ background: '#101820', border: '1px solid #223142' }} labelStyle={{ color: '#8fa3b8' }} />
                <Legend wrapperStyle={{ fontSize: 12 }} />
                <Line type="monotone" dataKey="ai" name="AI model" stroke="#22d3ee" strokeWidth={2} dot={false} />
                <Line type="monotone" dataKey="traditional" name="Traditional" stroke="#fb7185" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        ))}
      </div>
    </div>
  );
}
