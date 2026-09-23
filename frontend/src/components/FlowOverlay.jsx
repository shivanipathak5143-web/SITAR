export default function FlowOverlay({ vectors, size = 220 }) {
  return (
    <svg className="flow-overlay" width={size} height={size} viewBox="0 0 1 1">
      <defs>
        <marker id="arrowhead" markerWidth="6" markerHeight="6" refX="3" refY="3" orient="auto">
          <path d="M0,0 L6,3 L0,6 Z" fill="#fbbf24" />
        </marker>
      </defs>
      {vectors.map((v, i) => (
        <line
          key={i}
          x1={v.x}
          y1={v.y}
          x2={v.x + v.vx}
          y2={v.y + v.vy}
          stroke="#fbbf24"
          strokeWidth={0.006}
          markerEnd="url(#arrowhead)"
        />
      ))}
    </svg>
  );
}
