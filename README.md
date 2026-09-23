# PDL2026 — Fill in the Frames, Seamlessly

An AI/ML-based optical-flow frame interpolation project for raising the temporal resolution of
geostationary satellite thermal-infrared (TIR) imagery — covering satellites such as
**INSAT-3DS**, **INSAT-3DR**, **GOES-19**, and **Himawari-8**.

Geostationary weather satellites capture full-disk TIR imagery at fixed intervals (e.g. every
15/30 minutes). This project explores using deep-learning frame interpolation (optical-flow based,
e.g. RIFE / Super SloMo-style networks) to synthesize the intermediate frames in between, producing
a smoother, higher-temporal-resolution image sequence for applications like cloud-motion tracking
and nowcasting — instead of relying on naive linear cross-dissolve blending between frames.

## Project Status

This repository currently contains a **standalone frontend prototype**. No backend/ML pipeline is
wired up yet. The dashboard demonstrates the intended user experience and validation reporting
using a **synthetic cloud-motion simulator** in place of real satellite data — all displayed
metrics (MSE, PSNR, SSIM, FSIM) are genuinely computed, just against generated/simulated fields
rather than live satellite imagery.

## Repository Structure

```
PDL2026/
└── frontend/          React + Vite dashboard prototype
    ├── src/
    │   ├── components/    Dashboard, Metrics, About/Pipeline views
    │   └── lib/           Simulator, interpolation methods, and metrics math
    └── README.md          Frontend-specific setup and structure notes
```

## Frontend Dashboard

The `frontend/` app is a React + Vite single-page dashboard with three views:

- **Dashboard** — side-by-side comparison of native vs. interpolated frames, a scrubbable timeline,
  and an optical-flow overlay, with controls for satellite source, target temporal resolution, and
  interpolation method (traditional linear blend vs. AI-style motion warp).
- **Validation Report** — charts comparing image-quality metrics (MSE, PSNR, SSIM, FSIM) between
  the traditional and AI interpolation methods at the selected resolution.
- **Pipeline & About** — explains the intended production pipeline (optical-flow estimation →
  deep-learning frame interpolation → temporal resolution upscaling) and sketches the planned API
  contract for wiring in a real backend.

### Tech Stack

- [React](https://react.dev/) 19
- [Vite](https://vitejs.dev/) 8
- [Recharts](https://recharts.org/) for charting
- [oxlint](https://oxc.rs/docs/guide/usage/linter.html) for linting

### Getting Started

```bash
cd frontend
npm install
npm run dev
```

This starts the Vite dev server (default: http://localhost:5173).

Other available scripts (run from `frontend/`):

```bash
npm run build     # production build
npm run preview   # preview the production build locally
npm run lint      # run oxlint
```

### Key Files

| File | Purpose |
|---|---|
| `src/lib/simulate.js` | Synthetic TIR field generator; implements traditional (linear-blend) and AI-style (motion-warp) interpolation for the demo |
| `src/lib/metrics.js` | Computes MSE, PSNR, SSIM, and an FSIM approximation on generated fields |
| `src/lib/pipeline.js` | Builds a full timeline/report for a given resolution and interpolation method |
| `src/components/DashboardView.jsx` | Animation comparison, timeline, and optical-flow overlay |
| `src/components/MetricsView.jsx` | Validation report and charts |
| `src/components/AboutView.jsx` | Pipeline explanation and planned API contract |

## Swapping in a Real Backend

The frontend currently calls local functions (`buildReport`, `fieldAt`) to generate its data. To
connect a real ML pipeline, replace these calls with fetches to the endpoints sketched in the
"Pipeline & About" tab:

- `GET /api/frames`
- `POST /api/interpolate`
- `GET /api/metrics`

Keep the same row shape the UI already expects:

```js
{ t, isNative, truth, reconstructed, metrics }
```

## Roadmap

- [ ] Implement the actual optical-flow / deep-learning interpolation model
- [ ] Wire up a backend API matching the contract above
- [ ] Replace the synthetic simulator with real satellite TIR data ingestion
- [ ] Validate metrics against real (not simulated) ground-truth frames

## License

No license specified yet.
