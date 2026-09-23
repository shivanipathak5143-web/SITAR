const STEPS = [
  {
    title: '1. Optical flow estimation',
    body: 'Motion vectors are estimated between consecutive TIR frames, capturing cloud-top displacement between the two nearest captured timestamps.',
  },
  {
    title: '2. Deep-learning frame interpolation',
    body: 'A learned interpolation network (e.g. RIFE / Super SloMo-style) synthesizes the intermediate frame from the estimated flow, instead of a naive cross-dissolve.',
  },
  {
    title: '3. Temporal resolution upscaling',
    body: 'Repeating the process on the same bracket produces 15-minute and 7.5-minute equivalents from a native 30-minute cadence, without extra satellite passes.',
  },
  {
    title: '4. Validation against real high-cadence data',
    body: 'Reconstructed frames are scored against genuinely higher-frequency observations (Himawari-8 / GOES-19) using SSIM, MSE, PSNR and FSIM.',
  },
];

const DATASETS = [
  { name: 'INSAT-3DS / INSAT-3DR', band: 'TIR1 (~10.8 µm)', source: 'MOSDAC', cadence: '30 min' },
  { name: 'GOES-19', band: 'ABI Channel 13 (10.3 µm)', source: 'NOAA GOES-19 AWS bucket', cadence: '10 min' },
  { name: 'Himawari-8', band: 'Band 13 (10.4 µm)', source: 'JMA / AWS open data', cadence: '10 min' },
];

export default function AboutView() {
  return (
    <div className="about-view">
      <section className="card">
        <h2>Fill in the Frames, Seamlessly</h2>
        <p>
          Geostationary TIR imagery is captured at a fixed cadence (e.g. every 30 minutes for INSAT-3DS/3DR),
          which limits near-real-time tracking of fast-evolving phenomena — convective storms, cyclones,
          wildfires, flash floods. This project fills in the gaps with an AI/ML optical-flow-based frame
          interpolation model, raising effective temporal resolution without any new satellite passes.
        </p>
      </section>

      <section className="card">
        <h3>Pipeline</h3>
        <div className="step-grid">
          {STEPS.map((s) => (
            <div className="step-card" key={s.title}>
              <h4>{s.title}</h4>
              <p>{s.body}</p>
            </div>
          ))}
        </div>
      </section>

      <section className="card">
        <h3>Datasets</h3>
        <table className="dataset-table">
          <thead>
            <tr>
              <th>Satellite</th>
              <th>Band</th>
              <th>Source</th>
              <th>Native cadence</th>
            </tr>
          </thead>
          <tbody>
            {DATASETS.map((d) => (
              <tr key={d.name}>
                <td>{d.name}</td>
                <td>{d.band}</td>
                <td>{d.source}</td>
                <td>{d.cadence}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="fineprint">Input/output for the production model are .nc (NetCDF) files; this dashboard is
          the visualization layer described in the problem statement's "Step 2".</p>
      </section>

      <section className="card card--note">
        <h3>About this build</h3>
        <p>
          This is a standalone frontend prototype. Animations and metrics on screen are generated from a
          physically-plausible synthetic cloud-motion simulator (linear drift + non-linear wobble), so the
          SSIM/PSNR/MSE/FSIM numbers are genuinely computed against a simulated ground truth — not hand-picked
          — while a real optical-flow / frame-interpolation backend is trained on the datasets above.
        </p>
        <p>Suggested API contract for wiring in the real backend:</p>
        <pre className="code-block">{`GET  /api/frames?satellite=insat3ds&start=...&end=...&resolution=15
     -> { frames: [{ t, isNative, url | ncPath }] }

POST /api/interpolate
     body: { satellite, frameA, frameB, targetTime, method }
     -> { frame: { t, url | ncPath } }

GET  /api/metrics?satellite=insat3ds&resolution=15&method=ai
     -> { rows: [{ t, mse, psnr, ssim, fsim }], averages: {...} }`}</pre>
      </section>
    </div>
  );
}
