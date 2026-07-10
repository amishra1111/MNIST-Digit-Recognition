# MNIST Digit Recognizer — Vercel-ready

A full 0–9 handwritten digit recognizer. Draw on a canvas, a CNN trained
on MNIST predicts the digit, entirely in the browser via TensorFlow.js —
no backend, no API calls, no cold starts.

```
mnist-app/
├── train/
│   ├── train_and_export.py   <- run this ONCE on your own machine
│   └── requirements.txt
└── web/                      <- this is the actual Vercel project
    ├── app/
    │   ├── layout.js
    │   ├── page.js            <- canvas + prediction UI
    │   ├── page.module.css
    │   └── globals.css
    ├── public/model/          <- trained model files go here
    ├── package.json
    ├── next.config.js
    └── vercel.json
```

## Why two steps (train, then deploy)

Vercel can't run a multi-minute Python/TensorFlow training job — it's a
web hosting platform, not a training environment. So the workflow is:

1. **Train once, locally** (needs internet to download MNIST) → produces
   a small model file (~200 KB).
2. **Deploy the `web/` folder to Vercel** as a normal Next.js app. The
   browser loads that model file and does inference on-device.

You only do step 1 once. After that, `web/` is a self-contained,
static-ish Next.js app you can redeploy freely.

---

## Step 1 — Train the model (run on your laptop, not here)

```bash
cd mnist-app/train
pip install -r requirements.txt
python train_and_export.py
```

This will:
- Download MNIST automatically (via `keras.datasets.mnist`, ~11 MB, cached after first run)
- Train a small CNN for 8 epochs (~1–3 minutes on CPU)
- Print test accuracy (should land around 99%)
- Export the model directly into `../web/public/model/` as:
  - `model.json`
  - `group1-shard1of1.bin`

If it printed "Model exported to ../web/public/model" at the end, you're done with this step.

## Step 2 — Run it locally to check it works

```bash
cd ../web
npm install
npm run dev
```

Open `http://localhost:3000`, draw a digit, confirm you get a sensible prediction.

## Step 3 — Push to GitHub

```bash
cd mnist-app/web
git init
git add .
git commit -m "Digit recognizer - MNIST CNN + TensorFlow.js"
git branch -M main
git remote add origin <YOUR_GITHUB_REPO_URL>
git push -u origin main
```

**Important:** make sure `public/model/model.json` and the `.bin` shard
were actually committed (they're small, git handles them fine — just
don't let an overly broad `.gitignore` rule exclude them).

```bash
git status
# should NOT show model.json / *.bin as untracked if you already git add .
```

## Step 4 — Deploy on Vercel

1. Go to https://vercel.com/new
2. Import the GitHub repo you just pushed
3. Framework preset: Vercel will auto-detect **Next.js** — leave defaults
4. Root directory: if your repo root IS `web/`, leave as `.`. If you
   pushed the whole `mnist-app/` folder (train + web together), set
   **Root Directory** to `web` in the Vercel project settings.
5. Click **Deploy**

That's it — no environment variables, no serverless functions, no
Python runtime needed on Vercel at all.

---

## Troubleshooting

**"Model failed to load" in the browser**
→ `public/model/model.json` wasn't committed/deployed. Re-check Step 1
output and confirm the files exist locally, then re-commit and re-push.

**Predictions look wrong / random**
→ Usually a stroke-width or inversion mismatch. The canvas draws black
ink on white, then `page.js` inverts it before feeding the model
(MNIST images are white digit on black background). If you changed the
drawing colors in `page.module.css` or `page.js`, keep that inversion
logic in `predict()` consistent.

**Want higher accuracy / fewer misreads of messy handwriting**
→ Bump `epochs` in `train_and_export.py` from 8 to 15–20, or add a
`layers.Conv2D(64, ...)` block before the `Flatten()` layer, then
re-run Step 1 and Step 3.
