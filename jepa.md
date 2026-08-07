Most of what you know, you were never taught. No one labelled the world for you. You watched a cup slide off a table and shatter, watched a ball thrown in the air arc and fall, watched a thousand small futures unfold a half-second after you'd already guessed them — and somewhere in all that watching you built a working model of how things behave. A teenager learns to drive in about twenty hours. A cat that has never opened a physics textbook still flinches when the mug goes over the edge.

This is the gap that bothers Yann LeCun, and **JEPA** — the *Joint-Embedding Predictive Architecture* — is his bet on how to close it. The premise is that intelligence is mostly built by *prediction from observation*, without labels and without rewards, and that the way we've been doing self-supervised learning has been quietly fighting itself for a decade.

This guide comes in two halves. **Part I** builds the idea from the ground up, as a chain of problems where each design choice turns out to be the one you'd have been forced to invent yourself. **Part II** puts it in your hands: [`mlx-tune`](https://github.com/ARahim3/mlx-tune) brings the whole JEPA family to Apple Silicon, so you can train one from scratch or fine-tune Meta's pretrained models on your own Mac, natively on MLX, no CUDA. By the end you'll know *what JEPA is, why it works, and how to actually run it.*

---

# Part I — Understanding JEPA

## 1. The problem: learning from observation

LeCun's 2022 position paper, *A Path Towards Autonomous Machine Intelligence*, spends far less time on any particular model than on a single embarrassment: our best systems need orders of magnitude more data than an animal and still don't generalize like one. His answer is that animals build a **world model** — an internal, predictive simulator of how the world behaves — and they build it mostly by watching. That reframes the whole problem of learning into one sentence:

> How do you learn an **organized**, **actionable** representation of the world, mostly from **raw observation**?

Three words carry the weight, and most of JEPA falls out of taking each seriously:

- **organized** — distances and directions in the representation must *mean* something, so you can compute with it. (This word eventually forces LeJEPA's isotropic Gaussian.)
- **actionable** — you must be able to *predict* with it and *plan* with it. (This forces the predictor, and later action-conditioning.)
- **raw observation** — no labels. (This forces self-supervision — and self-supervision forces us to confront *collapse*.)

> Labels and rewards are thin, expensive trickles of information. The real firehose is the structure already sitting in unlabeled observation: in any image or video, the parts constrain each other. Mask part of it, predict the rest, and *the world itself is your teacher.* LeCun calls this the dark matter of intelligence, and the rest of this guide is about how to drink from that firehose without choking.

Prediction is the engine; everyone agrees on that part. GPT predicts the next token; BERT predicts masked words; masked autoencoders predict masked pixels. The disagreement, and the entire JEPA program, is about one question: **predict the missing part — but in *what space*?**

## 2. Predict — but in pixels, or in meaning?

There's a fork in the road, and JEPA is a bet on which branch to take.

**Branch A — reconstruct.** Predict the missing part *in input space*: the actual pixels, the actual waveform. Autoencoders, MAE, diffusion, and (discretely) GPT all live here.

**Branch B — predict in representation space.** Don't reconstruct the masked region's pixels; reconstruct the *embedding* a good encoder would assign to it.

Why would anyone choose B? Because of a problem B sidesteps and A cannot escape.

> Take a video of someone walking across a room and try to predict the next second of frames. Some of it is genuinely predictable: the person keeps moving, the chair stays put, the light doesn't invert. But a lot of it is irreducibly unpredictable — the exact texture of the carpet fibers, the flicker of leaves outside the window, the shifting micro-pattern of a shadow. A model forced to output **pixels** gets graded on all of it. It can't predict the carpet, so it hedges, and hedging in pixel space means blur: the grey averaged mush you've seen from every frame-prediction demo. Call it the carpet problem.

Here is the deep point. The world is only partially predictable, and the unpredictable part is mostly irrelevant. A pixel-space model is graded on the irrelevant detail anyway. A model that predicts in a *learned* representation can choose a representation that throws that detail away and keeps only what's predictable and semantic. That freedom, the encoder deciding for itself what is worth representing, is the reason JEPA exists.

<figure class="figure">
  <svg viewBox="0 0 760 300" width="760" role="img" style="display:block;width:100%;max-width:720px;margin:0 auto;height:auto;" aria-label="The fork: Branch A reconstructs pixels and is graded on every detail, producing blur; Branch B predicts the embedding of the target, discarding irrelevant detail, but the learned target can collapse.">
    <defs>
      <marker id="ab-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0 L10 5 L0 10 z" fill="var(--ink-soft)"/></marker>
      <filter id="blurpx" x="-30%" y="-30%" width="160%" height="160%"><feGaussianBlur stdDeviation="2.6"/></filter>
    </defs>
    <style>
      .hdr { font-family:var(--font-mono); font-size:13px; font-weight:500; fill:var(--ink); }
      .box { fill:var(--bg-alt); stroke:var(--rule); }
      .bt  { font-family:var(--font-mono); font-size:13px; fill:var(--ink); }
      .nl  { font-family:var(--font-mono); font-size:14px; fill:var(--ink); }
      .inode { fill:var(--bg-alt); stroke:var(--rule); }
      .arr { stroke:var(--ink-soft); stroke-width:1.5; fill:none; }
      .loss{ font-family:var(--font-mono); font-size:13px; fill:var(--ink); }
      .sub { font-family:var(--font-mono); font-size:11px; fill:var(--ink-dim); }
      .bar { fill:var(--accent); }
      .warn{ fill:#c2683f; }
    </style>
    <line x1="380" y1="28" x2="380" y2="268" stroke="var(--rule-soft)"/>
    <!-- Panel A -->
    <text class="hdr" x="40" y="44">BRANCH A — reconstruct pixels</text>
    <line x1="40" y1="54" x2="338" y2="54" stroke="var(--rule)"/>
    <circle class="inode" cx="62" cy="116" r="18"/><text class="nl" x="62" y="121" text-anchor="middle">x</text>
    <path class="arr" d="M82 116 L104 116" marker-end="url(#ab-arrow)"/>
    <rect class="box" x="108" y="98" width="96" height="38" rx="6"/><text class="bt" x="156" y="121" text-anchor="middle">decoder</text>
    <path class="arr" d="M206 116 L246 116" marker-end="url(#ab-arrow)"/>
    <g filter="url(#blurpx)">
      <rect x="252" y="94" width="82" height="28" fill="#9ec5e8"/>
      <rect x="252" y="118" width="82" height="22" fill="#a9cf86"/>
      <rect x="290" y="104" width="16" height="34" fill="#8a6a4a"/>
      <circle cx="266" cy="106" r="7" fill="#f2d27a"/>
    </g>
    <rect x="250" y="92" width="86" height="50" rx="4" fill="none" stroke="var(--rule)"/>
    <text class="sub" x="293" y="160" text-anchor="middle">predicted pixels</text>
    <text class="loss" x="40" y="206">loss vs. true pixels</text>
    <text class="sub" x="40" y="228">graded on every detail it cannot predict —</text>
    <text class="sub" x="40" y="245">carpet, leaves, ripples → <tspan class="warn">BLUR</tspan></text>
    <!-- Panel B -->
    <text class="hdr" x="404" y="44">BRANCH B — predict embeddings (JEPA)</text>
    <line x1="404" y1="54" x2="724" y2="54" stroke="var(--rule)"/>
    <circle class="inode" cx="424" cy="116" r="18"/><text class="nl" x="424" y="121" text-anchor="middle">x</text>
    <path class="arr" d="M444 116 L462 116" marker-end="url(#ab-arrow)"/>
    <rect class="box" x="466" y="98" width="84" height="38" rx="6"/><text class="bt" x="508" y="121" text-anchor="middle">encoder</text>
    <path class="arr" d="M552 116 L568 116" marker-end="url(#ab-arrow)"/>
    <rect class="box" x="572" y="98" width="92" height="38" rx="6"/><text class="bt" x="618" y="121" text-anchor="middle">predictor</text>
    <path class="arr" d="M666 116 L690 116" marker-end="url(#ab-arrow)"/>
    <g class="bar">
      <rect x="696" y="118" width="6" height="18" rx="1"/><rect x="706" y="104" width="6" height="32" rx="1"/>
      <rect x="716" y="124" width="6" height="12" rx="1"/><rect x="726" y="110" width="6" height="26" rx="1"/>
    </g>
    <text class="sub" x="700" y="160" text-anchor="middle">embedding ŝ</text>
    <text class="loss" x="404" y="206">loss vs. the encoder's embedding of y</text>
    <text class="sub" x="404" y="228">irrelevant detail discarded → semantic; but the</text>
    <text class="sub" x="404" y="245">target is learned → <tspan class="warn">CAN COLLAPSE</tspan></text>
  </svg>
  <figcaption>The fork. Branch A reconstructs pixels and is graded on every unpredictable detail, so it hedges into blur. Branch B predicts the <em>embedding</em> of the target, free to discard that detail — the win — but the target is produced by the very encoder being trained, which is what opens the door to collapse.</figcaption>
</figure>

> Say the difference between the two views is just "shift the crop 50 pixels right." A pixel predictor *must* reproduce the shifted pixels. A representation predictor can treat that shift as *invariant* (ignore it, if it's nuisance) or *equivariant* (predict it cleanly, if it matters). You choose what the representation is sensitive to; pixels never give you that choice.

But Branch B comes with a bill, and the rest of Part I is about paying it:

> There's a catch. In Branch A the target (the real pixels) is fixed by the data, so you literally cannot cheat. In Branch B the target embedding is produced by *the very encoder you are training.* Nothing stops that encoder from deciding, *"I'll map every input to the same constant vector; then prediction is trivial and my loss is zero."* That is **collapse**, and it's the price of predicting in latent space.

## 3. The villain: collapse

Let's name the enemy precisely, because every architectural oddity in the JEPA lineage exists to defeat one of its two modes.

- **Complete collapse** — the encoder maps *every* input to the *same* vector. Prediction is trivially perfect and the representation carries no information whatsoever.
- **Dimensional collapse** — subtler and far more common. The encoder uses only a thin subspace of the available dimensions; embeddings live on a line or plane inside a 768-d space. Most directions are wasted, and downstream models starve for features.

Why does latent prediction *want* to collapse? Write the simplest possible JEPA loss — two views $x, y$, one encoder $f_\theta$, pull their embeddings together:

$$\mathcal{L} = \lVert f_\theta(x) - f_\theta(y) \rVert^2$$

What is the global minimum over $\theta$? Set $f_\theta \equiv c$ for any constant $c$. Then $\mathcal{L}=0$ identically, for *every* pair. So the collapsed solution is no weird corner case; it is the literal optimum of the naive objective, and gradient descent, doing its job perfectly, will walk straight into it. So any working method *must* add something that makes the constant solution stop being optimal. That "something" is the anti-collapse mechanism, and the whole pre-2025 literature is a catalogue of them.

> One more lens, because it pays off later. Think of training as carving an *energy landscape* over (x, y) pairs, where low energy means "these go together." You push energy **down** on real data, but you also have to keep it **high everywhere else**, or the landscape goes flat and useless (collapse again). There are two ways to do that. **Contrastive**: push energy *up* at specific "negative" pairs, though in high dimensions you'd need exponentially many to cover every direction. **Regularized**: constrain the embeddings so only a *small volume* of space can ever have low energy. JEPA bets on the regularized branch, which scales where chasing negatives doesn't.

## 4. The family of fixes — and what makes it a *JEPA*

Here's the lineage, organized by *how each method kills the constant solution*. You don't need the details — you need to see that they're all the same shape: **a Siamese "pull matched views together" network, plus one bespoke anti-collapse trick.**

| Family | Examples | The trick that kills collapse | The cost |
| --- | --- | --- | --- |
| Contrastive | SimCLR, MoCo | **Negatives** — push other images apart | needs *many* negatives (the exponential) |
| Distillation / asymmetry | BYOL, SimSiam | **Stop-gradient + EMA + a predictor head** — the target is a lagging, detached copy of yourself | fragile, schedule-dependent |
| Regularized / info-max | VICReg, Barlow Twins | **Variance + covariance penalties** — keep every dimension alive and decorrelated | tuned coefficients |
| Clustering / self-distillation | DINO, DINOv2/v3 | **Centering + sharpening + EMA teacher** | more schedules |

One row deserves a closer look, because it is the one that comes back at the end. **VICReg** (Bardes, Ponce & LeCun, 2022) goes at collapse head-on, with three terms you can read straight off the taxonomy from the last chapter.

Start from the naive objective: pull the two views of an image together. That term is the **invariance** term, and by itself it is precisely the loss that collapses — map everything to one point and you are done, loss zero. So VICReg adds a **variance** term. Take each coordinate of the embedding, measure its standard deviation across the batch, and require it to stay above a floor. A single point has zero standard deviation in every coordinate, so the floor makes complete collapse unreachable.

That rules out the point but not the line. Nothing yet stops the encoder from spreading the batch along one tilted direction and letting every coordinate pick up its variance from that single direction. Every coordinate looks busy, the floor is satisfied, and the representation is still one-dimensional — dimensional collapse in disguise. Hence the third term, **covariance**: form the covariance matrix of the embeddings over the batch and drive its off-diagonal entries to zero. Two correlated coordinates are two coordinates storing the same fact twice, so zeroing the off-diagonals forces them apart, and the line opens out into a round cloud.

<figure class="figure fig-live" id="fig-vicreg">
  <div class="fig-stage"><canvas role="img" aria-label="Three panels. Left: invariance alone collapses every embedding to a single clumped point, marked with a cross. Middle: adding the variance term spreads the points but only along one tilted line, still marked with a cross. Right: adding the covariance term opens the line into a round isotropic cloud, marked with a check."></canvas></div>
  <div class="fig-controls">
    <button class="fig-btn" data-act="reseed">reseed</button>
    <button class="fig-btn" data-act="pause">pause</button>
  </div>
  <figcaption>VICReg in one picture. Invariance on its own pulls each sample's two views together and crushes the whole embedding to a point. The variance floor rules that out, but a single tilted line still satisfies it, every coordinate borrowing its spread from one direction. Zeroing the cross-correlations opens that line into a round cloud — the same isotropic shape Chapter 7 arrives at from a completely different direction.</figcaption>
  <script>
  (function () {
    var fig = document.getElementById('fig-vicreg');
    if (!fig || !window.FigKit) return;
    var G = FigKit.gauss, TAU = 6.283185307;
    var P1 = [], P2 = [], P3 = [], links = [];
    function rnd(a, b) { return a + Math.random() * (b - a); }
    function build() {
      P1 = []; P2 = []; P3 = []; links = [];
      var i, k, a, b, t, n;
      for (i = 0; i < 26; i++) P1.push({ x: G() * 0.13, y: G() * 0.13, ph: rnd(0, TAU), fr: rnd(0.5, 1.1) });
      for (k = 0; k < 7; k++) {
        a = (Math.random() * 26) | 0; b = (Math.random() * 26) | 0;
        if (a !== b) links.push([a, b]);
      }
      var A = -0.5, ca = Math.cos(A), sa = Math.sin(A);
      for (i = 0; i < 26; i++) {
        t = rnd(-1.9, 1.9); n = G() * 0.05;
        P2.push({ x: t * ca - n * sa, y: t * sa + n * ca, ph: rnd(0, TAU), fr: rnd(0.5, 1.1), t: t });
      }
      for (i = 0; i < 30; i++) P3.push({ x: G() * 0.85, y: G() * 0.85, ph: rnd(0, TAU), fr: rnd(0.4, 0.9) });
    }
    build();
    var st = FigKit.mount(fig, { ratio: 0.52, minH: 300, maxH: 420, draw: draw });
    function rr(ctx, x, y, w, h, r) {
      ctx.beginPath();
      ctx.moveTo(x + r, y);
      ctx.arcTo(x + w, y, x + w, y + h, r);
      ctx.arcTo(x + w, y + h, x, y + h, r);
      ctx.arcTo(x, y + h, x, y, r);
      ctx.arcTo(x, y, x + w, y, r);
      ctx.closePath();
    }
    function draw(ctx, s) {
      var p = s.pal, tt = s.t, i;
      var R = Math.min(s.W, s.H) * 0.235, cy = s.H * 0.46, sc = R / 2.7;
      var cxs = [s.W * 0.16, s.W * 0.5, s.W * 0.84];
      function chip(x, y, letter, color, active) {
        rr(ctx, x, y, 19, 19, 5);
        ctx.fillStyle = active ? color : p.ruleSoft;
        ctx.fill();
        if (!active) { ctx.strokeStyle = p.rule; ctx.lineWidth = 1; ctx.stroke(); }
        ctx.fillStyle = active ? p.bg : p.inkDim;
        ctx.font = 'italic 500 11px ' + s.mono;
        ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        ctx.fillText(letter, x + 9.5, y + 10);
        ctx.textBaseline = 'alphabetic';
      }
      function chips(cx, y, on) {
        var cols = [p.warm, p.amber, p.blue], lets = ['i', 'v', 'c'];
        var x0 = cx - (3 * 19 + 2 * 6) / 2;
        for (var q = 0; q < 3; q++) chip(x0 + q * 25, y, lets[q], cols[q], !!on[q]);
      }
      function label(cx, y, txt) {
        ctx.fillStyle = p.inkDim; ctx.font = '11.5px ' + s.mono;
        ctx.textAlign = 'center'; ctx.fillText(txt, cx, y);
      }
      function mark(cx, ok) {
        var x = cx + R * 0.68, y = cy - R * 0.68;
        ctx.lineWidth = 2; ctx.lineCap = 'round';
        ctx.strokeStyle = ok ? p.accent : p.bad;
        ctx.beginPath();
        if (ok) { ctx.moveTo(x - 4, y); ctx.lineTo(x - 1, y + 4); ctx.lineTo(x + 5, y - 4); }
        else { ctx.moveTo(x - 4, y - 4); ctx.lineTo(x + 4, y + 4); ctx.moveTo(x + 4, y - 4); ctx.lineTo(x - 4, y + 4); }
        ctx.stroke(); ctx.lineCap = 'butt';
      }
      function arrow(x0, x1, txt, col) {
        ctx.strokeStyle = p.rule; ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.moveTo(x0, cy); ctx.lineTo(x1 - 7, cy); ctx.stroke();
        FigKit.arrowHead(ctx, x1, cy, 0, p.rule, 8);
        ctx.fillStyle = col; ctx.font = '500 10.5px ' + s.mono;
        ctx.textAlign = 'center'; ctx.fillText(txt, (x0 + x1) / 2, cy - 10);
      }
      function dot(x, y, col) {
        ctx.beginPath(); ctx.arc(x, y, 4, 0, TAU); ctx.fillStyle = col; ctx.fill();
      }
      arrow(cxs[0] + R + 6, cxs[1] - R - 6, '+ variance', p.amber);
      arrow(cxs[1] + R + 6, cxs[2] - R - 6, '+ covariance', p.blue);
      /* 1 — invariance alone: everything pulled into one clump */
      var cx = cxs[0];
      FigKit.dashedCircle(ctx, cx, cy, R, p.rule);
      var breathe = 0.9 + 0.12 * Math.sin(tt * 1.1);
      ctx.strokeStyle = p.warm; ctx.lineWidth = 1;
      FigKit.alpha(ctx, 0.3, function () {
        for (var k = 0; k < links.length; k++) {
          var pa = P1[links[k][0]], pb = P1[links[k][1]];
          ctx.beginPath();
          ctx.moveTo(cx + pa.x * sc * breathe, cy - pa.y * sc * breathe);
          ctx.lineTo(cx + pb.x * sc * breathe, cy - pb.y * sc * breathe);
          ctx.stroke();
        }
      });
      for (i = 0; i < P1.length; i++) {
        dot(cx + P1[i].x * sc * breathe, cy - P1[i].y * sc * breathe, p.inkDim);
      }
      chips(cx, cy - R - 30, [1, 0, 0]);
      mark(cx, false);
      label(cx, cy + R + 26, 'collapses to a point');
      /* 2 — variance added: spread, but along a single direction */
      cx = cxs[1];
      FigKit.dashedCircle(ctx, cx, cy, R, p.rule);
      var A = -0.5, ca = Math.cos(A), sa = Math.sin(A);
      ctx.strokeStyle = p.amber; ctx.lineWidth = 1;
      FigKit.alpha(ctx, 0.35, function () {
        ctx.beginPath();
        ctx.moveTo(cx - 2.4 * ca * sc, cy + 2.4 * sa * sc);
        ctx.lineTo(cx + 2.4 * ca * sc, cy - 2.4 * sa * sc);
        ctx.stroke();
      });
      for (i = 0; i < P2.length; i++) {
        var q2 = P2[i], slide = 0.12 * Math.sin(tt * 0.6 + q2.ph);
        dot(cx + (q2.t + slide) * ca * sc + 1.5 * Math.sin(tt * q2.fr + q2.ph),
            cy - (q2.t + slide) * sa * sc + 1.0 * Math.cos(tt * q2.fr + q2.ph), p.inkDim);
      }
      chips(cx, cy - R - 30, [1, 1, 0]);
      mark(cx, false);
      label(cx, cy + R + 26, 'spreads, onto one line');
      /* 3 — covariance added: the line opens into a round cloud */
      cx = cxs[2];
      FigKit.dashedCircle(ctx, cx, cy, R, p.rule);
      var rot = tt * 0.06, cr = Math.cos(rot), sr = Math.sin(rot);
      var er = 1.85 * sc * (1 + 0.025 * Math.sin(tt * 0.9));
      ctx.beginPath(); ctx.arc(cx, cy, er, 0, TAU);
      ctx.fillStyle = p.accent;
      FigKit.alpha(ctx, 0.1, function () { ctx.fill(); });
      ctx.strokeStyle = p.accent; ctx.lineWidth = 1.6;
      FigKit.alpha(ctx, 0.55, function () { ctx.stroke(); });
      for (i = 0; i < P3.length; i++) {
        var q3 = P3[i];
        dot(cx + (q3.x * cr - q3.y * sr) * sc + 1.2 * Math.sin(tt * q3.fr + q3.ph),
            cy - (q3.x * sr + q3.y * cr) * sc + 1.2 * Math.cos(tt * q3.fr + q3.ph), p.accent);
      }
      chips(cx, cy - R - 30, [1, 1, 1]);
      mark(cx, true);
      label(cx, cy + R + 26, 'isotropic, every axis alive');
    }
    fig.querySelector('[data-act="reseed"]').addEventListener('click', function () {
      build(); st.repaint();
    });
    fig.querySelector('[data-act="pause"]').addEventListener('click', function () {
      st.paused = !st.paused;
      this.textContent = st.paused ? 'play' : 'pause';
      st.repaint();
    });
  })();
  </script>
</figure>

Three terms, three collapses denied, and one loose thread. Nothing in that construction ever says *round* is the right answer. The variance term floors the diagonal of the covariance matrix and the covariance term zeros everything off it, so between them they push $\Sigma$ toward a multiple of the identity, which is the covariance of an isotropic Gaussian. VICReg gets there by constraining the first two moments and stopping. Keep that in your pocket until Chapter 7.

Every one of these works. And every one pays for "no collapse" with *machinery* — negatives, or stop-gradient plus EMA plus a predictor, or variance/covariance terms with tuned weights, or centering plus temperature schedules. None of them can tell you *why* their particular machine is the right one, and (file this complaint) none can tell you from the training loss alone whether the learned features are any good. This is the swamp LeJEPA eventually drains.

Now, the one component that turns this family into a **JEPA**: a **predictor**.

> A plain joint-embedding architecture says: *make $f_\theta(x)$ and $f_\theta(y)$ equal.* A JEPA says: *make a predictor $g_\phi$ map one to the other* — $g_\phi\big(f_\theta(x)\big) \approx f_\theta(y)$.

Why is "predict" better than "make equal"? Because *equal* deletes everything that differs between the views, forcing pure invariance: a sledgehammer. *Predict* lets the representation stay **sensitive** to the differences while staying trainable. "Make equal" says *whatever changed, ignore it.* "Predict" says *whatever changed, model it* — which is exactly what a world model needs. It has to represent *that* things change and *how*, not pretend they didn't.

There's a name for this distinction, worth stating outright. A pure Siamese network (SimCLR, Barlow Twins) trains the encoder to be **invariant** to the transformation between views: shift the crop or jitter the colour and the embedding shouldn't budge. A JEPA *can* instead make it **equivariant**. The embedding is allowed to move, and the predictor, told *which* transformation happened through position/mask tokens (and later an action variable), reproduces that move in latent space. The key word is *can*, because this is a dial rather than a default. What sets the dial is the **predictor's capacity**. A weak or identity predictor forces the encoder to throw away whatever it can't predict, collapsing back to invariance; a capable predictor lets the encoder *keep* the structure and offloads the transformation to the predictor. Invariance for nuisances, equivariance for what matters: that controllable dial is what a world model needs, and it's what "make equal" can never give you.

## 5. I-JEPA: the first clean instantiation

*Assran et al., CVPR 2023.* I-JEPA ("Image-JEPA") is the first model that makes the abstract diagram work on real images, and it does it with **no hand-crafted data augmentation**. It's built on a Vision Transformer, so an image is a grid of patches (tokens), and it has three networks:

1. **Context encoder** $f_\theta$ — encodes the *visible* patches. This is the network you keep.
2. **Target encoder** $f_{\bar\theta}$ — architecturally identical, but its weights are an **EMA** (exponential moving average) of the context encoder's. It is **not** trained by gradients. It produces the targets.
3. **Predictor** $g_\phi$ — a narrow ViT that, given the context plus position tokens saying *where* the targets are, predicts the target embeddings.

<figure class="figure">
  <svg viewBox="0 0 780 384" width="780" role="img" style="display:block;width:100%;max-width:720px;margin:0 auto;height:auto;" aria-label="I-JEPA: a trained context encoder and predictor map visible context patches plus position tokens to predicted target embeddings; a frozen EMA target encoder produces the true target embeddings; the loss is their squared distance in latent space; the target encoder is an EMA of the context encoder.">
    <defs>
      <marker id="ij-arrow" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0 L10 5 L0 10 z" fill="var(--ink-soft)"/></marker>
      <marker id="ij-ema" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0 L10 5 L0 10 z" fill="var(--ink-dim)"/></marker>
    </defs>
    <style>
      .tbox { fill:var(--bg-alt); stroke:var(--accent); stroke-width:2; }
      .fbox { fill:var(--bg-alt); stroke:#d8932a; stroke-width:2; }
      .nbox { fill:var(--bg-alt); stroke:var(--rule); }
      .arr  { stroke:var(--ink-soft); stroke-width:1.5; fill:none; }
      .ema  { stroke:var(--ink-dim); stroke-width:1.5; fill:none; stroke-dasharray:5 4; }
      .bt   { font-family:var(--font-mono); font-size:12.5px; fill:var(--ink); }
      .sub  { font-family:var(--font-mono); font-size:10.5px; fill:var(--ink-dim); }
      .el   { font-family:var(--font-mono); font-size:11px; fill:var(--ink-soft); }
      .leg  { font-family:var(--font-mono); font-size:11px; fill:var(--ink-soft); }
    </style>
    <!-- trained top path -->
    <rect class="nbox" x="36" y="54" width="156" height="46" rx="6"/>
    <text class="bt" x="114" y="76" text-anchor="middle">context patches</text>
    <text class="sub" x="114" y="91" text-anchor="middle">visible</text>
    <rect class="tbox" x="228" y="54" width="158" height="46" rx="6"/>
    <text class="bt" x="307" y="76" text-anchor="middle">context encoder fθ</text>
    <text class="sub" x="307" y="91" text-anchor="middle">TRAINED</text>
    <rect class="tbox" x="470" y="54" width="120" height="46" rx="6"/>
    <text class="bt" x="530" y="76" text-anchor="middle">predictor gφ</text>
    <text class="sub" x="530" y="91" text-anchor="middle">TRAINED</text>
    <rect class="nbox" x="618" y="54" width="124" height="46" rx="6"/>
    <text class="bt" x="680" y="82" text-anchor="middle">predicted ŝ</text>
    <!-- position tokens -->
    <rect class="nbox" x="430" y="150" width="184" height="44" rx="6"/>
    <text class="bt" x="522" y="171" text-anchor="middle">position / mask tokens</text>
    <text class="sub" x="522" y="186" text-anchor="middle">where the targets are</text>
    <!-- frozen bottom path -->
    <rect class="nbox" x="36" y="266" width="156" height="46" rx="6"/>
    <text class="bt" x="114" y="288" text-anchor="middle">target blocks</text>
    <text class="sub" x="114" y="303" text-anchor="middle">large; removed</text>
    <rect class="fbox" x="228" y="266" width="200" height="46" rx="6"/>
    <text class="bt" x="328" y="288" text-anchor="middle">target encoder f_θ̄</text>
    <text class="sub" x="328" y="303" text-anchor="middle">EMA copy · NO gradient</text>
    <rect class="nbox" x="618" y="266" width="124" height="46" rx="6"/>
    <text class="bt" x="680" y="294" text-anchor="middle">targets s</text>
    <!-- loss -->
    <rect class="nbox" x="612" y="150" width="136" height="46" rx="6"/>
    <text class="bt" x="680" y="171" text-anchor="middle">loss = ‖ŝ − s‖²</text>
    <text class="sub" x="680" y="186" text-anchor="middle">in latent space</text>
    <!-- arrows -->
    <path class="arr" d="M192 77 L228 77" marker-end="url(#ij-arrow)"/>
    <path class="arr" d="M386 77 L470 77" marker-end="url(#ij-arrow)"/>
    <text class="el" x="428" y="69" text-anchor="middle">s_ctx</text>
    <path class="arr" d="M590 77 L618 77" marker-end="url(#ij-arrow)"/>
    <path class="arr" d="M522 150 L530 100" marker-end="url(#ij-arrow)"/>
    <path class="arr" d="M192 289 L228 289" marker-end="url(#ij-arrow)"/>
    <path class="arr" d="M428 289 L618 289" marker-end="url(#ij-arrow)"/>
    <path class="arr" d="M680 100 L680 150" marker-end="url(#ij-arrow)"/>
    <path class="arr" d="M680 266 L680 196" marker-end="url(#ij-arrow)"/>
    <!-- EMA update -->
    <path class="ema" d="M307 100 L307 266" marker-end="url(#ij-ema)"/>
    <text class="el" x="298" y="176" text-anchor="end">θ̄ ← m·θ̄ + (1−m)·θ</text>
    <text class="sub" x="298" y="191" text-anchor="end">EMA · stop-grad</text>
    <!-- legend -->
    <rect class="tbox" x="232" y="344" width="14" height="14" rx="3"/>
    <text class="leg" x="254" y="355">trained (gradients flow)</text>
    <rect class="fbox" x="470" y="344" width="14" height="14" rx="3"/>
    <text class="leg" x="492" y="355">frozen — EMA target, no gradient</text>
  </svg>
  <figcaption>I-JEPA. The trained context encoder and predictor (green) turn visible patches — plus position tokens marking where the targets are — into predicted target embeddings ŝ. A frozen EMA copy of the encoder (amber, no gradient) produces the true targets s. The loss is their distance <em>in latent space</em>, and the target encoder is nudged toward the context encoder by a slow EMA. That lagging, detached target is what keeps the whole thing from collapsing.</figcaption>
</figure>

Two things make I-JEPA tick. First, the **masking strategy**, which the authors are emphatic is the core design choice. You sample one large, *scattered* context block and several *large* target blocks, and you remove any context patch that overlaps a target.

> Why does that help? If targets were single tiny patches, the model could predict them by local texture continuation — "next to green grass is probably more green grass" — which teaches it nothing. Make the targets *large* and the context *scattered and incomplete*, and the only way to predict a target block's representation is to understand what object is there and how it continues: the dog's head from its body, the bird's leg from its torso. The masking is a curriculum engineered to **ban cheating**, so the gradient has no choice but to build real semantic understanding.

Second, the **anti-collapse mechanism** is the BYOL trick from the table above: the target comes from the EMA encoder, which *lags* the context encoder and receives *no gradient*. The encoder can't instantly "agree with itself" to collapse, because the target is a slow, detached copy of its own past. It works — but notice it leans on the EMA momentum schedule. Push that wrong and you can still collapse. (Hold that thought.)

> If you keep one line about I-JEPA, keep this: *mask big chunks, predict their embeddings (not pixels) from a scattered context, and keep the target encoder as a lagging EMA copy so the whole thing doesn't collapse.*

## 6. V-JEPA 2: when JEPA becomes a world model

Images are static. A world model has to understand **dynamics** — how things change, what actions do — and that needs video. **V-JEPA** lifts the exact I-JEPA skeleton to spatiotemporal tokens: mask tubes of patches across space *and* time, predict their representations from the visible context. The Branch-B payoff is even bigger here, because video is where pixel prediction drowns in the carpet problem most badly.

**V-JEPA 2** (Meta, June 2025) is the headline result, and it comes in two stages:

- **Stage 1 — learn how the world moves, by watching.** Pretrain on **over a million hours** of internet video with the masked-latent-prediction objective. No actions, no rewards — pure observation. The result is state-of-the-art motion understanding and action *anticipation*, from a **frozen** encoder that was never trained on those tasks.
- **Stage 2 — connect actions to the world.** Freeze that encoder and train a small **action-conditioned predictor** on a *tiny* slice of robot video (~62 hours): given the current latent state and an action, predict the next latent state. That is a learned simulator of the world, running in representation space.

> Look at the data ratio, because that's the whole story. A million hours of free, passive, unlabeled internet video to learn general physics; sixty-two hours of cheap robot footage to learn "how my body changes that world." No task labels, no rewards. The bet is that most of physics can be learned by watching, and only a thin layer of embodiment needs embodied data — the animal-learning story from Chapter 1, finally instantiated.

<figure class="figure">
  <svg viewBox="0 0 760 320" width="760" role="img" style="display:block;width:100%;max-width:700px;margin:0 auto;height:auto;" aria-label="Action-conditioned world model: an observation x and outcome y are encoded; a predictor maps the encoded state plus an action to a predicted outcome embedding; an energy D scores it against the real outcome embedding — all in representation space.">
    <defs>
      <marker id="wm-arrow" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0 L10 5 L0 10 z" fill="var(--ink-soft)"/></marker>
    </defs>
    <style>
      .nbox { fill: var(--bg-alt); stroke: var(--rule); }
      .pbox { fill: var(--bg-alt); stroke: var(--accent); stroke-width: 2; }
      .dbox { fill: var(--bg-alt); stroke: #d9534f; stroke-width: 2; }
      .inode{ fill: var(--bg-alt); stroke: var(--rule); }
      .arr  { stroke: var(--ink-soft); stroke-width: 1.5; fill: none; }
      .bt   { font-family: var(--font-mono); font-size: 14px; fill: var(--ink); }
      .nl   { font-family: var(--font-mono); font-size: 15px; fill: var(--ink); }
      .sub  { font-family: var(--font-mono); font-size: 11px; fill: var(--ink-dim); }
      .el   { font-family: var(--font-mono); font-size: 12px; fill: var(--ink-soft); }
    </style>
    <!-- nodes -->
    <circle class="inode" cx="90" cy="248" r="24"/><text class="nl" x="90" y="253" text-anchor="middle">x</text>
    <text class="sub" x="90" y="292" text-anchor="middle">observation</text>
    <circle class="inode" cx="364" cy="248" r="22"/><text class="nl" x="364" y="253" text-anchor="middle">a</text>
    <text class="sub" x="364" y="294" text-anchor="middle">action / intervention</text>
    <circle class="inode" cx="680" cy="248" r="24"/><text class="nl" x="680" y="253" text-anchor="middle">y</text>
    <text class="sub" x="680" y="292" text-anchor="middle">outcome</text>
    <!-- boxes -->
    <rect class="nbox" x="140" y="150" width="96" height="42" rx="6"/><text class="bt" x="188" y="176" text-anchor="middle">Enc(x)</text>
    <rect class="nbox" x="536" y="150" width="96" height="42" rx="6"/><text class="bt" x="584" y="176" text-anchor="middle">Enc(y)</text>
    <rect class="pbox" x="300" y="46" width="128" height="48" rx="6"/><text class="bt" x="364" y="75" text-anchor="middle">Pred(s_x)</text>
    <rect class="dbox" x="470" y="46" width="180" height="48" rx="6"/><text class="bt" x="560" y="75" text-anchor="middle">D(s_y, ŝ_y)</text>
    <text class="sub" x="560" y="34" text-anchor="middle">energy = prediction error</text>
    <!-- arrows -->
    <path class="arr" d="M104 233 L152 194" marker-end="url(#wm-arrow)"/>
    <path class="arr" d="M214 150 L320 96" marker-end="url(#wm-arrow)"/><text class="el" x="248" y="124">s_x</text>
    <path class="arr" d="M364 226 L364 96" marker-end="url(#wm-arrow)"/>
    <path class="arr" d="M428 70 L470 70" marker-end="url(#wm-arrow)"/><text class="el" x="449" y="60" text-anchor="middle">ŝ_y</text>
    <path class="arr" d="M666 233 L618 194" marker-end="url(#wm-arrow)"/>
    <path class="arr" d="M584 150 L562 96" marker-end="url(#wm-arrow)"/><text class="el" x="600" y="124">s_y</text>
  </svg>
  <figcaption>The action-conditioned world model (after LeCun). Both the observation <em>x</em> and the outcome <em>y</em> are encoded; the predictor maps the current latent <em>plus an action a</em> to a predicted outcome embedding ŝ<sub>y</sub>, and the energy <em>D</em> scores it against the real outcome's embedding s<sub>y</sub> — never in pixels, always in representation space.</figcaption>
</figure>

And it pays off: the robot **plans in latent space**. Encode a goal image, imagine candidate action sequences by rolling the predictor forward, and score each by a *planning* energy, namely the distance between the imagined future latent and the goal latent. Then search for the action sequence that minimizes it, re-planning after each step. The result is *zero-shot* reaching, grasping, and pick-and-place with new objects in new labs, with no task-specific training and no reward function.

> One scope check before Part II. Of V-JEPA 2's two stages, `mlx-tune` ports Stage 1 whole: the **encoder** *and* the masked-latent **predictor** — so clip representations, fine-tuning, and anticipation-with-surprise-scoring all run on your Mac. What it doesn't port is Stage 2: the *action-conditioned* predictor (V-JEPA 2-AC) and the robot-planning loop from the headlines. But "a world model you can plan with" isn't off the table — it arrives by a different door. Chapter 13's **LeWM** lets you *train* a small latent world model from scratch and plan with it locally. So the honest line is *load Meta's robot planner* (not yet) versus *train your own and plan* (yes, today). Keep it in mind as the second half gets concrete.

## 7. LeJEPA: deleting the heuristics

Step back and feel the discomfort that anyone training these models knows in their hands:

1. **It's held together with tape.** Stop-gradients, EMA encoders with tuned momentum schedules, asymmetric predictor heads, centering and sharpening. Get one wrong and it collapses. The recipes are brittle folklore, not theory.
2. **No theory says why these tricks are right**, or what the ideal representation even *is*.
3. **You can't tell if it worked without labels.** A JEPA's training loss is *not* a reliable indicator of downstream quality, so you're forced into supervised probing to do model selection — which partly defeats the point.

The question that breaks the logjam: *among all distributions the embeddings could follow, is there one that is **provably optimal** — and if we just enforced it directly, would all the machinery become unnecessary?* **LeJEPA** (Balestriero & LeCun, Nov 2025) answers yes, and the answer is the **isotropic Gaussian**.

An isotropic Gaussian $\mathcal{N}(0, \sigma^2 I)$ is a perfectly round cloud: equal variance in every direction, zero correlation between directions. A circle in 2D, a sphere in 3D, the same symmetry in 768-d. It is the round ball from the VICReg picture, and it is the anti-collapse distribution by construction, because the two failure modes from Chapter 3 are precisely the shapes it cannot be. Complete collapse means zero variance everywhere. Dimensional collapse means zero variance along some directions. Demand full, equal variance along *all* of them and you have ruled out both at once, without naming either.

But appealing is not the same as optimal, and LeJEPA *proves* optimal. The argument: you'll pretrain the encoder, and later someone attaches a probe for a task you don't know in advance. Which embedding distribution minimizes the *worst-case* downstream error over all the tasks you might face?

> The intuition is one sentence: you don't know what question will be asked of your features later, so make them equally good at answering in **every** direction and hide **no** structure. "Equally good in every direction" pins the covariance to $\Sigma \propto I$, since any weak direction is one an adversarial task can exploit. "Hide no structure" forces the *full* Gaussian: among all distributions with a given covariance, the Gaussian has **maximum entropy**, the least committed, with no privileged directions or hidden clusters a task could fall outside of. Equally good everywhere with nothing hidden *is* the definition of an isotropic Gaussian.

That's the "Latent-Euclidean" in the name: in such a space, ordinary Euclidean geometry (distances, directions) is meaningful and uniform everywhere, which is exactly what a downstream probe wants.

### SIGReg: how to actually enforce it

Knowing the target is half the battle. Matching a cloud of embeddings to a Gaussian in 768 dimensions directly is hopeless — the curse of dimensionality. LeJEPA's second contribution, **SIGReg** (*Sketched Isotropic Gaussian Regularization*), makes it linear-time by welding two classical ideas:

- **Cramér–Wold — kill dimensionality with projections.** A distribution in $\mathbb{R}^d$ is *completely determined* by all its 1-D projections. So instead of comparing two $d$-dimensional clouds, project onto many random 1-D directions and check each *shadow*. If every shadow of your embeddings looks like a standard 1-D Gaussian, the cloud *is* an isotropic Gaussian.
- **Epps–Pulley — a good 1-D normality test.** On each shadow you need a smooth, stable "how Gaussian is this?" loss. Raw moments (skew, kurtosis) explode on outliers; CDF tests give bad gradients. The winner is the **characteristic function** — it uniquely identifies a distribution (no loopholes) and is built from *bounded* $\cos$/$\sin$ terms (stable gradients, robust to outliers). SIGReg measures the gap between your shadow's empirical characteristic function and the standard normal's.

> Here's the picture. To check whether a fruit is a perfect sphere you don't measure its volume; you spin it and confirm the *shadow* is a circle from every angle. SIGReg does the same to the embedding cloud: cast its shadow along many random directions and check that each one looks like the same bell curve. Many circular shadows mean a round, isotropic-Gaussian cloud. It's linear in dimension and sample count, shards trivially across GPUs, and fits in about 50 lines.

<figure class="figure fig-live" id="fig-sigreg">
  <div class="fig-stage"><canvas role="img" aria-label="Interactive figure: a projection direction sweeps around an embedding cloud and the resulting one-dimensional shadow is compared against a standard normal bell curve, with a live Epps-Pulley gap readout. Switching the cloud to stretched or two-clusters makes the shadow depend on direction and the gap grow."></canvas></div>
  <div class="fig-controls">
    <span class="fig-seg">
      <button class="fig-btn" data-cloud="iso" aria-pressed="true">isotropic</button>
      <button class="fig-btn" data-cloud="stretch" aria-pressed="false">stretched</button>
      <button class="fig-btn" data-cloud="bimodal" aria-pressed="false">two clusters</button>
    </span>
    <button class="fig-btn" data-act="fix" disabled>apply SIGReg</button>
    <button class="fig-btn" data-act="pause">pause</button>
  </div>
  <figcaption>SIGReg in one picture (Cramér–Wold plus Epps–Pulley). The direction sweeps; each point drops onto it, and those feet are the <em>shadow</em>, drawn on the right against the standard normal it is being pushed toward. While the cloud is isotropic the shadow stays the same bell curve at every angle and the Epps–Pulley gap <em>T</em> stays near zero. Stretch the cloud, or split it in two, and the shadow starts to depend on which direction you look from — which is exactly the signal <em>T</em> reports, and exactly what the regularizer removes.</figcaption>
  <script>
  (function () {
    var fig = document.getElementById('fig-sigreg');
    if (!fig || !window.FigKit) return;
    var G = FigKit.gauss, N = 42;
    var pts = [], target = [], cloud = 'iso', theta = 0, lastT = 0;
    var morph = { active: false, k: 0, from: null };
    function makeIso() {
      var a = [], i;
      for (i = 0; i < N; i++) a.push({ x: G(), y: G() });
      return a;
    }
    function seed(type) {
      cloud = type;
      var i, c, s, x, y, sgn;
      if (type === 'iso') { pts = makeIso(); }
      else if (type === 'stretch') {
        pts = []; c = Math.cos(0.5); s = Math.sin(0.5);
        for (i = 0; i < N; i++) {
          x = G() * 1.95; y = G() * 0.45;
          pts.push({ x: x * c - y * s, y: x * s + y * c });
        }
      } else {
        pts = [];
        for (i = 0; i < N; i++) {
          sgn = i < N / 2 ? -1 : 1;
          pts.push({ x: sgn * 1.7 + G() * 0.42, y: G() * 0.72 });
        }
      }
      target = makeIso();
      morph.active = false;
    }
    seed('iso');
    /* Epps-Pulley: integrate the weighted gap between the shadow's empirical
       characteristic function and the standard normal's. */
    function eppsPulley(z) {
      var n = z.length, T = 0, dt = 0.25, t, j, re, im, phi0, w;
      for (t = -5; t <= 5; t += dt) {
        re = 0; im = 0;
        for (j = 0; j < n; j++) { re += Math.cos(t * z[j]); im += Math.sin(t * z[j]); }
        re /= n; im /= n;
        phi0 = Math.exp(-t * t / 2); w = phi0;
        T += w * ((re - phi0) * (re - phi0) + im * im) * dt;
      }
      return T;
    }
    var st = FigKit.mount(fig, { ratio: 0.50, minH: 300, maxH: 440, draw: draw });
    function draw(ctx, s) {
      var p = s.pal, dt = Math.max(0, s.t - lastT);
      lastT = s.t;
      if (morph.active) {
        morph.k = Math.min(1, morph.k + dt * 1.1);
        var e = morph.k * morph.k * (3 - 2 * morph.k);
        for (var m = 0; m < pts.length; m++) {
          pts[m].x = morph.from[m].x + (target[m].x - morph.from[m].x) * e;
          pts[m].y = morph.from[m].y + (target[m].y - morph.from[m].y) * e;
        }
        if (morph.k >= 1) { morph.active = false; cloud = 'iso'; syncButtons(); }
      } else if (!s.paused && !s.reduced) {
        theta += dt * 0.36;
      }
      var W = s.W, H = s.H;
      var cx = W * 0.215, cy = H * 0.46, R = Math.min(W, H) * 0.30, sc = R / 2.7;
      var rcx = W * 0.795, rbase = H * 0.66, rHalf = W * 0.165, rPeak = H * 0.40;
      var c = Math.cos(theta), sn = Math.sin(theta), i, z = [];
      for (i = 0; i < pts.length; i++) z.push(pts[i].x * c + pts[i].y * sn);
      var T = eppsPulley(z);
      FigKit.dashedCircle(ctx, cx, cy, R, p.rule);
      /* drop-lines from each point to its foot on the direction */
      ctx.strokeStyle = p.accent; ctx.lineWidth = 1;
      FigKit.alpha(ctx, 0.22, function () {
        for (i = 0; i < pts.length; i++) {
          ctx.beginPath();
          ctx.moveTo(cx + pts[i].x * sc, cy - pts[i].y * sc);
          ctx.lineTo(cx + z[i] * c * sc, cy - z[i] * sn * sc);
          ctx.stroke();
        }
      });
      var L = R * 1.16;
      ctx.strokeStyle = p.inkSoft; ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(cx - L * c, cy + L * sn); ctx.lineTo(cx + L * c, cy - L * sn);
      ctx.stroke();
      FigKit.arrowHead(ctx, cx + L * c, cy - L * sn, Math.atan2(-sn, c), p.inkSoft);
      FigKit.alpha(ctx, 0.55, function () {
        ctx.fillStyle = p.accent;
        for (i = 0; i < pts.length; i++) {
          ctx.beginPath();
          ctx.arc(cx + z[i] * c * sc, cy - z[i] * sn * sc, 2.2, 0, 6.2832);
          ctx.fill();
        }
      });
      ctx.fillStyle = p.accent;
      for (i = 0; i < pts.length; i++) {
        ctx.beginPath();
        ctx.arc(cx + pts[i].x * sc, cy - pts[i].y * sc, 4.1, 0, 6.2832);
        ctx.fill();
      }
      ctx.fillStyle = p.inkDim; ctx.font = '12px ' + s.mono; ctx.textAlign = 'center';
      ctx.fillText('project onto a random direction', cx, cy + R + 34);
      var mx0 = W * 0.40, mx1 = W * 0.585;
      ctx.strokeStyle = p.rule; ctx.lineWidth = 1.6;
      ctx.beginPath(); ctx.moveTo(mx0, cy); ctx.lineTo(mx1, cy); ctx.stroke();
      FigKit.arrowHead(ctx, mx1, cy, 0, p.rule);
      ctx.fillStyle = p.inkDim; ctx.font = '11px ' + s.mono;
      ctx.fillText('shadow', (mx0 + mx1) / 2, cy - 12);
      ctx.strokeStyle = p.ruleSoft; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(rcx - rHalf - 6, rbase); ctx.lineTo(rcx + rHalf + 6, rbase); ctx.stroke();
      function vToX(v) { return rcx + (v / 4) * rHalf; }
      function dToY(d) { return rbase - (d / 0.42) * rPeak; }
      function bell(v) { return Math.exp(-v * v / 2) / 2.5066282746; }
      ctx.beginPath();
      for (var v = -4; v <= 4.0001; v += 0.1) {
        if (v <= -4) ctx.moveTo(vToX(v), dToY(bell(v))); else ctx.lineTo(vToX(v), dToY(bell(v)));
      }
      ctx.lineTo(vToX(4), rbase); ctx.lineTo(vToX(-4), rbase); ctx.closePath();
      ctx.fillStyle = p.accent;
      FigKit.alpha(ctx, 0.13, function () { ctx.fill(); });
      ctx.strokeStyle = p.accent; ctx.lineWidth = 1.4;
      FigKit.alpha(ctx, 0.5, function () {
        ctx.beginPath();
        for (var v2 = -4; v2 <= 4.0001; v2 += 0.1) {
          if (v2 <= -4) ctx.moveTo(vToX(v2), dToY(bell(v2))); else ctx.lineTo(vToX(v2), dToY(bell(v2)));
        }
        ctx.stroke();
      });
      /* the live shadow, smoothed just enough to read as a density */
      var h = 0.36, n = z.length, first = true;
      ctx.strokeStyle = p.accentInk; ctx.lineWidth = 2.4;
      ctx.beginPath();
      for (var q = -4; q <= 4.0001; q += 0.08) {
        var d = 0;
        for (var j = 0; j < n; j++) { var u = (q - z[j]) / h; d += Math.exp(-u * u / 2); }
        d /= (n * h * 2.5066282746);
        if (first) { ctx.moveTo(vToX(q), dToY(d)); first = false; } else ctx.lineTo(vToX(q), dToY(d));
      }
      ctx.stroke();
      ctx.textAlign = 'center';
      if (cloud === 'iso') {
        ctx.fillStyle = p.accentInk; ctx.font = '13px ' + s.mono;
        ctx.fillText('always ≈ N(0, 1)', rcx, rbase + 30);
      } else {
        ctx.fillStyle = p.bad; ctx.font = '13px ' + s.mono;
        ctx.fillText('shape depends on direction', rcx, rbase + 30);
      }
      ctx.fillStyle = p.inkDim; ctx.font = '11px ' + s.mono;
      ctx.fillText(cloud === 'iso' ? 'a standard bell curve' : 'target N(0,1) shown faint', rcx, rbase + 48);
      ctx.fillStyle = p.inkDim; ctx.font = '10.5px ' + s.mono; ctx.textAlign = 'right';
      ctx.fillText('epps-pulley gap', rcx + rHalf + 6, H * 0.15);
      ctx.fillStyle = T < 0.03 ? p.accentInk : (T < 0.10 ? p.amber : p.bad);
      ctx.font = '500 15px ' + s.mono;
      ctx.fillText('T = ' + T.toFixed(3), rcx + rHalf + 6, H * 0.15 + 20);
    }
    function syncButtons() {
      fig.querySelectorAll('[data-cloud]').forEach(function (b) {
        b.setAttribute('aria-pressed', b.dataset.cloud === cloud ? 'true' : 'false');
      });
      fig.querySelector('[data-act="fix"]').disabled = (cloud === 'iso');
    }
    fig.querySelectorAll('[data-cloud]').forEach(function (b) {
      b.addEventListener('click', function () { seed(b.dataset.cloud); syncButtons(); st.repaint(); });
    });
    fig.querySelector('[data-act="fix"]').addEventListener('click', function () {
      if (cloud === 'iso') return;
      morph.from = pts.map(function (q) { return { x: q.x, y: q.y }; });
      morph.k = 0; morph.active = true;
      st.repaint();
    });
    fig.querySelector('[data-act="pause"]').addEventListener('click', function () {
      st.paused = !st.paused;
      this.textContent = st.paused ? 'play' : 'pause';
      st.repaint();
    });
  })();
  </script>
</figure>

### The whole method, in one equation

$$\mathcal{L}_{\text{LeJEPA}} \;=\; \underbrace{\mathcal{L}_{\text{pred}}}_{\text{predict matched views}} \;+\; \lambda \cdot \underbrace{\text{SIGReg}}_{\text{stay isotropic Gaussian}}$$

One prediction term, one regularizer, **one knob** $\lambda$. And now collect the payoff:

- **No heuristics.** Because SIGReg makes collapse *structurally impossible*, you can delete the stop-gradient, the EMA teacher, the predictor asymmetry, the centering/sharpening, the schedules. The tape comes off.
- **One hyperparameter**, stable across 60+ architectures and 10+ datasets, from tiny ViTs up to ~1.8B params, with no per-model babysitting.
- **The loss finally means something.** Because the objective is grounded, *training loss correlates with downstream accuracy* — you can do model selection **without labels**, the thing that was impossible above.
- **In-domain SSL can beat frontier transfer.** Because it scales to any domain, pretraining LeJEPA *in-domain* (medical, satellite, documents, EEG) can beat fine-tuning a giant general model like DINOv2. For specialist domains this is the headline.

> And one line for LeJEPA: *prove the ideal feature distribution is an isotropic Gaussian, enforce it directly and cheaply by checking that random shadows of the embeddings all look like a bell curve, and every anti-collapse hack becomes unnecessary.*

Which recasts everything in Chapter 4. Every trick in that table was a *partial, implicit* approximation of "be an isotropic Gaussian." VICReg's variance and covariance terms pin the first two moments, and pinning the first two moments is as close as you get to the Gaussian without ever naming it. EMA and stop-gradient hold the cloud open dynamically, with no explicit target at all. What looked like a zoo of competing ideas reads better in hindsight as one idea seen through frosted glass from a dozen angles.

## 8. Beyond vision: JEPA as a training principle

One last turn before we get our hands dirty. Every method so far (I-JEPA, V-JEPA 2, LeJEPA) has been about images or video. But look back at the move at the heart of all of them: *encode two views, and predict one's embedding from the other's.* Nothing in that sentence mentions pixels. So it's worth asking whether the idea was ever really about vision, and the answer turns out to be no.

**LLM-JEPA** ([Huang, LeCun & Balestriero, 2025](https://arxiv.org/abs/2509.14252)) carries the JEPA objective into **language models**. It needs only one ingredient: data that comes in *paired views of the same meaning*. Plenty does — a plain-English description and the `\d+` regex that implements it, a question and its SQL, a word problem and its worked solution. Run each view through the LLM, take the last token's final hidden state as that view's embedding, and train a predictor to map one embedding onto the other. It's the Siamese-plus-predictor shape from Chapter 4, with text on both sides.

But there's a twist, and it pays off the fork from Chapter 2. Back there, predicting in *input space* (Branch A) and predicting in *embedding space* (Branch B) were rival roads, and for vision Branch B won outright — you could throw the pixels away. Language can't do that: an LLM still has to *generate*, token by token, so you can't discard next-token prediction the way I-JEPA discards pixels. So LLM-JEPA doesn't replace it — it **adds** the JEPA term on top:

$$\mathcal{L}_{\text{LLM-JEPA}} \;=\; \underbrace{\mathcal{L}_{\text{NTP}}}_{\text{generate (Branch A)}} \;+\; \lambda \cdot \underbrace{d\big(\text{Pred}(\text{Enc}(\text{text})),\ \text{Enc}(\text{code})\big)}_{\text{align embeddings (Branch B)}}$$

The two roads of Chapter 2, finally running in harness: the generative term keeps the model able to *produce* the answer, and the JEPA term shapes its *representation* so the two views land in the same place. One network plays every part — it encodes both views with shared weights, and it *is* its own predictor: append a few learnable `[PRED]` tokens and the hidden state over them is the prediction (use none and the predictor is the identity). The distance $d$ is cosine. That single extra term, the paper reports, buys accuracy *and* resistance to overfitting across regex, SQL, and math benchmarks.

<figure class="figure">
  <svg viewBox="0 0 760 300" width="760" role="img" style="display:block;width:100%;max-width:720px;margin:0 auto;height:auto;" aria-label="LLM-JEPA: the same shared LLM encodes two views of one item — a natural-language description and its code; a predictor formed from appended PRED tokens maps the text embedding toward the code embedding, scored by cosine distance (the JEPA term), while ordinary next-token prediction trains generation on top.">
    <defs>
      <marker id="lj-arrow" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0 L10 5 L0 10 z" fill="var(--ink-soft)"/></marker>
    </defs>
    <style>
      .ljbox  { fill:var(--bg-alt); stroke:var(--accent); stroke-width:2; }
      .ljnode { fill:var(--bg-alt); stroke:var(--rule); }
      .ljd    { fill:var(--bg-alt); stroke:#d9534f; stroke-width:2; }
      .ljarr  { stroke:var(--ink-soft); stroke-width:1.5; fill:none; }
      .ljshare{ stroke:var(--ink-dim); stroke-width:1.5; fill:none; stroke-dasharray:5 4; }
      .ljbt   { font-family:var(--font-mono); font-size:13px; fill:var(--ink); }
      .ljsub  { font-family:var(--font-mono); font-size:10.5px; fill:var(--ink-dim); }
      .ljel   { font-family:var(--font-mono); font-size:11px; fill:var(--ink-soft); }
    </style>
    <!-- NTP annotation -->
    <text class="ljel" x="258" y="22" text-anchor="middle">next-token loss · generate</text>
    <path class="ljarr" d="M258 56 L258 32" marker-end="url(#lj-arrow)"/>
    <!-- top path: text view -->
    <rect class="ljnode" x="20" y="56" width="156" height="46" rx="6"/>
    <text class="ljbt" x="98" y="78" text-anchor="middle">text view</text>
    <text class="ljsub" x="98" y="93" text-anchor="middle">“one or more digits”</text>
    <rect class="ljbox" x="214" y="56" width="88" height="46" rx="6"/>
    <text class="ljbt" x="258" y="84" text-anchor="middle">LLM</text>
    <rect class="ljbox" x="360" y="56" width="118" height="46" rx="6"/>
    <text class="ljbt" x="419" y="79" text-anchor="middle">predictor</text>
    <text class="ljsub" x="419" y="93" text-anchor="middle">[PRED] tokens</text>
    <rect class="ljnode" x="520" y="60" width="70" height="38" rx="6"/>
    <text class="ljbt" x="555" y="84" text-anchor="middle">ŝ</text>
    <!-- bottom path: code view -->
    <rect class="ljnode" x="20" y="214" width="156" height="46" rx="6"/>
    <text class="ljbt" x="98" y="236" text-anchor="middle">code view</text>
    <text class="ljsub" x="98" y="251" text-anchor="middle">\d+</text>
    <rect class="ljbox" x="214" y="214" width="88" height="46" rx="6"/>
    <text class="ljbt" x="258" y="242" text-anchor="middle">LLM</text>
    <rect class="ljnode" x="520" y="218" width="100" height="38" rx="6"/>
    <text class="ljbt" x="570" y="242" text-anchor="middle">Enc(code)</text>
    <!-- shared weights -->
    <path class="ljshare" d="M258 102 L258 214"/>
    <text class="ljsub" x="266" y="162" text-anchor="start">shared weights</text>
    <!-- loss -->
    <rect class="ljd" x="648" y="128" width="96" height="56" rx="6"/>
    <text class="ljbt" x="696" y="151" text-anchor="middle">D</text>
    <text class="ljsub" x="696" y="167" text-anchor="middle">cosine dist</text>
    <text class="ljsub" x="696" y="116" text-anchor="middle">JEPA term</text>
    <!-- arrows: top -->
    <path class="ljarr" d="M176 79 L214 79" marker-end="url(#lj-arrow)"/>
    <path class="ljarr" d="M302 79 L360 79" marker-end="url(#lj-arrow)"/>
    <text class="ljel" x="326" y="70" text-anchor="middle">Enc(text)</text>
    <path class="ljarr" d="M478 79 L520 79" marker-end="url(#lj-arrow)"/>
    <!-- arrows: bottom -->
    <path class="ljarr" d="M176 237 L214 237" marker-end="url(#lj-arrow)"/>
    <path class="ljarr" d="M302 237 L520 237" marker-end="url(#lj-arrow)"/>
    <!-- into D -->
    <path class="ljarr" d="M590 79 L648 145" marker-end="url(#lj-arrow)"/>
    <path class="ljarr" d="M620 237 L648 168" marker-end="url(#lj-arrow)"/>
  </svg>
  <figcaption>LLM-JEPA. The same LLM (shared weights) encodes two views of one item — a natural-language description and the code it specifies. A predictor, formed by appending learnable <code>[PRED]</code> tokens, maps the text embedding toward the code embedding; their cosine distance is the JEPA term. Ordinary next-token prediction still trains generation on top, so the fine-tuned model stays a normal, generating LLM — the JEPA term only sharpens its representation.</figcaption>
</figure>

> If you keep one line about LLM-JEPA, keep this: *it's the proof that JEPA was never a vision trick. Bolt an embedding-prediction term onto ordinary next-token training, and the same "shape the representation, don't just match the output" principle that drove the whole vision story lifts a language model too.*

So the thread that began with a cat flinching at a falling mug ends up somewhere wider than vision: a general stance on learning, *predict in representation space*, that pays off wherever you can form two views of the same thing. Part II makes all of it runnable.

---

# Part II — JEPA on your Mac

Everything above is theory and big-lab compute. This half is the opposite: concrete, local, and runnable. [`mlx-tune`](https://github.com/ARahim3/mlx-tune) brings the JEPA family to Apple Silicon: train **LeJEPA** from scratch, or load Meta's pretrained **I-JEPA** (images) and **V-JEPA 2** (video) and fine-tune them with LoRA, all natively on MLX, no CUDA.

> To my knowledge this is the first solid, tested, train-and-fine-tune JEPA implementation on Apple Silicon. Both pretrained ports are *numerically identical to the official HuggingFace PyTorch models* (cosine similarity 1.000000), verified in the test suite — the features you get on your Mac are bit-for-bit the reference encoder's, not an approximation.

## 9. Setup and the tracks

```bash
pip install mlx-tune
```

→ Source: [github.com/ARahim3/mlx-tune](https://github.com/ARahim3/mlx-tune) — and every example cited below links straight to its runnable file.

`mlx-tune` exposes JEPA through an Unsloth-flavoured API. Pick the track that matches your data and goal — each maps cleanly onto a chapter from Part I:

| Track | What it is | From Part I | Use it to |
| --- | --- | --- | --- |
| **LeJEPA** | random-init ViT, train from scratch | Chapter 7 | learn representations on *your* unlabeled images, no labels |
| **I-JEPA** | Meta's pretrained image encoder (ViT-H/14, 630M) | Chapter 5 | feature extraction, probing, LoRA fine-tuning |
| **V-JEPA 2** | Meta's pretrained video encoder + predictor (ViT-L, 326M) | Chapter 6 | clip features, video classification, anticipation & surprise |
| **LeWM** | train a latent world model from pixels | Chapter 6 | plan in latent space (CEM/MPC), toy control |
| **LLM-JEPA** | a JEPA term added to LLM fine-tuning | Chapter 8 | sharpen an LLM fine-tune on paired-view data |

> **Why does LoRA apply to a vision model?** LoRA isn't LLM-specific: it works on any model built from large linear layers, and a transformer is mostly linear layers (the q/k/v/out projections plus the MLP), whether its inputs are words or image patches. So you can LoRA-fine-tune a 630M-parameter JEPA encoder on a Mac while training well under 1% of the weights.

## 10. LeJEPA from scratch

This is Chapter 7 made real: a single ViT, the prediction loss plus SIGReg, and the one knob `lam` (the $\lambda$ from the equation, the SIGReg weight). No labels.

```python
from mlx_tune import FastJEPAModel, JEPATrainer, JEPAConfig, linear_probe

# 1. A randomly-initialised ViT. Presets: vit-debug / vit-tiny / vit-small / vit-base.
model, _ = FastJEPAModel.from_pretrained("vit-tiny", img_size=128)

# 2. Self-supervised pretraining — `images` is just a list of HWC arrays / PIL images.
trainer = JEPATrainer(
    model,
    args=JEPAConfig(num_epochs=5, batch_size=64, lam=0.05),  # lam = the SIGReg weight
    train_dataset=images,
)
trainer.train()

# 3. Use the learned encoder: frozen features, or a quick linear probe.
feats = model.encode(images)                       # (N, dim) frozen features
acc = linear_probe(model, tr_x, tr_y, te_x, te_y)  # representation-quality check

# 4. Save / reload.
model.save_pretrained("my_lejepa")
model, _ = FastJEPAModel.from_pretrained("my_lejepa")
```

Remember the LeJEPA payoff about the loss being meaningful? At the paper's scale that's a real convenience — the SIGReg-regularized training loss tracks downstream quality, so you can compare runs and pick checkpoints *without* labels. The single `lam` (~0.05) is the only knob you'll touch.

> **A reality check on scale.** From-scratch LeJEPA is the one track where Mac-scale bites hardest. At the small config above (ViT-tiny, batch 64, a few hundred steps) the encoder *collapses* — the embeddings shrink toward a point and a linear probe sits near chance. That's a budget effect rather than a bug: the SIGReg loss is verified correct, and a larger `lam` provably holds the variance up, but the paper's full result needs batch ≥ 128 and ~100 epochs. So on a Mac, treat from-scratch LeJEPA as the *API* plus the *in-domain warm-start* path (Chapter 11); for strong features today, start from pretrained I-JEPA. The "loss tracks accuracy" convenience is a property of the converged, paper-scale run — not a few hundred local steps.

→ Runnable example: [`examples/58_lejepa_pretraining.py`](https://github.com/ARahim3/mlx-tune/blob/main/examples/58_lejepa_pretraining.py) (LeJEPA on a CIFAR-10 subset, with a synthetic fallback so it runs offline → linear probe).

**Real corpora, not just in-memory lists.** Point `JEPATrainer` at an `ImageFolderDataset("/path/to/images")` and it streams images lazily from disk — the set never has to fit in RAM — while `save_steps` plus `resume=True` checkpoint the encoder and optimizer so an interrupted run picks up where it left off. That's what turns "LeJEPA from scratch on your Mac" from a demo into an overnight job on a real dataset.

## 11. I-JEPA: pretrained features, probes, and LoRA

Load Meta's pretrained encoder and put its features to work. Inputs are 224×224 (the resolution it was trained at). Two sizes load from the same one-liner — the 630M ViT-H/14 used below, or the billion-parameter ViT-g/16 (`facebook/ijepa_vitg16_22k`) when you want more capacity.

```python
from mlx_tune import FastJEPAModel, linear_probe, knn_probe, attentive_probe

# Downloads ~2.5 GB on first run, then converts HF → MLX (no torch needed).
model, _ = FastJEPAModel.from_pretrained("facebook/ijepa_vith14_1k")

# Three frozen-feature probes — none of them backprop through the 630M encoder:
linear    = linear_probe(model, tr_x, tr_y, te_x, te_y)     # logistic reg on pooled features
knn       = knn_probe(model, tr_x, tr_y, te_x, te_y, k=20)   # training-free sanity check
attentive = attentive_probe(model, tr_x, tr_y, te_x, te_y)   # attention-pooling head

feats  = model.encode(my_images)          # (N, 1280) mean-pooled
tokens = model.encode_tokens(my_images)   # (N, T, 1280) per-patch tokens
```

> **Use the attentive probe for the real numbers.** I-JEPA and V-JEPA 2 encoders don't mean-pool cleanly — a plain linear probe on pooled features *under-reads* their quality. `attentive_probe` trains a small attention-pooling head over the token features; it's the evaluation both papers actually use. `knn_probe` is a cheap, training-free gut check. (This is about *readout strength* — the encoder features themselves are identical regardless of probe.)

To adapt the encoder itself, attach a head and choose how much to train:

| `finetune=` | What trains | Use when |
| --- | --- | --- |
| `"frozen"` | just the linear head | fast, tiny data, sanity check |
| `"lora"` (default) | LoRA adapters in every block + head (<1% of params) | adapt cheaply without overfitting |
| `"full"` | everything | lots of data, max accuracy |

```python
from mlx_tune import FastJEPAModel, JEPAClassifierTrainer, JEPAClassifierConfig

model, _ = FastJEPAModel.from_pretrained("facebook/ijepa_vith14_1k")
clf = FastJEPAModel.for_image_classification(model, num_classes=10, finetune="lora", r=8)

trainer = JEPAClassifierTrainer(
    clf,
    JEPAClassifierConfig(img_size=224, batch_size=6, num_epochs=5),
    train_images, train_labels, eval_images=val_images, eval_labels=val_labels,
)
trainer.train()
print(f"accuracy: {trainer.evaluate():.3f}")
```

> **LoRA on a deep ViT needs warmup.** The classifier config defaults to `learning_rate=3e-4` with `warmup_ratio=0.15`. It matters: LoRA adapters perturbing all 24–32 transformer layers from step 0 can diverge without it.

**Warm-start LeJEPA — the two halves meet.** Because LeJEPA and the pretrained encoders are *all just ViTs*, you can continue self-supervised pretraining *from* a pretrained checkpoint — take Meta's ImageNet-trained I-JEPA and keep training it on your own unlabeled domain images with the LeJEPA/SIGReg objective. This is exactly the "in-domain SSL" payoff from Chapter 7, in three lines:

```python
model, _ = FastJEPAModel.from_pretrained("facebook/ijepa_vith14_1k")   # strong start
JEPATrainer(model, args=JEPAConfig(img_size=224, num_epochs=3, learning_rate=1e-4),
            train_dataset=my_domain_images).train()                    # refine with SIGReg
model.save_pretrained("ijepa_my_domain")
```

Use a *lower* learning rate (e.g. `1e-4`) than a from-scratch run — you're refining strong features, not learning from nothing, so you don't want to wash out the pretrained representation.

→ Runnable example: [`examples/59_ijepa_feature_extraction.py`](https://github.com/ARahim3/mlx-tune/blob/main/examples/59_ijepa_feature_extraction.py) (linear probe + LoRA fine-tune + save/load/predict).

## 12. V-JEPA 2: video features and fine-tuning

Same downstream API, lifted to video. Clips are `(T, 256, 256, 3)` arrays; `T` must be a multiple of the tubelet size (2), and frames are resized to 256×256 automatically. The ViT-L (326M) below is the default; the larger ViT-H/g checkpoints (up to ~1B, `facebook/vjepa2-vit{h,g}-fpc64-*`) load the same way.

```python
from mlx_tune import (
    FastVideoJEPAModel, video_linear_probe, VideoClassifierTrainer, VideoClassifierConfig,
)

model, _ = FastVideoJEPAModel.from_pretrained("facebook/vjepa2-vitl-fpc64-256")

clip_feats = model.encode(my_clips)                            # (N, 1024) frozen features
acc = video_linear_probe(model, tr_x, tr_y, te_x, te_y)        # or video_attentive_probe

# LoRA fine-tune the video encoder + a classification head:
clf = FastVideoJEPAModel.for_video_classification(model, num_classes=2, finetune="lora", r=8)
trainer = VideoClassifierTrainer(
    clf, VideoClassifierConfig(batch_size=2, num_epochs=5),
    train_clips, train_labels, eval_videos=val_clips, eval_labels=val_labels,
)
trainer.train()
```

**Anticipation and surprise: the predictor is here too.** The encoder is only half of what Stage 1 trained. The other half, the masked-latent **predictor**, is ported as well (same bit-for-bit parity, loads by default). Hand it the latents of a clip's first few frames and it imagines the latents of the rest, which is the action *anticipation* from Chapter 6, running locally. Better still, compare what it imagined against what actually happened and you get a number for how *surprising* the future turned out to be. That number is an old friend: the energy $D$ from the Chapter 6 figure, prediction error in representation space, as a function call.

```python
from mlx_tune import latent_energy

# watch the first 4 frames, imagine the rest
predicted, actual = model.predict_latents(clip, context_frames=4)

energy = latent_energy(predicted, actual)                  # scalar surprise
e_map  = latent_energy(predicted, actual, per_token=True)  # where the surprise lives
```

A coherent clip scores low; one with a hard cut or an impossible event scores high. That's label-free anomaly detection in latent space — the energy landscape from Chapter 3, finally something you can poke at.

**Meta's fine-tuned action classifiers, zero training.** The `VJEPA2ForVideoClassification` checkpoints (attentive pooler + head, fine-tuned on Something-Something-v2) also load straight in — `from_pretrained` auto-detects them, and the logits match HuggingFace:

```python
clf, _ = FastVideoJEPAModel.from_pretrained("facebook/vjepa2-vitl-fpc16-256-ssv2")
results = clf.predict([clip], top_k=5)    # 174 action classes, no training
```

> The scope line, redrawn: encoder *and* predictor are ported — features, fine-tuning, anticipation, surprise. What remains out is **V-JEPA 2-AC**, the separately post-trained *action-conditioned* predictor, and its CEM/MPC robot-planning loop (Chapter 6). If it's a trainable world model with planning you're after, that's the very next section — just built from scratch rather than loaded.

→ Runnable examples: [`examples/60_vjepa2_video.py`](https://github.com/ARahim3/mlx-tune/blob/main/examples/60_vjepa2_video.py) (video probes + LoRA fine-tune + save/load/predict) and [`examples/64_vjepa2_predictor_classifier.py`](https://github.com/ARahim3/mlx-tune/blob/main/examples/64_vjepa2_predictor_classifier.py) (anticipation + surprise energy, and the pretrained SSv2 classifier).

## 13. LeWM: a world model you can actually train

Chapter 6 left an IOU. V-JEPA 2 *plans* in latent space, but loading Meta's pretrained action-conditioned planner and running it on a Mac isn't on the table. **LeWM** (*LeWorldModel*, [Maes et al., 2026](https://arxiv.org/abs/2603.19312)) settles that IOU from the opposite direction: instead of *loading* a giant pretrained world model, you *train a small one from scratch*, end-to-end from pixels, and plan with it locally.

It's the two threads of Part I finally braided together — the world-model energy from Chapter 6, regularized by the trick from Chapter 7:

$$\mathcal{L}_{\text{LeWM}} \;=\; \underbrace{\mathcal{L}_{\text{pred}}}_{\text{predict the next latent}} \;+\; \lambda \cdot \underbrace{\text{SIGReg}}_{\text{don't collapse}}$$

Two loss terms, one knob, and because SIGReg makes collapse structurally impossible, *no* stop-gradient and *no* EMA target. The same "tape comes off" story as LeJEPA, now for dynamics. What you get is a trainable latent simulator: encode the current frame plus a candidate action, predict the next latent, and **plan** by searching for the action sequence whose imagined latent lands closest to a goal latent — CEM / MPC, re-planning after each step. That's the planning loop from Chapter 6, running on your Mac.

```python
from mlx_tune import FastWorldModel, LeWMConfig, LeWMTrainer, plan_cem, PointMassEnv

env = PointMassEnv(size=48)                    # toy 2-D control environment
data = env.collect(n_episodes=80, ep_len=10)   # {"frames", "actions"} trajectories

# train the latent world model: next-latent prediction + SIGReg
model = FastWorldModel.from_pretrained("lewm-tiny", img_size=48, action_dim=2)
LeWMTrainer(model, LeWMConfig(img_size=48, action_dim=2, sigreg_lambda=0.05), data).train()

# plan: search for the actions whose imagined latent lands on a goal latent (MPC)
goal_z = model.encode([env.render([0.8, 0.2])])[0]
action = plan_cem(model, model.encode([env.render()])[0], goal_z, horizon=3)
```

> The honest scope, kept honest. The pipeline is real and the planner is exact — on the bundled toy dynamics (a 2-D point-mass) it drives the planning cost to zero. What's demo-scale is the *setting*, not the code: matching a paper-scale controller on real robots needs real trajectories and real compute. So read this as "the train-a-world-model-then-plan loop, faithfully implemented and runnable locally — prototype here, scale the same code up," not "a SOTA robot policy on a MacBook." And it stays distinct from V-JEPA 2-AC: *train your own and plan* is here today; *load Meta's pretrained planner and plan* is a future release.

→ Runnable example: [`examples/62_lewm_world_model.py`](https://github.com/ARahim3/mlx-tune/blob/main/examples/62_lewm_world_model.py) (train a latent world model on a toy point-mass, then plan with CEM/MPC).

## 14. LLM-JEPA: improve an LLM fine-tune

This is Chapter 8 made real — the one track that isn't vision. Hand the trainer paired views (a description and its regex, a question and its SQL) and it adds the JEPA alignment term on top of an ordinary LoRA fine-tune. The result is a **perfectly normal LoRA-fine-tuned LLM**: generate, merge, and serve it exactly as you would after plain SFT — the JEPA term does its work during training, then steps aside.

```python
from mlx_tune import FastLanguageModel, LLMJEPATrainer, LLMJEPAConfig

model, tokenizer = FastLanguageModel.from_pretrained("mlx-community/Qwen3.5-0.8B-MLX-4bit")
model = FastLanguageModel.get_peft_model(model, r=16)

# two views of the same item: a description and its regex
data = [{"text": "match one or more digits", "code": r"\d+"}, ...]

LLMJEPATrainer(model, data, tokenizer=tokenizer,
               args=LLMJEPAConfig(jepa_lambda=0.1, jepa_distance="cosine")).train()
```

Set `jepa_lambda=0` and you fall back to a plain fine-tune — a clean A/B baseline for measuring what the JEPA term actually buys.

→ Runnable example: [`examples/61_llm_jepa_finetuning.py`](https://github.com/ARahim3/mlx-tune/blob/main/examples/61_llm_jepa_finetuning.py) (natural-language → regex views, Qwen3.5-0.8B).

## 15. Where JEPA actually shines — and the honest limits

**When to reach for JEPA over a generic pretrained model.** The clearest win, straight from the LeJEPA result, is **specialist domains**. If your data isn't natural-internet imagery (medical scans, satellite tiles, documents, scientific imaging), pretraining your *own* encoder in-domain with LeJEPA can beat fine-tuning a giant general model, across data regimes from one-shot to fully supervised. The warm-start path in Chapter 11 is the pragmatic middle: start from Meta's I-JEPA, adapt it to your domain with SIGReg, *then* fine-tune. And you're not boxed into classification. The same frozen/LoRA/full machinery drives **regression and dense heads** too, meaning object counting, depth maps and segmentation, which are I-JEPA's *own* paper-headline tasks ([`examples/63_jepa_dense_regression.py`](https://github.com/ARahim3/mlx-tune/blob/main/examples/63_jepa_dense_regression.py)). For plain feature extraction — clustering, retrieval, a quick probe — the pretrained encoders are excellent off the shelf.

**Save, reload, predict — verified end to end.** A trained classifier saves the encoder, the LoRA adapters, and the head together; reloading reconstructs the architecture and produces *identical* predictions (covered by the test suite for frozen, LoRA, and full modes).

```python
clf.save_pretrained("my_classifier")
clf = FastJEPAModel.load_classifier("my_classifier")   # video: FastVideoJEPAModel.load_classifier
preds = clf.predict(new_images)                        # class ids   (N,)
probs = clf.predict(new_images, return_probs=True)     # softmax probs (N, num_classes)
```

**Practical notes.**

- **Memory (M4 Pro, 48 GB).** I-JEPA LoRA fine-tune (batch 6, 224×224) peaks ~13 GB; V-JEPA 2 LoRA (8-frame clip) peaks ~7.6 GB. Frozen-feature probing is far lighter — no backprop through the encoder.
- **Tuned to keep the GPU fed.** Attention runs through MLX's fused `scaled_dot_product_attention`, the trainers raise the wired-memory limit, and image/video preprocessing runs on a background thread that overlaps GPU compute — so there's no slow, stuttering first epoch.
- **Input sizes.** V-JEPA 2 wants 256×256 with `T` a multiple of 2 — its RoPE grid is tied to that. I-JEPA is now flexible: the pretrained position embeddings are bicubically interpolated at load, so you can run it at, say, 384×384 by passing `img_size=384` to `from_pretrained`. The loaders resize for you.
- **Small-scale results are demos.** Accuracy on a small dataset on a Mac is a capability demonstration, not a SOTA claim. Prototype here; scale the same code on a bigger machine.

**The honest limits.** Two boundaries are worth stating plainly. The first is scope — for the *pretrained* models, `mlx-tune` reaches the encoders *and* V-JEPA 2's masked-latent predictor, but still not Meta's action-conditioned **V-JEPA 2-AC** planner. You can train and plan with your *own* latent world model (LeWM, Chapter 13), but loading the famous robot controller and planning with it isn't here yet. The second is theoretical — LeJEPA's optimality and the [2026 identifiability results](https://klindtlab.github.io/lejepa-identifiability/) assume the latent world is well-described by an *isotropic Gaussian with stationary, additive-noise* dynamics. Worlds with fat tails, phase transitions, or strong nonlinear feedback (markets, some physical systems) violate those assumptions, and the guarantees weaken there. I'd read that as a map rather than a flaw, since it tells you exactly where the theory holds.

---

> The whole arc, in one breath: *animals build world models by watching. To copy that, predict the missing parts of observations, but predict their **embeddings** rather than their pixels, so you can ignore unpredictable detail. Latent prediction wants to collapse, and for a decade everyone fought collapse with fragile, unexplained tricks. Then someone proved the ideal feature distribution is an isotropic Gaussian and enforced it directly by checking that random shadows of the features all look like a bell curve, and the tricks fell away. `mlx-tune` puts the runnable pieces of that story on your Mac: LeJEPA, I-JEPA, V-JEPA 2, a world model you can train, and the same objective reaching into language.*

For the API in depth, see the [`mlx-tune` JEPA docs](https://arahim3.github.io/mlx-tune/jepa.html), [the source on GitHub](https://github.com/ARahim3/mlx-tune), and [the runnable examples](https://github.com/ARahim3/mlx-tune/tree/main/examples). The primary sources are LeCun's *A Path Towards Autonomous Machine Intelligence* (2022), I-JEPA (Assran et al., 2023), the invariance-vs-equivariance analysis in [*Learning and Leveraging World Models in Visual Representation Learning*](https://arxiv.org/abs/2403.00504) (Garrido et al., 2024), V-JEPA 2 (Meta, 2025), LeJEPA (Balestriero & LeCun, 2025), the identifiability result [*When Does LeJEPA Learn a World Model?*](https://klindtlab.github.io/lejepa-identifiability/) (Klindt, LeCun & Balestriero, 2026), LeWM (Maes et al., 2026), and LLM-JEPA (Huang, LeCun & Balestriero, 2025).
