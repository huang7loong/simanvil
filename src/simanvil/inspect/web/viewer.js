/* SimAnvil v0.1 Universal Multi-Model Physics Session Viewer
 * Strict Z-Up convention: THREE.Object3D.DefaultUp = (0, 0, 1)
 * MuJoCo quaternion [w, x, y, z] -> Three.js Quaternion(x, y, z, w)
 */
(function () {
  "use strict";

  THREE.Object3D.DefaultUp.set(0, 0, 1);

  const STAGE_PALETTE = [
    "#0284c7",
    "#0d9488",
    "#d97706",
    "#7c3aed",
    "#059669",
    "#db2777",
    "#4f46e5",
    "#475569",
  ];

  const state = {
    ready: false,
    session: null,
    runKey: null,
    activeModelKey: null,
    activeCamPreset: "hero",
    layerMode: "ghost", // 'visual' | 'ghost' | 'hulls'
    showForces: true,
    showWireframe: true,
    playing: true,
    speed: 1.0,
    frame: 0,
    accumSec: 0,
    lastTickMs: performance.now(),
  };

  window.__SIMANVIL_VIEWER_STATE__ = state;

  // DOM Elements
  const container = document.getElementById("canvas-container");
  const runSwitcherEl = document.getElementById("run-switcher");
  const vidOverview = document.getElementById("vid-overview");
  const vidWrist = document.getElementById("vid-wrist");
  const lblCamOverview = document.getElementById("lbl-cam-overview");
  const lblCamWrist = document.getElementById("lbl-cam-wrist");
  const slider = document.getElementById("timeline-slider");
  const btnPlay = document.getElementById("btn-play");
  const btnPrev = document.getElementById("btn-prev");
  const btnNext = document.getElementById("btn-next");
  const selSpeed = document.getElementById("sel-speed");
  const frameCounter = document.getElementById("frame-counter");
  const timeCounter = document.getElementById("time-counter");
  const phaseRibbon = document.getElementById("phase-ribbon");
  const verdictPill = document.getElementById("verdict-pill");
  const verdictDetail = document.getElementById("verdict-detail");
  const contactCountEl = document.getElementById("contact-count");
  const peakForceEl = document.getElementById("peak-force-readout");
  const contactListEl = document.getElementById("contact-list");
  const compilerSummaryEl = document.getElementById("compiler-summary");

  const annG1 = document.getElementById("ann-g1");
  const annG2 = document.getElementById("ann-g2");
  const annG3 = document.getElementById("ann-g3");
  const annProg = document.getElementById("ann-prog");
  const annWeight = document.getElementById("ann-weight");

  const valPadF = document.getElementById("val-pad-f");
  const valTrayF = document.getElementById("val-tray-f");
  const chart2LegendEl = document.getElementById("chart2-legend");

  const chartForces = document.getElementById("chart-forces");
  const chartSeating = document.getElementById("chart-seating");

  // Three.js Scene & Renderer
  const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.outputEncoding = THREE.sRGBEncoding;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.12;
  container.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(40, 16 / 10, 0.01, 16.0);
  const controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;

  let activePresets = {
    hero: { pos: [0.98, -0.52, 0.50], target: [0.46, -0.02, 0.08] },
    tray: { pos: [0.74, -0.16, 0.24], target: [0.51, 0.055, 0.042] },
    side: { pos: [0.51, -0.52, 0.09], target: [0.51, 0.03, 0.05] },
    top: { pos: [0.50, -0.001, 0.78], target: [0.49, 0.00, 0.03] },
  };

  function setCameraPreset(name) {
    state.activeCamPreset = name;
    const p = activePresets[name] || activePresets.hero;
    camera.position.set(p.pos[0], p.pos[1], p.pos[2]);
    controls.target.set(p.target[0], p.target[1], p.target[2]);
    controls.update();
    document.querySelectorAll(".cam-btn").forEach((b) => {
      b.classList.toggle("active", b.dataset.cam === name);
    });
  }

  // Studio Z-Up Lighting
  const hemi = new THREE.HemisphereLight(0xdbeafe, 0x1e293b, 0.88);
  hemi.position.set(0, 0, 2.5);
  scene.add(hemi);

  const keyLight = new THREE.DirectionalLight(0xffffff, 1.15);
  keyLight.position.set(1.1, -0.9, 1.8);
  scene.add(keyLight);

  const rimLight = new THREE.DirectionalLight(0x93c5fd, 0.55);
  rimLight.position.set(-0.9, 1.0, 1.2);
  scene.add(rimLight);

  // Z-Up XY Metric Floor Grid
  const grid = new THREE.GridHelper(2.4, 40, 0x33475b, 0x1c2734);
  grid.rotation.x = Math.PI / 2;
  grid.position.set(0.35, 0.0, -0.14);
  scene.add(grid);

  const axes = new THREE.AxesHelper(0.09);
  axes.position.set(0.18, -0.36, 0.021);
  scene.add(axes);

  // Multi-Model Cache: modelKey -> { rootGroup, records, spec }
  const worldRoot = new THREE.Group();
  scene.add(worldRoot);
  const loadedModels = {};

  // Reusable 3D Contact Force Arrow Pool
  const forceGroup = new THREE.Group();
  scene.add(forceGroup);
  const MAX_ARROWS = 40;
  const arrowPool = [];
  const sphereGeom = new THREE.SphereGeometry(0.0038, 12, 12);
  for (let i = 0; i < MAX_ARROWS; i++) {
    const arrow = new THREE.ArrowHelper(
      new THREE.Vector3(0, 0, 1),
      new THREE.Vector3(0, 0, 0),
      0.06,
      0x38bdf8,
      0.014,
      0.009
    );
    const dotMat = new THREE.MeshBasicMaterial({ color: 0x38bdf8 });
    const dot = new THREE.Mesh(sphereGeom, dotMat);
    arrow.visible = false;
    dot.visible = false;
    forceGroup.add(arrow);
    forceGroup.add(dot);
    arrowPool.push({ arrow, dot, dotMat });
  }

  const gltfLoader = new THREE.GLTFLoader();

  function loadGLB(url) {
    return new Promise((resolve, reject) => {
      if (!url) {
        resolve(null);
        return;
      }
      gltfLoader.load(
        url,
        (gltf) => resolve(gltf.scene),
        undefined,
        (err) => reject(err)
      );
    });
  }

  function decorateHullGroup(root, isShrinkwrap, seedOffset) {
    if (!root) return { mats: [], lines: [] };
    const mats = [];
    const lines = [];
    const meshes = [];
    root.traverse((node) => {
      if (node.isMesh) meshes.push(node);
    });
    meshes.forEach((mesh, idx) => {
      if (mesh.geometry) {
        mesh.geometry.computeVertexNormals();
      }
      const hullColor = new THREE.Color().setHSL(((idx + seedOffset) * 0.61803398875) % 1.0, 0.78, 0.56);
      const baseColor = isShrinkwrap ? new THREE.Color(0xef4444) : hullColor;
      const mat = new THREE.MeshStandardMaterial({
        color: baseColor,
        emissive: baseColor.clone().multiplyScalar(isShrinkwrap ? 0.35 : 0.18),
        roughness: 0.38,
        metalness: 0.12,
        transparent: true,
        opacity: isShrinkwrap ? 0.46 : 0.68,
        polygonOffset: true,
        polygonOffsetFactor: 1,
        polygonOffsetUnits: 1,
      });
      mesh.material = mat;
      mats.push(mat);

      const edges = new THREE.EdgesGeometry(mesh.geometry, 18);
      const lineMat = new THREE.LineBasicMaterial({
        color: isShrinkwrap ? 0xfca5a5 : 0xe2e8f0,
        transparent: true,
        opacity: isShrinkwrap ? 0.92 : 0.65,
      });
      const wire = new THREE.LineSegments(edges, lineMat);
      mesh.add(wire);
      lines.push(wire);
    });
    return { mats, lines };
  }

  function collectVisualMaterials(root) {
    const mats = [];
    if (!root) return mats;
    root.traverse((node) => {
      if (node.isMesh && node.material) {
        const cloned = node.material.clone();
        node.material = cloned;
        mats.push(cloned);
      }
    });
    return mats;
  }

  async function ensureModelLoaded(modelKey) {
    if (loadedModels[modelKey]) return loadedModels[modelKey];
    const spec = state.session.models[modelKey];
    const rootGroup = new THREE.Group();
    rootGroup.name = `model_${modelKey}`;
    worldRoot.add(rootGroup);

    const records = new Array(spec.bodies.length);
    await Promise.all(
      spec.bodies.map(async (b, idx) => {
        const grp = new THREE.Group();
        grp.name = b.name;
        rootGroup.add(grp);

        const [visScene, coacdScene, shrinkScene] = await Promise.all([
          loadGLB(b.visual_glb),
          loadGLB(b.coacd_glb),
          b.shrinkwrap_glb && b.shrinkwrap_glb !== b.coacd_glb ? loadGLB(b.shrinkwrap_glb) : Promise.resolve(null),
        ]);

        const visMats = collectVisualMaterials(visScene);
        if (visScene) grp.add(visScene);

        const coacdDec = decorateHullGroup(coacdScene, false, idx * 3);
        if (coacdScene) grp.add(coacdScene);

        let shrinkGroup = coacdScene;
        const colLines = [...coacdDec.lines];
        if (shrinkScene) {
          const shrinkDec = decorateHullGroup(shrinkScene, true, 0);
          colLines.push(...shrinkDec.lines);
          grp.add(shrinkScene);
          shrinkGroup = shrinkScene;
        }

        records[idx] = {
          meta: b,
          group: grp,
          visGroup: visScene,
          coacdGroup: coacdScene,
          shrinkGroup: shrinkGroup,
          visMats,
          colLines,
        };
      })
    );

    const entry = { rootGroup, records, spec };
    loadedModels[modelKey] = entry;
    return entry;
  }

  function applyLayerVisibility() {
    if (!state.activeModelKey || !loadedModels[state.activeModelKey]) return;
    const run = state.session.runs[state.runKey];
    const useShrink = Boolean(run && run.use_shrinkwrap_mesh);
    const bodyRecords = loadedModels[state.activeModelKey].records;

    bodyRecords.forEach((rec) => {
      const b = rec.meta;
      const activeHullGroup = useShrink ? rec.shrinkGroup : rec.coacdGroup;
      const inactiveHullGroup = useShrink ? rec.coacdGroup : rec.shrinkGroup;
      if (inactiveHullGroup && inactiveHullGroup !== activeHullGroup) {
        inactiveHullGroup.visible = false;
      }

      const isFocusPart = Boolean(b.is_focus_part || b.is_compiled_asset);

      if (state.layerMode === "visual") {
        if (rec.visGroup) {
          rec.visGroup.visible = true;
          rec.visMats.forEach((m) => {
            m.transparent = false;
            m.opacity = 1.0;
            m.depthWrite = true;
            m.needsUpdate = true;
          });
        }
        if (activeHullGroup) activeHullGroup.visible = false;
      } else if (state.layerMode === "ghost") {
        if (rec.visGroup) {
          rec.visGroup.visible = true;
          rec.visMats.forEach((m) => {
            if (isFocusPart) {
              m.transparent = true;
              m.opacity = 0.28;
              m.depthWrite = false;
            } else {
              m.transparent = false;
              m.opacity = 1.0;
              m.depthWrite = true;
            }
            m.needsUpdate = true;
          });
        }
        if (activeHullGroup) {
          activeHullGroup.visible = isFocusPart;
        }
      } else {
        if (activeHullGroup) {
          activeHullGroup.visible = true;
          if (rec.visGroup) rec.visGroup.visible = false;
        } else if (rec.visGroup) {
          rec.visGroup.visible = true;
        }
      }

      rec.colLines.forEach((ln) => {
        ln.visible = state.showWireframe;
      });
    });

    forceGroup.visible = state.showForces;
  }

  function syncVideosToFrame(run, frameIdx) {
    const targetTime = Math.min(
      Math.max(0, frameIdx / run.fps),
      Math.max(0, (run.frame_count - 1) / run.fps)
    );
    [vidOverview, vidWrist].forEach((vid) => {
      if (vid.readyState >= 1 && Math.abs(vid.currentTime - targetTime) > 0.018) {
        try {
          vid.currentTime = targetTime;
        } catch (_) {}
      }
    });
  }

  function drawLineSeries(ctx, xs, ys, color, lineWidth, fillAlpha) {
    if (!xs.length) return;
    ctx.beginPath();
    for (let i = 0; i < xs.length; i++) {
      if (i === 0) ctx.moveTo(xs[i], ys[i]);
      else ctx.lineTo(xs[i], ys[i]);
    }
    ctx.strokeStyle = color;
    ctx.lineWidth = lineWidth;
    ctx.stroke();

    if (fillAlpha > 0) {
      ctx.lineTo(xs[xs.length - 1], ctx.canvas.height - 18);
      ctx.lineTo(xs[0], ctx.canvas.height - 18);
      ctx.closePath();
      ctx.fillStyle = color.replace(")", `, ${fillAlpha})`).replace("rgb(", "rgba(");
      ctx.fill();
    }
  }

  function renderTelemetryCharts(run, frameIdx) {
    const tel = run.telemetry;
    const N = tel.length;
    if (!N) return;

    // Chart 1: Forces (N)
    {
      const ctx = chartForces.getContext("2d");
      const W = chartForces.width;
      const H = chartForces.height;
      ctx.clearRect(0, 0, W, H);
      const padL = 36, padR = 10, padT = 10, padB = 18;
      const plotW = W - padL - padR;
      const plotH = H - padT - padB;

      const maxF =
        Math.max(15.0, ...tel.map((t) => Math.max(t.pad_force_n || 0, t.part_tray_force_n || 0))) * 1.12;

      ctx.strokeStyle = "#1e293b";
      ctx.lineWidth = 1;
      ctx.fillStyle = "#6c7f96";
      ctx.font = "10px monospace";
      [0, 0.5, 1.0].forEach((frac) => {
        const y = padT + plotH * (1 - frac);
        ctx.beginPath();
        ctx.moveTo(padL, y);
        ctx.lineTo(W - padR, y);
        ctx.stroke();
        ctx.fillText(`${Math.round(maxF * frac)}N`, 2, y + 3);
      });

      const xs = tel.map((_, i) => padL + (i / Math.max(1, N - 1)) * plotW);
      const yPad = tel.map((t) => padT + plotH * (1 - (t.pad_force_n || 0) / maxF));
      const yTray = tel.map((t) => padT + plotH * (1 - (t.part_tray_force_n || 0) / maxF));

      drawLineSeries(ctx, xs, yPad, "rgb(56, 189, 248)", 1.8, 0.12);
      drawLineSeries(ctx, xs, yTray, "rgb(245, 158, 11)", 2.1, 0.16);

      const cx = padL + (frameIdx / Math.max(1, N - 1)) * plotW;
      ctx.strokeStyle = "#f8fafc";
      ctx.lineWidth = 1.4;
      ctx.beginPath();
      ctx.moveTo(cx, padT);
      ctx.lineTo(cx, H - padB);
      ctx.stroke();
    }

    // Chart 2: Dynamic Kinematics/Seating Series (mm) & Progress (%)
    {
      const ctx = chartSeating.getContext("2d");
      const W = chartSeating.width;
      const H = chartSeating.height;
      ctx.clearRect(0, 0, W, H);
      const padL = 36, padR = 10, padT = 10, padB = 18;
      const plotW = W - padL - padR;
      const plotH = H - padT - padB;

      const sdefs = run.series_defs || [];
      let allVals = [-10.0, 32.0];
      sdefs.forEach((sd) => {
        tel.forEach((t) => {
          if (typeof t[sd.key] === "number") allVals.push(t[sd.key]);
        });
      });
      const minMM = Math.min(...allVals) - 4.0;
      const maxMM = Math.max(...allVals) + 4.0;
      const span = Math.max(1e-3, maxMM - minMM);
      const mapMM = (v) => padT + plotH * (1 - (v - minMM) / span);

      tel.forEach((t, i) => {
        if (t.anomaly) {
          const x = padL + (i / Math.max(1, N - 1)) * plotW;
          ctx.fillStyle = "rgba(239, 68, 68, 0.14)";
          ctx.fillRect(x, padT, Math.ceil(plotW / N) + 1, plotH);
        }
      });

      ctx.font = "10px monospace";
      [minMM + span * 0.2, minMM + span * 0.8].forEach((val) => {
        const y = mapMM(val);
        ctx.save();
        ctx.setLineDash([4, 4]);
        ctx.strokeStyle = "#334155";
        ctx.beginPath();
        ctx.moveTo(padL, y);
        ctx.lineTo(W - padR, y);
        ctx.stroke();
        ctx.restore();
        ctx.fillStyle = "#94a3b8";
        ctx.fillText(`${Math.round(val)}`, 2, y + 3);
      });

      const xs = tel.map((_, i) => padL + (i / Math.max(1, N - 1)) * plotW);
      const yProg = tel.map((t) => padT + plotH * (1 - (t.task_progress_pct || 0) / 100.0));
      drawLineSeries(ctx, xs, yProg, "rgb(52, 211, 153)", 1.5, 0.08);

      sdefs.forEach((sd) => {
        const ys = tel.map((t) => mapMM(Number(t[sd.key] || 0)));
        drawLineSeries(ctx, xs, ys, sd.color || "rgb(248, 113, 113)", 2.0, 0.0);
      });

      const cx = padL + (frameIdx / Math.max(1, N - 1)) * plotW;
      ctx.strokeStyle = "#f8fafc";
      ctx.lineWidth = 1.4;
      ctx.beginPath();
      ctx.moveTo(cx, padT);
      ctx.lineTo(cx, H - padB);
      ctx.stroke();
    }
  }

  function updateChart2Legend(run, telFrame) {
    const sdefs = run.series_defs || [];
    const parts = sdefs.map((sd) => {
      const val = typeof telFrame[sd.key] === "number" ? telFrame[sd.key].toFixed(1) : "0.0";
      return `<i class="swatch" style="background:${sd.color}"></i>${sd.label} <strong class="mono">${val}${sd.unit || ""}</strong>`;
    });
    parts.push(`<i class="swatch green"></i>进度`);
    chart2LegendEl.innerHTML = parts.join(" ");
  }

  function applyFrame(frameIdx) {
    if (!state.session || !state.runKey) return;
    const run = state.session.runs[state.runKey];
    const modelEntry = loadedModels[state.activeModelKey];
    if (!run || !modelEntry) return;

    const clamped = Math.max(0, Math.min(run.frame_count - 1, frameIdx));
    state.frame = clamped;
    slider.value = String(clamped);

    const bodyRecords = modelEntry.records;
    const xposFrame = run.xpos[clamped];
    const xquatFrame = run.xquat[clamped];
    for (let i = 0; i < bodyRecords.length; i++) {
      const rec = bodyRecords[i];
      const p = xposFrame[i];
      const q = xquatFrame[i];
      if (rec && p && q) {
        rec.group.position.set(p[0], p[1], p[2]);
        rec.group.quaternion.set(q[1], q[2], q[3], q[0]);
      }
    }

    const contacts = run.contacts[clamped] || [];
    let peakMag = 0.0;
    for (let i = 0; i < MAX_ARROWS; i++) {
      const slot = arrowPool[i];
      if (i < contacts.length) {
        const c = contacts[i];
        peakMag = Math.max(peakMag, c.mag);
        const fx = c.force[0], fy = c.force[1], fz = c.force[2];
        const fLen = Math.hypot(fx, fy, fz);
        if (fLen > 1e-4) {
          const dir = new THREE.Vector3(fx / fLen, fy / fLen, fz / fLen);
          const arrowLen = Math.min(0.16, 0.022 + 0.0042 * c.mag);
          const colHex = c.mag > 20.0 ? 0xef4444 : c.kind === "gripper" ? 0x38bdf8 : 0xfbbf24;
          slot.arrow.position.set(c.pos[0], c.pos[1], c.pos[2]);
          slot.arrow.setDirection(dir);
          slot.arrow.setLength(arrowLen, Math.min(0.024, arrowLen * 0.32), Math.min(0.014, arrowLen * 0.2));
          slot.arrow.setColor(new THREE.Color(colHex));
          slot.arrow.visible = true;

          slot.dot.position.set(c.pos[0], c.pos[1], c.pos[2]);
          slot.dotMat.color.setHex(colHex);
          slot.dot.visible = true;
        } else {
          slot.arrow.visible = false;
          slot.dot.visible = false;
        }
      } else {
        slot.arrow.visible = false;
        slot.dot.visible = false;
      }
    }

    contactCountEl.textContent = String(contacts.length);
    peakForceEl.textContent = `Peak ${peakMag.toFixed(1)} N`;
    contactListEl.innerHTML = contacts
      .slice()
      .sort((a, b) => b.mag - a.mag)
      .slice(0, 5)
      .map((c) => `<li><span>${c.pair}</span><strong>${c.mag.toFixed(2)} N</strong></li>`)
      .join("");

    const t = run.telemetry[clamped];
    annG2.textContent = t.g2_stage;
    annG3.textContent = t.g3_primitive;
    annProg.textContent = `${Number(t.task_progress_pct || 0).toFixed(1)}%`;
    const wVal = Number(t.bc_loss_weight !== undefined ? t.bc_loss_weight : 1.0);
    annWeight.textContent = `${wVal.toFixed(2)}×`;
    annWeight.className = `mono ${wVal < 1.0 ? "weight-clamped" : "weight-clean"}`;

    valPadF.textContent = `${Number(t.pad_force_n || 0).toFixed(1)}N`;
    valTrayF.textContent = `${Number(t.part_tray_force_n || 0).toFixed(1)}N`;
    updateChart2Legend(run, t);

    frameCounter.textContent = `F ${String(clamped).padStart(3, "0")} / ${String(run.frame_count - 1).padStart(3, "0")}`;
    timeCounter.textContent = `${run.times[clamped].toFixed(2)} s`;

    syncVideosToFrame(run, clamped);
    renderTelemetryCharts(run, clamped);
  }

  function buildPhaseRibbon(run) {
    phaseRibbon.innerHTML = "";
    const stageColorMap = {};
    let colorCounter = 0;
    run.phases.forEach((seg) => {
      if (!stageColorMap[seg.stage]) {
        stageColorMap[seg.stage] = STAGE_PALETTE[colorCounter % STAGE_PALETTE.length];
        colorCounter += 1;
      }
      const spanLen = seg.end - seg.start + 1;
      const div = document.createElement("div");
      div.className = "phase-seg";
      div.style.flex = String(spanLen);
      div.style.background = stageColorMap[seg.stage];
      div.textContent = seg.stage.replace(/^S\d_/, "");
      div.title = `${seg.stage} (F${seg.start}–F${seg.end})`;
      div.addEventListener("click", () => {
        state.playing = false;
        updatePlayButton();
        applyFrame(seg.start);
      });
      phaseRibbon.appendChild(div);
    });
  }

  async function switchRun(runKey) {
    if (!state.session || !state.session.runs[runKey]) return;
    const run = state.session.runs[runKey];
    const prevModelKey = state.activeModelKey;

    await ensureModelLoaded(run.model_key);
    state.runKey = runKey;
    state.activeModelKey = run.model_key;

    Object.keys(loadedModels).forEach((mk) => {
      loadedModels[mk].rootGroup.visible = mk === run.model_key;
    });

    const mSpec = loadedModels[run.model_key].spec;
    grid.position.z = mSpec.floor_z;
    activePresets = mSpec.camera_presets || activePresets;
    compilerSummaryEl.innerHTML = mSpec.compiler_note || "";
    if (prevModelKey !== run.model_key) {
      setCameraPreset(state.activeCamPreset || "hero");
    }

    document.querySelectorAll(".run-tab").forEach((btn) => {
      btn.classList.toggle("active", btn.dataset.run === runKey);
    });

    annG1.textContent = run.g1_instruction || state.session.g1_instruction;
    if (run.camera_labels) {
      lblCamOverview.textContent = run.camera_labels.overview || "OVERVIEW CAM";
      lblCamWrist.textContent = run.camera_labels.wrist || "WRIST CAM";
    }

    slider.max = String(run.frame_count - 1);
    vidOverview.src = run.videos.overview;
    vidWrist.src = run.videos.wrist;
    vidOverview.load();
    vidWrist.load();

    const passed = Boolean(run.audit && run.audit.passed);
    verdictPill.className = `verdict-pill ${passed ? "pass" : "fail"}`;
    verdictPill.textContent =
      (run.audit && run.audit.badge_text) || (passed ? "AUDIT PASS" : "ANOMALY / BLOCKED");

    verdictDetail.textContent = (run.audit && run.audit.verdict) || "";
    verdictDetail.style.borderLeftColor = passed ? "#34d399" : "#ef4444";

    buildPhaseRibbon(run);
    applyLayerVisibility();
    applyFrame(prevModelKey === run.model_key ? Math.min(state.frame, run.frame_count - 1) : 0);
  }

  function updatePlayButton() {
    btnPlay.textContent = state.playing ? "⏸ PAUSE" : "▶ PLAY";
  }

  function wireChartScrubbing(canvasEl) {
    let dragging = false;
    const scrub = (ev) => {
      if (!state.session || !state.runKey) return;
      const rect = canvasEl.getBoundingClientRect();
      const padL = 36 * (rect.width / canvasEl.width);
      const padR = 10 * (rect.width / canvasEl.width);
      const frac = Math.max(0, Math.min(1, (ev.clientX - rect.left - padL) / Math.max(1, rect.width - padL - padR)));
      const run = state.session.runs[state.runKey];
      state.playing = false;
      updatePlayButton();
      applyFrame(Math.round(frac * (run.frame_count - 1)));
    };
    canvasEl.addEventListener("pointerdown", (ev) => {
      dragging = true;
      canvasEl.setPointerCapture(ev.pointerId);
      scrub(ev);
    });
    canvasEl.addEventListener("pointermove", (ev) => {
      if (dragging) scrub(ev);
    });
    canvasEl.addEventListener("pointerup", () => {
      dragging = false;
    });
  }

  function resizeViewport() {
    const w = container.clientWidth || 800;
    const h = container.clientHeight || 600;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
  }
  window.addEventListener("resize", resizeViewport);

  document.querySelectorAll(".seg-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      state.layerMode = btn.dataset.layer;
      document.querySelectorAll(".seg-btn").forEach((b) => b.classList.toggle("active", b === btn));
      applyLayerVisibility();
    });
  });

  document.querySelectorAll(".cam-btn").forEach((btn) => {
    btn.addEventListener("click", () => setCameraPreset(btn.dataset.cam));
  });

  document.getElementById("chk-forces").addEventListener("change", (e) => {
    state.showForces = e.target.checked;
    applyLayerVisibility();
  });

  document.getElementById("chk-wireframe").addEventListener("change", (e) => {
    state.showWireframe = e.target.checked;
    applyLayerVisibility();
  });

  btnPlay.addEventListener("click", () => {
    state.playing = !state.playing;
    updatePlayButton();
  });
  btnPrev.addEventListener("click", () => {
    state.playing = false;
    updatePlayButton();
    applyFrame(state.frame - 1);
  });
  btnNext.addEventListener("click", () => {
    state.playing = false;
    updatePlayButton();
    applyFrame(state.frame + 1);
  });
  selSpeed.addEventListener("change", (e) => {
    state.speed = parseFloat(e.target.value) || 1.0;
  });
  slider.addEventListener("input", (e) => {
    state.playing = false;
    updatePlayButton();
    applyFrame(parseInt(e.target.value, 10) || 0);
  });

  window.addEventListener("keydown", (e) => {
    if (e.code === "Space") {
      e.preventDefault();
      state.playing = !state.playing;
      updatePlayButton();
    } else if (e.code === "ArrowLeft") {
      state.playing = false;
      updatePlayButton();
      applyFrame(state.frame - 1);
    } else if (e.code === "ArrowRight") {
      state.playing = false;
      updatePlayButton();
      applyFrame(state.frame + 1);
    } else if (e.key === "1") {
      document.querySelector('.seg-btn[data-layer="visual"]').click();
    } else if (e.key === "2") {
      document.querySelector('.seg-btn[data-layer="ghost"]').click();
    } else if (e.key === "3") {
      document.querySelector('.seg-btn[data-layer="hulls"]').click();
    } else if (e.key.toLowerCase() === "f") {
      const chk = document.getElementById("chk-forces");
      chk.checked = !chk.checked;
      chk.dispatchEvent(new Event("change"));
    }
  });

  wireChartScrubbing(chartForces);
  wireChartScrubbing(chartSeating);

  async function init() {
    resizeViewport();
    const resp = await fetch("session_data.json");
    const session = await resp.json();
    state.session = session;

    document.getElementById("session-title").textContent = session.title;
    const runKeys = session.run_order || Object.keys(session.runs);

    runSwitcherEl.innerHTML = "";
    runKeys.forEach((rk) => {
      const r = session.runs[rk];
      const btn = document.createElement("button");
      btn.type = "button";
      btn.id = `btn-run-${rk}`;
      btn.className = "run-tab";
      btn.dataset.run = rk;
      const passed = Boolean(r.audit && r.audit.passed);
      btn.innerHTML = `<span class="dot ${passed ? "pass" : "fail"}"></span><span>${r.label}</span>`;
      btn.addEventListener("click", () => switchRun(rk));
      runSwitcherEl.appendChild(btn);
    });

    const urlParams = new URLSearchParams(window.location.search);
    const requestedRun = urlParams.get("run");
    const initialRun = requestedRun && session.runs[requestedRun] ? requestedRun : runKeys[0];
    await switchRun(initialRun);
    setCameraPreset("hero");
    updatePlayButton();
    state.ready = true;
    state.lastTickMs = performance.now();
  }

  function animate(nowMs) {
    requestAnimationFrame(animate);
    const dt = Math.min(0.1, (nowMs - state.lastTickMs) / 1000.0);
    state.lastTickMs = nowMs;

    if (state.ready && state.playing && state.session && state.runKey) {
      const run = state.session.runs[state.runKey];
      state.accumSec += dt * state.speed;
      const frameDur = 1.0 / run.fps;
      if (state.accumSec >= frameDur) {
        const stepFrames = Math.floor(state.accumSec / frameDur);
        state.accumSec -= stepFrames * frameDur;
        const nextFrame = (state.frame + stepFrames) % run.frame_count;
        applyFrame(nextFrame);
      }
    }

    controls.update();
    renderer.render(scene, camera);
  }

  init().catch((err) => {
    console.error("Failed to initialize Open-Argon Session Viewer:", err);
    verdictPill.className = "verdict-pill fail";
    verdictPill.textContent = "SESSION LOAD ERROR";
  });
  requestAnimationFrame(animate);
})();
