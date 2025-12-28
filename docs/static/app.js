// Food Access Explorer (static Leaflet app) with data-driven color breaks

let map, manifest, currentLayer, boundaryLayer, usdaLayer, layerControl;
let state = {
  metro: null,
  geo: null,
  mode: null,
  minutes: null,
  fillOpacity: 0.7,
  strokeOpacity: 0.6
};

fetch("data/manifest.json")
  .then(r => r.json())
  .then(m => { manifest = m; initUI(); });

function initUI() {
  const fillOpacity = document.getElementById("fillOpacity");
  const strokeOpacity = document.getElementById("strokeOpacity");

  fillOpacity.oninput = e => {
    state.fillOpacity = Number(e.target.value);
    if (currentLayer) currentLayer.setStyle({ fillOpacity: state.fillOpacity });
  };
  strokeOpacity.oninput = e => {
    state.strokeOpacity = Number(e.target.value);
    if (currentLayer) currentLayer.setStyle({ opacity: state.strokeOpacity });
  };

  const metroSelect = document.getElementById("metroSelect");
  const metros = manifest.metros || {};
  Object.entries(metros).forEach(([slug, meta]) => {
    const opt = document.createElement("option");
    opt.value = slug;
    opt.textContent = meta.name;
    metroSelect.appendChild(opt);
  });
  metroSelect.onchange = e => loadMetro(e.target.value);

  const first = Object.keys(metros)[0];
  if (!first) {
    console.warn("No metros found in manifest.json");
    return;
  }
  metroSelect.value = first;
  loadMetro(first);
}

function loadMetro(slug) {
  state.metro = slug;
  const meta = manifest.metros[slug];

  if (!map) {
    map = L.map("map").setView(meta.center, meta.zoom + 1);
    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png").addTo(map);
    layerControl = L.control.layers(null, {}).addTo(map);
  } else {
    map.setView(meta.center, meta.zoom + 1);
  }

  populate("geoSelect", meta.geographies, v => { state.geo = v; loadLayer(); });
  populate("modeSelect", meta.modes, v => { state.mode = v; loadLayer(); });
  populate("timeSelect", meta.minutes, v => { state.minutes = v; loadLayer(); });

  loadBoundary(meta);
  loadUSDA(meta);
}

function populate(id, values, cb) {
  const el = document.getElementById(id);
  el.innerHTML = "";
  values.forEach(v => {
    const o = document.createElement("option");
    o.value = v; o.textContent = v;
    el.appendChild(o);
  });
  el.onchange = e => cb(e.target.value);
  let initial = values[0];
  if (id === "geoSelect" && values.includes("metro_grid")) {
    initial = "metro_grid";
  }
  el.value = initial;
  cb(initial);
}

function loadBoundary(meta) {
  if (boundaryLayer) {
    map.removeLayer(boundaryLayer);
    layerControl.removeLayer(boundaryLayer);
    boundaryLayer = null;
  }
  if (!meta.has_boundary) return;

  fetch(`data/${state.metro}/metro_boundary.geojson`)
    .then(r => r.json())
    .then(d => {
      boundaryLayer = L.geoJSON(d, {
        style: { color: "#000", weight: 2, opacity: 1, fillOpacity: 0 }
      }).addTo(map);
      layerControl.addOverlay(boundaryLayer, "Metro boundary");
    });
}

function loadUSDA(meta) {
  if (usdaLayer) {
    map.removeLayer(usdaLayer);
    layerControl.removeLayer(usdaLayer);
    usdaLayer = null;
  }
  if (!meta.has_usda) return;

  fetch(`data/${state.metro}/usda_fara_tracts.geojson`)
    .then(r => r.json())
    .then(d => {
      usdaLayer = L.geoJSON(d, { style: { color: "#555", weight: 1, fillOpacity: 0 } });
      layerControl.addOverlay(usdaLayer, "USDA Food Access (FARA)");
    });
}

function getBreaks(meta) {
  try {
    const t = meta.breaks[state.geo][state.mode][String(state.minutes)];
    if (!t || t.length !== 2) return null;
    return [Number(t[0]), Number(t[1])];
  } catch (e) {
    return null;
  }
}

function layerRange(data) {
  const vals = [];
  data.features.forEach(f => {
    const v = Number(f.properties.coverage_pct);
    if (!isNaN(v)) vals.push(v);
  });
  if (!vals.length) return { min: 0, max: 1 };
  return { min: Math.min(...vals), max: Math.max(...vals) };
}

function loadLayer() {
  if (!state.metro || !state.geo || !state.mode || !state.minutes) return;

  if (currentLayer) {
    map.removeLayer(currentLayer);
    layerControl.removeLayer(currentLayer);
    currentLayer = null;
  }

  const meta = manifest.metros[state.metro];
  const path = `data/${state.metro}/${state.geo}_${state.mode}_${state.minutes}.geojson`;

  fetch(path)
    .then(r => r.json())
    .then(data => {
      const range = layerRange(data);
      currentLayer = L.geoJSON(data, {
        style: f => {
          const x = Number(f.properties.coverage_pct);
          return {
            fillColor: color(x, range),
            fillOpacity: state.fillOpacity,
            color: "#333",
            weight: 0.4,
            opacity: state.strokeOpacity
          };
        },
        onEachFeature: (f, l) => {
          l.bindPopup(
            `<b>ID:</b> ${f.properties.GEOID}<br>
             <b>Population:</b> ${Math.round(f.properties.POPULATION)}<br>
             <b>Access:</b> ${(Number(f.properties.coverage_pct)*100).toFixed(1)}%`
          );
        }
      }).addTo(map);
      layerControl.addOverlay(currentLayer, "Food access layer");
    });
}

function color(x, range) {
  if (isNaN(x)) return "#000";
  const min = range?.min ?? 0;
  const max = range?.max ?? 1;
  const denom = max - min;
  const t = denom > 0 ? (x - min) / denom : 0;
  const pct = Math.max(0, Math.min(1, t));
  return lerpColor("#b10026", "#006837", pct);
}

function lerpColor(a, b, t) {
  const pa = parseInt(a.slice(1), 16);
  const pb = parseInt(b.slice(1), 16);
  const ar = (pa >> 16) & 255, ag = (pa >> 8) & 255, ab = pa & 255;
  const br = (pb >> 16) & 255, bg = (pb >> 8) & 255, bb = pb & 255;
  const rr = Math.round(ar + (br - ar) * t);
  const rg = Math.round(ag + (bg - ag) * t);
  const rb = Math.round(ab + (bb - ab) * t);
  return `rgb(${rr},${rg},${rb})`;
}

document.getElementById("downloadBtn").onclick = () => {
  window.open(`data/${state.metro}/${state.geo}_summary.csv`);
};
