// Food Access Explorer (static Leaflet app) with data-driven color breaks

let map, manifest, currentLayer, boundaryLayer, usdaLayer, layerControl;
let state = { metro: null, geo: null, mode: null, minutes: null };

fetch("data/manifest.json")
  .then(r => r.json())
  .then(m => { manifest = m; initUI(); });

function initUI() {
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
    map = L.map("map").setView(meta.center, meta.zoom);
    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png").addTo(map);
    layerControl = L.control.layers(null, {}).addTo(map);
  } else {
    map.setView(meta.center, meta.zoom);
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
  el.value = values[0];
  cb(values[0]);
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
      boundaryLayer = L.geoJSON(d, { style: { color: "#000", weight: 2, fillOpacity: 0 } }).addTo(map);
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

function loadLayer() {
  if (!state.metro || !state.geo || !state.mode || !state.minutes) return;

  if (currentLayer) {
    map.removeLayer(currentLayer);
    layerControl.removeLayer(currentLayer);
    currentLayer = null;
  }

  const meta = manifest.metros[state.metro];
  const breaks = getBreaks(meta);
  const path = `data/${state.metro}/${state.geo}_${state.mode}_${state.minutes}.geojson`;

  fetch(path)
    .then(r => r.json())
    .then(data => {
      currentLayer = L.geoJSON(data, {
        style: f => {
          const x = Number(f.properties.coverage_pct);
          return {
            fillColor: color(x, breaks),
            fillOpacity: 0.7,
            color: "#333",
            weight: 0.4
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

function color(x, breaks) {
  if (isNaN(x)) return "#000";
  if (breaks) {
    const [t1, t2] = breaks;
    if (x > t2) return "#1a9850";
    if (x > t1) return "#fee08b";
    return "#d73027";
  }
  if (x > 0.75) return "#1a9850";
  if (x > 0.4) return "#fee08b";
  return "#d73027";
}

document.getElementById("downloadBtn").onclick = () => {
  window.open(`data/${state.metro}/${state.geo}_summary.csv`);
};
