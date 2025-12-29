// Food Access Explorer (static Leaflet app) with data-driven color breaks

let map, manifest, currentLayer, boundaryLayer, usdaLayer, layerControl, groceryLayer, stationLayer, busLayer;
let state = {
  metro: null,
  geo: null,
  mode: null,
  minutes: null,
  timeMode: "fixed",
  fillOpacity: 0.7,
  strokeOpacity: 0.6,
  range: { min: 0, max: 1 },
  metric: "coverage_pct",
  showGroceries: false,
  showStations: false,
  showBusStops: false,
  colorStops: {
    low: 0,
    mid: 0.5,
    high: 1,
    colorLow: "#b10026",
    colorMid: "#ffffbf",
    colorHigh: "#006837"
  }
};

fetch("data/manifest.json")
  .then(r => r.json())
  .then(m => { manifest = m; initUI(); });

function initUI() {
  const fillOpacity = document.getElementById("fillOpacity");
  const strokeOpacity = document.getElementById("strokeOpacity");
  const colorLow = document.getElementById("colorLow");
  const colorMid = document.getElementById("colorMid");
  const colorHigh = document.getElementById("colorHigh");
  const stopLow = document.getElementById("stopLow");
  const stopMid = document.getElementById("stopMid");
  const stopHigh = document.getElementById("stopHigh");
  const colorLowValue = document.getElementById("colorLowValue");
  const colorMidValue = document.getElementById("colorMidValue");
  const colorHighValue = document.getElementById("colorHighValue");
  const metricSelect = document.getElementById("metricSelect");
  const timeModeSelect = document.getElementById("timeModeSelect");
  const groceryToggle = document.getElementById("groceryToggle");
  const stationToggle = document.getElementById("stationToggle");
  const busToggle = document.getElementById("busToggle");

  fillOpacity.oninput = e => {
    state.fillOpacity = Number(e.target.value);
    if (currentLayer) currentLayer.setStyle({ fillOpacity: state.fillOpacity });
  };
  strokeOpacity.oninput = e => {
    state.strokeOpacity = Number(e.target.value);
    if (currentLayer) currentLayer.setStyle({ opacity: state.strokeOpacity });
  };

  function normalizeStops() {
    let low = Number(stopLow.value) / 100;
    let mid = Number(stopMid.value) / 100;
    let high = Number(stopHigh.value) / 100;
    low = Math.max(0, Math.min(0.98, low));
    high = Math.max(0.02, Math.min(1, high));
    if (high <= low) high = Math.min(1, low + 0.02);
    mid = Math.max(low + 0.01, Math.min(high - 0.01, mid));
    stopLow.value = Math.round(low * 100);
    stopMid.value = Math.round(mid * 100);
    stopHigh.value = Math.round(high * 100);
    state.colorStops.low = low;
    state.colorStops.mid = mid;
    state.colorStops.high = high;
  }

  function applyColorUpdates() {
    normalizeStops();
    state.colorStops.colorLow = colorLow.value;
    state.colorStops.colorMid = colorMid.value;
    state.colorStops.colorHigh = colorHigh.value;
    updateLayerStyle();
  }

  [colorLow, colorMid, colorHigh, stopLow, stopMid, stopHigh].forEach(el => {
    el.oninput = applyColorUpdates;
  });

  metricSelect.onchange = e => {
    state.metric = e.target.value;
    updateLayerStyle();
  };
  groceryToggle.onchange = e => {
    state.showGroceries = e.target.checked;
    togglePOILayers();
  };
  stationToggle.onchange = e => {
    state.showStations = e.target.checked;
    togglePOILayers();
  };
  busToggle.onchange = e => {
    state.showBusStops = e.target.checked;
    togglePOILayers();
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

  timeModeSelect.innerHTML = "";
  [
    { value: "fixed", label: "Fixed minutes" },
    { value: "min_access", label: "Minimum time to access" }
  ].forEach(opt => {
    const o = document.createElement("option");
    o.value = opt.value;
    o.textContent = opt.label;
    timeModeSelect.appendChild(o);
  });
  timeModeSelect.onchange = e => {
    state.timeMode = e.target.value;
    updateMetricOptions();
    toggleTimeSelect();
    updateLayerStyle();
    loadLayer();
  };
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
  toggleTimeSelect();
  updateMetricOptions();

  loadBoundary(meta);
  loadUSDA(meta);
  togglePOILayers();
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

function toggleTimeSelect() {
  const timeSelect = document.getElementById("timeSelect");
  const show = state.timeMode === "fixed";
  timeSelect.style.display = show ? "block" : "none";
}

function updateMetricOptions() {
  const metricSelect = document.getElementById("metricSelect");
  if (state.timeMode === "min_access") {
    metricSelect.innerHTML = "";
    const o = document.createElement("option");
    o.value = "min_minutes";
    o.textContent = "Minimum minutes";
    metricSelect.appendChild(o);
    state.metric = "min_minutes";
    metricSelect.value = state.metric;
    return;
  }
  const base = [
    { value: "coverage_pct", label: "Coverage %" },
    { value: "pop_with_access", label: "Population w/ access" },
    { value: "pop_adjusted_coverage", label: "Pop-adjusted coverage" }
  ];
  const metroGridExtras = [
    { value: "non_transit_commuter_share", label: "Non-transit commuter share" },
    { value: "vehicle_share", label: "Vehicle share" },
    { value: "MEDIAN_RENT_PER_ROOM", label: "Median housing cost per room" },
    { value: "coverage_non_transit_adjusted", label: "Coverage × non-transit share" },
    { value: "coverage_vehicle_adjusted", label: "Coverage × vehicle share" },
    { value: "coverage_rent_adjusted", label: "Coverage × rent per room (norm)" }
  ];
  const options = state.geo === "metro_grid" ? base.concat(metroGridExtras) : base;

  metricSelect.innerHTML = "";
  options.forEach(opt => {
    const o = document.createElement("option");
    o.value = opt.value;
    o.textContent = opt.label;
    metricSelect.appendChild(o);
  });

  if (!options.find(o => o.value === state.metric)) {
    state.metric = "coverage_pct";
  }
  metricSelect.value = state.metric;
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

function togglePOILayers() {
  if (!map) return;
  if (!state.showGroceries && groceryLayer) {
    map.removeLayer(groceryLayer);
    layerControl.removeLayer(groceryLayer);
    groceryLayer = null;
  }
  if (!state.showStations && stationLayer) {
    map.removeLayer(stationLayer);
    layerControl.removeLayer(stationLayer);
    stationLayer = null;
  }
  if (!state.showBusStops && busLayer) {
    map.removeLayer(busLayer);
    layerControl.removeLayer(busLayer);
    busLayer = null;
  }

  if (state.showGroceries && !groceryLayer) {
    fetch(`data/${state.metro}/groceries.geojson`)
      .then(r => r.json())
      .then(d => {
        const groceryIcon = L.divIcon({
          className: "grocery-icon",
          html: "🏪",
          iconSize: [18, 18],
          iconAnchor: [9, 9]
        });
        groceryLayer = L.geoJSON(d, {
          pointToLayer: (f, latlng) => L.marker(latlng, { icon: groceryIcon })
        }).addTo(map);
        layerControl.addOverlay(groceryLayer, "Groceries");
      });
  }
  if ((state.showStations || state.showBusStops) && (!stationLayer || !busLayer)) {
    fetch(`data/${state.metro}/transit.geojson`)
      .then(r => r.json())
      .then(d => {
        const trainIcon = L.divIcon({
          className: "train-icon",
          html: "🚆",
          iconSize: [18, 18],
          iconAnchor: [9, 9]
        });
        if (state.showStations && !stationLayer) {
          stationLayer = L.geoJSON(d, {
            filter: f => f?.properties?.railway === "station" || f?.properties?.public_transport === "station",
            pointToLayer: (f, latlng) => L.marker(latlng, { icon: trainIcon })
          }).addTo(map);
          layerControl.addOverlay(stationLayer, "Stations");
        }
        if (state.showBusStops && !busLayer) {
          busLayer = L.geoJSON(d, {
            filter: f => f?.properties?.highway === "bus_stop",
            pointToLayer: (f, latlng) => L.circleMarker(latlng, {
              radius: 2.5,
              color: "#000",
              fillColor: "#000",
              fillOpacity: 0.9,
              weight: 1
            })
          }).addTo(map);
          layerControl.addOverlay(busLayer, "Bus stops");
        }
      });
  }
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

function layerRange(data, metric) {
  const vals = [];
  data.features.forEach(f => {
    const v = Number(f.properties[metric]);
    if (!isNaN(v)) vals.push(v);
  });
  if (!vals.length) return { min: 0, max: 1, median: 0 };
  vals.sort((a, b) => a - b);
  const mid = Math.floor(vals.length / 2);
  const median = vals.length % 2 ? vals[mid] : (vals[mid - 1] + vals[mid]) / 2;
  const min = metric.startsWith("min_") ? 0 : vals[0];
  return { min, max: vals[vals.length - 1], median };
}

function loadLayer() {
  if (!state.metro || !state.geo || !state.mode) return;
  if (state.timeMode === "fixed" && !state.minutes) return;
  updateMetricOptions();

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
      computeDerivedMetrics(data);
      const range = layerRange(data, metricKey());
      state.range = range;
      seedMidStopFromMedian();
      updateColorRangeLabels();
      currentLayer = L.geoJSON(data, {
        style: styleForFeature,
        onEachFeature: (f, l) => {
          l.bindPopup(
            `<b>ID:</b> ${f.properties.GEOID}<br>
             <b>Population:</b> ${Math.round(f.properties.POPULATION)}<br>
             <b>Access:</b> ${(Number(f.properties.coverage_pct)*100).toFixed(1)}%<br>
             <b>Metric:</b> ${formatMetric(f.properties, metricKey())}`
          );
        }
      }).addTo(map);
      layerControl.addOverlay(currentLayer, "Food access layer");
    });
}

function updateLayerStyle() {
  if (!currentLayer) return;
  state.range = layerRange(currentLayer.toGeoJSON(), metricKey());
  seedMidStopFromMedian();
  updateColorRangeLabels();
  currentLayer.setStyle(styleForFeature);
}

function styleForFeature(f) {
  const x = Number(f?.properties?.[metricKey()]);
  return {
    fillColor: color(x, state.range),
    fillOpacity: state.fillOpacity,
    color: "#333",
    weight: 0.4,
    opacity: state.strokeOpacity
  };
}

function updateColorRangeLabels() {
  const min = state.range?.min ?? 0;
  const max = state.range?.max ?? 1;
  const lowStop = state.colorStops.low ?? 0;
  const midStop = state.colorStops.mid ?? 0.5;
  const highStop = state.colorStops.high ?? 1;
  const lowVal = min + (max - min) * lowStop;
  const midVal = min + (max - min) * midStop;
  const highVal = min + (max - min) * highStop;
  const lowEl = document.getElementById("colorLowValue");
  const midEl = document.getElementById("colorMidValue");
  const highEl = document.getElementById("colorHighValue");
  const lowText = `Low: ${formatMetricValue(lowVal, metricKey())}`;
  const highText = `High: ${formatMetricValue(highVal, metricKey())}`;
  if (lowEl) lowEl.textContent = lowText;
  if (midEl) midEl.textContent = `Mid: ${formatMetricValue(midVal, metricKey())}`;
  if (highEl) highEl.textContent = highText;
}

function seedMidStopFromMedian() {
  const min = state.range?.min ?? 0;
  const max = state.range?.max ?? 1;
  const median = state.range?.median;
  if (median === undefined || max <= min) return;
  const t = (median - min) / (max - min);
  const mid = Math.max(0, Math.min(1, t));
  state.colorStops.mid = mid;
  const stopMid = document.getElementById("stopMid");
  if (stopMid) stopMid.value = String(Math.round(mid * 100));
}

function color(x, range) {
  if (isNaN(x)) return "#000";
  const min = range?.min ?? 0;
  const max = range?.max ?? 1;
  const denom = max - min;
  const t = denom > 0 ? (x - min) / denom : 0;
  const pct = Math.max(0, Math.min(1, t));
  return lerp3(
    state.colorStops.colorLow,
    state.colorStops.colorMid,
    state.colorStops.colorHigh,
    pct,
    state.colorStops.low,
    state.colorStops.mid,
    state.colorStops.high
  );
}

function metricKey() {
  if (state.timeMode === "min_access") {
    if (state.mode === "walk") return "min_grocery_minutes";
    return state.geo === "metro_grid" ? "min_gtfs_minutes" : "min_transit_minutes";
  }
  return state.metric;
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

function computeDerivedMetrics(data) {
  let maxPopAccess = 0;
  let maxRent = 0;
  data.features.forEach(f => {
    const v = Number(f.properties.pop_with_access);
    if (!isNaN(v)) maxPopAccess = Math.max(maxPopAccess, v);
    const r = Number(f.properties.MEDIAN_RENT_PER_ROOM);
    if (!isNaN(r)) maxRent = Math.max(maxRent, r);
  });
  data.features.forEach(f => {
    const v = Number(f.properties.pop_with_access);
    f.properties.pop_adjusted_coverage = maxPopAccess > 0 ? (v / maxPopAccess) : 0;
    const cov = Number(f.properties.coverage_pct);
    const transit = Number(f.properties.transit_commuter_share);
    const zeroVeh = Number(f.properties.zero_vehicle_share);
    const rent = Number(f.properties.MEDIAN_RENT_PER_ROOM);
    const rentNorm = maxRent > 0 ? (rent / maxRent) : 0;
    const vehicleShare = isNaN(zeroVeh) ? 0 : (1 - zeroVeh);
    const nonTransitShare = isNaN(transit) ? 0 : (1 - transit);
    f.properties.vehicle_share = vehicleShare;
    f.properties.non_transit_commuter_share = nonTransitShare;
    f.properties.coverage_vehicle_adjusted = isNaN(cov) ? 0 : cov * vehicleShare;
    f.properties.coverage_non_transit_adjusted = isNaN(cov) ? 0 : cov * nonTransitShare;
    f.properties.coverage_rent_adjusted = isNaN(cov) || isNaN(rentNorm) ? 0 : cov * rentNorm;
  });
}

function formatMetric(props, metric) {
  const v = Number(props?.[metric]);
  return formatMetricValue(v, metric);
}

function formatMetricValue(v, metric) {
  if (isNaN(v)) return "n/a";
  if (
    metric.endsWith("_share") ||
    metric === "coverage_pct" ||
    metric === "pop_adjusted_coverage" ||
    metric.startsWith("coverage_")
  ) {
    return `${(v * 100).toFixed(1)}%`;
  }
  if (metric === "MEDIAN_RENT_PER_ROOM") {
    return `$${Math.round(v)}`;
  }
  if (metric.startsWith("min_")) {
    return `${Math.round(v)} min`;
  }
  return Math.round(v).toLocaleString();
}

function lerp3(a, b, c, t, t0, t1, t2) {
  if (t <= t0) return a;
  if (t >= t2) return c;
  if (t <= t1) {
    const local = (t - t0) / Math.max(1e-6, (t1 - t0));
    return lerpColor(a, b, local);
  }
  const local = (t - t1) / Math.max(1e-6, (t2 - t1));
  return lerpColor(b, c, local);
}

document.getElementById("downloadBtn").onclick = () => {
  window.open(`data/${state.metro}/${state.geo}_summary.csv`);
};
