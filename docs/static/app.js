// Food Access Explorer (static Leaflet app) with data-driven color breaks

let map, manifest, currentLayer, boundaryLayer, usdaLayer, layerControl, groceryLayer, stationLayer, busLayer, foodBankLayer;
const FOOD_BANK_MAX = 10;
let state = {
  metro: null,
  geo: null,
  mode: null,
  minutes: null,
  timeMode: "fixed",
  fillOpacity: 0.7,
  strokeOpacity: 0.6,
  range: { min: 0, max: 1 },
  opacityRange: { min: 0, max: 1 },
  metric: "coverage_pct",
  opacityMetric: "none",
  minAccessKey: "min_transit_minutes",
  foodBankCount: 0,
  currentData: null,
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
  const opacityMetricSelect = document.getElementById("opacityMetricSelect");
  const foodBankCount = document.getElementById("foodBankCount");
  const foodBankValue = document.getElementById("foodBankValue");
  const timeModeSelect = document.getElementById("timeModeSelect");
  const groceryToggle = document.getElementById("groceryToggle");
  const stationToggle = document.getElementById("stationToggle");
  const busToggle = document.getElementById("busToggle");

  fillOpacity.oninput = e => {
    state.fillOpacity = Number(e.target.value);
    updateLayerStyle();
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
  opacityMetricSelect.onchange = e => {
    state.opacityMetric = e.target.value;
    updateLayerStyle();
  };
  if (foodBankCount) {
    const updateFoodBankLabel = () => {
      if (foodBankValue) foodBankValue.textContent = String(state.foodBankCount);
    };
    foodBankCount.oninput = e => {
      state.foodBankCount = Number(e.target.value);
      updateFoodBankLabel();
      updateLayerStyle();
    };
    updateFoodBankLabel();
  }
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
    map.on("moveend zoomend", () => {
      if (state.currentData) updateDistributions(state.currentData);
    });
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
  const base = [
    { value: "coverage_pct", label: "Coverage %" },
    { value: "POPULATION", label: "Population" },
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

  if (state.timeMode === "min_access") {
    metricSelect.innerHTML = "";
    const o = document.createElement("option");
    o.value = "min_minutes";
    o.textContent = "Minimum minutes";
    metricSelect.appendChild(o);
    state.metric = "min_minutes";
    metricSelect.value = state.metric;
  } else {
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

  if (opacityMetricSelect) {
    opacityMetricSelect.innerHTML = "";
    const on = document.createElement("option");
    on.value = "none";
    on.textContent = "None (uniform)";
    opacityMetricSelect.appendChild(on);
    options.forEach(opt => {
      const o = document.createElement("option");
      o.value = opt.value;
      o.textContent = opt.label;
      opacityMetricSelect.appendChild(o);
    });
    if (!["none", ...options.map(o => o.value)].includes(state.opacityMetric)) {
      state.opacityMetric = "none";
    }
    opacityMetricSelect.value = state.opacityMetric;
  }
  if (state.timeMode === "min_access") {
    return;
  }

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
    const v = parseMetricValue(f.properties[metric]);
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
      state.currentData = data;
      computeDerivedMetrics(data);
      updateMinAccessKey(data);
      syncOpacityMetric();
      const range = layerRange(data, metricKey());
      state.range = range;
      state.opacityRange = layerRangeOpacity(data, state.opacityMetric);
      seedMidStopFromMedian();
      updateColorRangeLabels();
      updateDistributions(data);
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
      updateFoodBankLayer(data);
    });
}

function updateLayerStyle() {
  if (!currentLayer) return;
  state.currentData = currentLayer.toGeoJSON();
  updateMinAccessKey(state.currentData);
  syncOpacityMetric();
  state.range = layerRange(state.currentData, metricKey());
  state.opacityRange = layerRangeOpacity(state.currentData, state.opacityMetric);
  updateColorRangeLabels();
  currentLayer.setStyle(styleForFeature);
  updateFoodBankLayer(state.currentData);
  updateDistributions(state.currentData);
}

function styleForFeature(f) {
  const x = Number(f?.properties?.[metricKey()]);
  const opacity = featureOpacity(f);
  return {
    fillColor: color(x, state.range),
    fillOpacity: opacity,
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

function featureOpacity(f) {
  if (!state.opacityMetric || state.opacityMetric === "none") {
    return state.fillOpacity;
  }
  const v = parseMetricValue(f?.properties?.[state.opacityMetric]);
  if (isNaN(v)) return 0;
  const min = state.opacityRange?.min ?? 0;
  const max = state.opacityRange?.max ?? 1;
  const denom = max - min;
  if (denom <= 0) return state.fillOpacity;
  const t = (v - min) / denom;
  const pct = Math.max(0, Math.min(1, t));
  return state.fillOpacity * pct;
}

function parseMetricValue(val) {
  if (val === null || val === undefined) return NaN;
  if (typeof val === "number") return val;
  if (typeof val === "string") {
    const cleaned = val.replace(/,/g, "");
    const num = Number(cleaned);
    return Number.isFinite(num) ? num : NaN;
  }
  const num = Number(val);
  return Number.isFinite(num) ? num : NaN;
}

function updateDistributions(data) {
  const metricSvgX = document.getElementById("metricDistX");
  const metricSvgY = document.getElementById("metricDistY");
  if (!metricSvgX || !metricSvgY || !map || !data) return;
  const bounds = map.getBounds();
  const binsX = axisBins("x");
  const binsY = axisBins("y");
  const countsX = buildAxisCounts(data, bounds, "x", binsX);
  const countsY = buildAxisCounts(data, bounds, "y", binsY);
  let opacityCountsX = [];
  let opacityCountsY = [];
  if (state.opacityMetric && state.opacityMetric !== "none") {
    opacityCountsX = buildAxisCounts(data, bounds, "x", binsX, opacityWeightForFeature);
    opacityCountsY = buildAxisCounts(data, bounds, "y", binsY, opacityWeightForFeature);
  }
  const xSize = getSvgSize(metricSvgX, 300, 70);
  const ySize = getSvgSize(metricSvgY, 70, 300);
  renderAxisHistogram(metricSvgX, countsX, opacityCountsX, "x", xSize);
  renderAxisHistogram(metricSvgY, countsY, opacityCountsY, "y", ySize);
}

function axisBins(axis) {
  const zoom = map ? map.getZoom() : 0;
  const base = Math.round(6 + zoom * 2);
  const minBins = 8;
  const maxBins = axis === "x" ? 48 : 48;
  return Math.max(minBins, Math.min(maxBins, base));
}

function buildAxisCounts(data, bounds, axis, bins, weightFn) {
  const counts = new Array(bins).fill(0);
  const minX = bounds.getWest();
  const maxX = bounds.getEast();
  const minY = bounds.getSouth();
  const maxY = bounds.getNorth();
  const denomX = maxX - minX || 1;
  const denomY = maxY - minY || 1;
  data.features.forEach(f => {
    const center = featureCenterCoords(f);
    if (!center) return;
    const lng = center[0];
    const lat = center[1];
    if (lng < minX || lng > maxX || lat < minY || lat > maxY) return;
    const t = axis === "x"
      ? (lng - minX) / denomX
      : (lat - minY) / denomY;
    const idx = Math.max(0, Math.min(bins - 1, Math.floor(t * bins)));
    const w = weightFn ? weightFn(f) : 1;
    if (w > 0) counts[idx] += w;
  });
  return counts;
}

function featureCenterCoords(feature) {
  const geom = feature?.geometry;
  if (!geom) return null;
  if (geom.type === "Point" && Array.isArray(geom.coordinates)) {
    return geom.coordinates;
  }
  const bounds = {
    minX: Infinity,
    minY: Infinity,
    maxX: -Infinity,
    maxY: -Infinity
  };
  walkCoords(geom.coordinates, bounds);
  if (!isFinite(bounds.minX) || !isFinite(bounds.minY)) return null;
  return [(bounds.minX + bounds.maxX) / 2, (bounds.minY + bounds.maxY) / 2];
}

function walkCoords(coords, bounds) {
  if (!Array.isArray(coords)) return;
  if (typeof coords[0] === "number") {
    const x = coords[0];
    const y = coords[1];
    if (typeof x !== "number" || typeof y !== "number") return;
    bounds.minX = Math.min(bounds.minX, x);
    bounds.maxX = Math.max(bounds.maxX, x);
    bounds.minY = Math.min(bounds.minY, y);
    bounds.maxY = Math.max(bounds.maxY, y);
    return;
  }
  coords.forEach(c => walkCoords(c, bounds));
}

function getSvgSize(svgEl, fallbackW, fallbackH) {
  const rect = svgEl.getBoundingClientRect();
  const w = rect.width || fallbackW;
  const h = rect.height || fallbackH;
  return { w, h };
}

function renderAxisHistogram(svgEl, counts, overlayCounts, axis, size) {
  const w = size?.w ?? (axis === "x" ? 300 : 70);
  const h = size?.h ?? (axis === "x" ? 70 : 300);
  svgEl.setAttribute("viewBox", `0 0 ${w} ${h}`);
  while (svgEl.firstChild) svgEl.removeChild(svgEl.firstChild);
  if (!counts.length) return;
  const maxOverlay = overlayCounts && overlayCounts.length ? Math.max(...overlayCounts) : 0;
  const maxCount = Math.max(...counts, maxOverlay, 1);
  const addClickLabel = (el, text) => {
    el.addEventListener("click", evt => {
      evt.stopPropagation();
      showHistogramTooltip(text, evt);
    });
  };

  if (axis === "x") {
    const barW = w / counts.length;
    counts.forEach((c, i) => {
      const bh = (c / maxCount) * (h - 16);
      const x = i * barW;
      const y = h - bh;
      const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
      rect.setAttribute("x", x.toFixed(2));
      rect.setAttribute("y", y.toFixed(2));
      rect.setAttribute("width", Math.max(1, barW - 1).toFixed(2));
      rect.setAttribute("height", bh.toFixed(2));
      rect.setAttribute("fill", "#2c7fb8");
      rect.setAttribute("fill-opacity", "0.75");
      addClickLabel(rect, `bin ${i + 1}: ${c.toFixed(2)}`);
      svgEl.appendChild(rect);
    });
    if (overlayCounts && overlayCounts.length) {
      overlayCounts.forEach((c, i) => {
        const bh = (c / maxCount) * (h - 16);
        const x = i * barW + 2;
        const y = h - bh;
        const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
        rect.setAttribute("x", x.toFixed(2));
        rect.setAttribute("y", y.toFixed(2));
        rect.setAttribute("width", Math.max(1, barW - 5).toFixed(2));
        rect.setAttribute("height", bh.toFixed(2));
        rect.setAttribute("fill", "#f2c400");
        rect.setAttribute("fill-opacity", "0.6");
        addClickLabel(rect, `opacity bin ${i + 1}: ${c.toFixed(2)}`);
        svgEl.appendChild(rect);
      });
    }
    addAxisCountLabels(svgEl, w, h, maxCount, "x");
  } else {
    const barH = h / counts.length;
    counts.forEach((c, i) => {
      const bw = (c / maxCount) * (w - 16);
      const y = h - (i + 1) * barH;
      const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
      rect.setAttribute("x", "0");
      rect.setAttribute("y", y.toFixed(2));
      rect.setAttribute("width", bw.toFixed(2));
      rect.setAttribute("height", Math.max(1, barH - 1).toFixed(2));
      rect.setAttribute("fill", "#2c7fb8");
      rect.setAttribute("fill-opacity", "0.75");
      addClickLabel(rect, `bin ${i + 1}: ${c.toFixed(2)}`);
      svgEl.appendChild(rect);
    });
    if (overlayCounts && overlayCounts.length) {
      overlayCounts.forEach((c, i) => {
        const bw = (c / maxCount) * (w - 16);
        const y = h - (i + 1) * barH + 2;
        const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
        rect.setAttribute("x", "0");
        rect.setAttribute("y", y.toFixed(2));
        rect.setAttribute("width", Math.max(1, bw - 3).toFixed(2));
        rect.setAttribute("height", Math.max(1, barH - 5).toFixed(2));
        rect.setAttribute("fill", "#f2c400");
        rect.setAttribute("fill-opacity", "0.6");
        addClickLabel(rect, `opacity bin ${i + 1}: ${c.toFixed(2)}`);
        svgEl.appendChild(rect);
      });
    }
    addAxisCountLabels(svgEl, w, h, maxCount, "y");
  }
}

function opacityWeightForFeature(f) {
  if (!state.opacityMetric || state.opacityMetric === "none") return 0;
  const v = parseMetricValue(f?.properties?.[state.opacityMetric]);
  if (isNaN(v)) return 0;
  const min = state.opacityRange?.min ?? 0;
  const max = state.opacityRange?.max ?? 1;
  const denom = max - min;
  if (denom <= 0) return 0;
  const t = (v - min) / denom;
  return Math.max(0, Math.min(1, t));
}

function showHistogramTooltip(text, evt) {
  const mapWrap = document.getElementById("map-wrap");
  if (!mapWrap) return;
  let tip = document.getElementById("histogramTooltip");
  if (!tip) {
    tip = document.createElement("div");
    tip.id = "histogramTooltip";
    mapWrap.appendChild(tip);
  }
  tip.textContent = text;
  const rect = mapWrap.getBoundingClientRect();
  const x = evt.clientX - rect.left;
  const y = evt.clientY - rect.top;
  tip.style.left = `${x}px`;
  tip.style.top = `${y}px`;
  tip.style.display = "block";
  clearTimeout(tip._hideTimer);
  tip._hideTimer = setTimeout(() => {
    tip.style.display = "none";
  }, 1200);
}

function addAxisCountLabels(svgEl, w, h, maxCount, axis) {
  const makeText = (x, y, text, anchor = "start") => {
    const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
    label.setAttribute("x", String(x));
    label.setAttribute("y", String(y));
    label.setAttribute("font-size", "9");
    label.setAttribute("fill", "#555");
    label.setAttribute("text-anchor", anchor);
    label.textContent = text;
    svgEl.appendChild(label);
  };
  if (axis === "x") {
    makeText(2, 10, "0");
    makeText(2, 20, String(maxCount));
  } else {
    makeText(4, h - 4, "0");
    makeText(4, 10, String(maxCount));
  }
}

function syncOpacityMetric() {
  const opacityMetricSelect = document.getElementById("opacityMetricSelect");
  if (opacityMetricSelect && opacityMetricSelect.value) {
    state.opacityMetric = opacityMetricSelect.value;
  }
}

function layerRangeOpacity(data, metric) {
  if (!metric || metric === "none") return { min: 0, max: 1, median: 0 };
  const vals = [];
  data.features.forEach(f => {
    const v = parseMetricValue(f.properties[metric]);
    if (!isNaN(v)) vals.push(v);
  });
  if (!vals.length) return { min: 0, max: 1, median: 0 };
  vals.sort((a, b) => a - b);
  const mid = Math.floor(vals.length / 2);
  const median = vals.length % 2 ? vals[mid] : (vals[mid - 1] + vals[mid]) / 2;
  const min = vals[0];
  const p95 = vals[Math.floor(0.95 * (vals.length - 1))];
  const max = p95 > 0 ? p95 : vals[vals.length - 1];
  return { min, max, median };
}

function updateMinAccessKey(data) {
  if (state.timeMode !== "min_access" || state.mode !== "walk_transit") return;
  const props = data?.features?.[0]?.properties || {};
  state.minAccessKey = Object.prototype.hasOwnProperty.call(props, "min_access_minutes")
    ? "min_access_minutes"
    : "min_transit_minutes";
}

function foodBankMinKey() {
  if (state.foodBankCount <= 0 || state.geo !== "metro_grid") return null;
  const count = Math.min(state.foodBankCount, FOOD_BANK_MAX);
  if (state.mode === "walk") {
    return `min_grocery_minutes_fb_${count}`;
  }
  return `min_access_minutes_fb_${count}`;
}

function updateFoodBankLayer(data) {
  const layerData = data || state.currentData;
  if (foodBankLayer) {
    map.removeLayer(foodBankLayer);
    layerControl.removeLayer(foodBankLayer);
    foodBankLayer = null;
  }
  if (!layerData || !map || state.foodBankCount <= 0) return;
  const count = Math.min(state.foodBankCount, FOOD_BANK_MAX);
  const modeKey = state.mode === "walk" ? "walk" : "walk_transit";
  const fbPath = `data/${state.metro}/foodbanks_${modeKey}.geojson`;

  fetch(fbPath)
    .then(r => r.ok ? r.json() : null)
    .then(fbData => {
      if (!fbData || !fbData.features) return;
      const ranked = fbData.features
        .slice()
        .sort((a, b) => (a.properties?.rank ?? 999) - (b.properties?.rank ?? 999))
        .slice(0, count);
      if (!ranked.length) return;
      const markers = ranked.map(f => {
        const center = getFeatureCenter(f.geometry);
        if (!center) return null;
        return L.marker(center, {
          icon: L.divIcon({
            className: "foodbank-star",
            html: "★",
            iconSize: [16, 16],
            iconAnchor: [8, 8]
          })
        });
      }).filter(Boolean);
      if (!markers.length) return;
      foodBankLayer = L.layerGroup(markers).addTo(map);
      layerControl.addOverlay(foodBankLayer, "Food banks (optimized)");
    });
  return;
}

function getFeatureCenter(geom) {
  if (!geom) return null;
  try {
    const bounds = L.latLngBounds();
    const addCoords = coords => {
      if (!Array.isArray(coords)) return;
      if (typeof coords[0] === "number" && typeof coords[1] === "number") {
        bounds.extend([coords[1], coords[0]]);
        return;
      }
      coords.forEach(addCoords);
    };
    addCoords(geom.coordinates);
    if (!bounds.isValid()) return null;
    return bounds.getCenter();
  } catch (e) {
    return null;
  }
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
    const fbKey = foodBankMinKey();
    if (fbKey) return fbKey;
    if (state.mode === "walk") return "min_grocery_minutes";
    return state.minAccessKey || "min_transit_minutes";
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
