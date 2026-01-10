export async function onRequest({ params, env }) {
  if (!env?.DATA) {
    return new Response("R2 binding DATA is not configured.", { status: 500 });
  }
  const key = Array.isArray(params?.path)
    ? params.path.join("/")
    : params?.path || "";
  if (!key) {
    return new Response("Not found", { status: 404 });
  }
  let obj = await env.DATA.get(key);
  if (!obj && key.startsWith("data/")) {
    obj = await env.DATA.get(key.slice(5));
  }
  if (!obj && !key.startsWith("data/")) {
    obj = await env.DATA.get(`data/${key}`);
  }
  if (!obj) {
    return new Response("Not found", { status: 404 });
  }
  const headers = new Headers();
  if (obj.httpMetadata?.contentType) {
    headers.set("content-type", obj.httpMetadata.contentType);
  } else if (key.endsWith(".json") || key.endsWith(".geojson")) {
    headers.set("content-type", "application/json");
  } else if (key.endsWith(".csv")) {
    headers.set("content-type", "text/csv");
  }
  headers.set("cache-control", "public, max-age=86400");
  return new Response(obj.body, { headers });
}
