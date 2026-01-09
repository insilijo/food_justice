export async function onRequest({ params, env }) {
  const key = Array.isArray(params?.path)
    ? params.path.join("/")
    : params?.path || "";
  if (!key) {
    return new Response("Not found", { status: 404 });
  }
  const obj = await env.DATA.get(key);
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
