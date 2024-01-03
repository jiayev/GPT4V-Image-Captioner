import * as svelte from './svelte/svelte.js';

true&&(function polyfill() {
    const relList = document.createElement('link').relList;
    if (relList && relList.supports && relList.supports('modulepreload')) {
        return;
    }
    for (const link of document.querySelectorAll('link[rel="modulepreload"]')) {
        processPreload(link);
    }
    new MutationObserver((mutations) => {
        for (const mutation of mutations) {
            if (mutation.type !== 'childList') {
                continue;
            }
            for (const node of mutation.addedNodes) {
                if (node.tagName === 'LINK' && node.rel === 'modulepreload')
                    processPreload(node);
            }
        }
    }).observe(document, { childList: true, subtree: true });
    function getFetchOpts(link) {
        const fetchOpts = {};
        if (link.integrity)
            fetchOpts.integrity = link.integrity;
        if (link.referrerPolicy)
            fetchOpts.referrerPolicy = link.referrerPolicy;
        if (link.crossOrigin === 'use-credentials')
            fetchOpts.credentials = 'include';
        else if (link.crossOrigin === 'anonymous')
            fetchOpts.credentials = 'omit';
        else
            fetchOpts.credentials = 'same-origin';
        return fetchOpts;
    }
    function processPreload(link) {
        if (link.ep)
            // ep marker = processed
            return;
        link.ep = true;
        // prepopulate the load record
        const fetchOpts = getFetchOpts(link);
        fetch(link.href, fetchOpts);
    }
}());

const scriptRel = 'modulepreload';const assetsURL = function(dep, importerUrl) { return new URL(dep, importerUrl).href };const seen = {};const __vitePreload = function preload(baseModule, deps, importerUrl) {
    // @ts-expect-error true will be replaced with boolean later
    if (!true || !deps || deps.length === 0) {
        return baseModule();
    }
    const links = document.getElementsByTagName('link');
    return Promise.all(deps.map((dep) => {
        // @ts-expect-error assetsURL is declared before preload.toString()
        dep = assetsURL(dep, importerUrl);
        if (dep in seen)
            return;
        seen[dep] = true;
        const isCss = dep.endsWith('.css');
        const cssSelector = isCss ? '[rel="stylesheet"]' : '';
        const isBaseRelative = !!importerUrl;
        // check if the file is already preloaded by SSR markup
        if (isBaseRelative) {
            // When isBaseRelative is true then we have `importerUrl` and `dep` is
            // already converted to an absolute URL by the `assetsURL` function
            for (let i = links.length - 1; i >= 0; i--) {
                const link = links[i];
                // The `links[i].href` is an absolute URL thanks to browser doing the work
                // for us. See https://html.spec.whatwg.org/multipage/common-dom-interfaces.html#reflecting-content-attributes-in-idl-attributes:idl-domstring-5
                if (link.href === dep && (!isCss || link.rel === 'stylesheet')) {
                    return;
                }
            }
        }
        else if (document.querySelector(`link[href="${dep}"]${cssSelector}`)) {
            return;
        }
        const link = document.createElement('link');
        link.rel = isCss ? 'stylesheet' : scriptRel;
        if (!isCss) {
            link.as = 'script';
            link.crossOrigin = '';
        }
        link.href = dep;
        document.head.appendChild(link);
        if (isCss) {
            return new Promise((res, rej) => {
                link.addEventListener('load', res);
                link.addEventListener('error', () => rej(new Error(`Unable to preload CSS for ${dep}`)));
            });
        }
    })).then(() => baseModule());
};

const reset = '';

const global$1 = '';

const pollen = '';

const typography = '';

var fn = new Intl.Collator(0, { numeric: 1 }).compare;
function semiver(a, b, bool) {
  a = a.split(".");
  b = b.split(".");
  return fn(a[0], b[0]) || fn(a[1], b[1]) || (b[2] = b.slice(2).join("."), bool = /[.-]/.test(a[2] = a.slice(2).join(".")), bool == /[.-]/.test(b[2]) ? fn(a[2], b[2]) : bool ? -1 : 1);
}
function resolve_root(base_url, root_path, prioritize_base) {
  if (root_path.startsWith("http://") || root_path.startsWith("https://")) {
    return prioritize_base ? base_url : root_path;
  }
  return base_url + root_path;
}
function determine_protocol(endpoint) {
  if (endpoint.startsWith("http")) {
    const { protocol, host } = new URL(endpoint);
    if (host.endsWith("hf.space")) {
      return {
        ws_protocol: "wss",
        host,
        http_protocol: protocol
      };
    }
    return {
      ws_protocol: protocol === "https:" ? "wss" : "ws",
      http_protocol: protocol,
      host
    };
  } else if (endpoint.startsWith("file:")) {
    return {
      ws_protocol: "ws",
      http_protocol: "http:",
      host: "lite.local"
      // Special fake hostname only used for this case. This matches the hostname allowed in `is_self_host()` in `js/wasm/network/host.ts`.
    };
  }
  return {
    ws_protocol: "wss",
    http_protocol: "https:",
    host: endpoint
  };
}
const RE_SPACE_NAME = /^[^\/]*\/[^\/]*$/;
const RE_SPACE_DOMAIN = /.*hf\.space\/{0,1}$/;
async function process_endpoint(app_reference, token) {
  const headers = {};
  if (token) {
    headers.Authorization = `Bearer ${token}`;
  }
  const _app_reference = app_reference.trim();
  if (RE_SPACE_NAME.test(_app_reference)) {
    try {
      const res = await fetch(
        `https://huggingface.co/api/spaces/${_app_reference}/host`,
        { headers }
      );
      if (res.status !== 200)
        throw new Error("Space metadata could not be loaded.");
      const _host = (await res.json()).host;
      return {
        space_id: app_reference,
        ...determine_protocol(_host)
      };
    } catch (e) {
      throw new Error("Space metadata could not be loaded." + e.message);
    }
  }
  if (RE_SPACE_DOMAIN.test(_app_reference)) {
    const { ws_protocol, http_protocol, host } = determine_protocol(_app_reference);
    return {
      space_id: host.replace(".hf.space", ""),
      ws_protocol,
      http_protocol,
      host
    };
  }
  return {
    space_id: false,
    ...determine_protocol(_app_reference)
  };
}
function map_names_to_ids(fns) {
  let apis = {};
  fns.forEach(({ api_name }, i) => {
    if (api_name)
      apis[api_name] = i;
  });
  return apis;
}
const RE_DISABLED_DISCUSSION = /^(?=[^]*\b[dD]iscussions{0,1}\b)(?=[^]*\b[dD]isabled\b)[^]*$/;
async function discussions_enabled(space_id) {
  try {
    const r = await fetch(
      `https://huggingface.co/api/spaces/${space_id}/discussions`,
      {
        method: "HEAD"
      }
    );
    const error = r.headers.get("x-error-message");
    if (error && RE_DISABLED_DISCUSSION.test(error))
      return false;
    return true;
  } catch (e) {
    return false;
  }
}
function normalise_file(file, server_url, proxy_url) {
  if (file == null) {
    return null;
  }
  if (Array.isArray(file)) {
    const normalized_file = [];
    for (const x of file) {
      if (x == null) {
        normalized_file.push(null);
      } else {
        normalized_file.push(normalise_file(x, server_url, proxy_url));
      }
    }
    return normalized_file;
  }
  if (file.is_stream) {
    if (proxy_url == null) {
      return new FileData({
        ...file,
        url: server_url + "/stream/" + file.path
      });
    }
    return new FileData({
      ...file,
      url: "/proxy=" + proxy_url + "stream/" + file.path
    });
  }
  return new FileData({
    ...file,
    url: get_fetchable_url_or_file(file.path, server_url, proxy_url)
  });
}
function is_url(str) {
  try {
    const url = new URL(str);
    return url.protocol === "http:" || url.protocol === "https:";
  } catch {
    return false;
  }
}
function get_fetchable_url_or_file(path, server_url, proxy_url) {
  if (path == null) {
    return proxy_url ? `/proxy=${proxy_url}file=` : `${server_url}/file=`;
  }
  if (is_url(path)) {
    return path;
  }
  return proxy_url ? `/proxy=${proxy_url}file=${path}` : `${server_url}/file=${path}`;
}
async function upload(file_data, root, upload_id, upload_fn = upload_files) {
  let files = (Array.isArray(file_data) ? file_data : [file_data]).map(
    (file_data2) => file_data2.blob
  );
  return await Promise.all(
    await upload_fn(root, files, void 0, upload_id).then(
      async (response) => {
        if (response.error) {
          throw new Error(response.error);
        } else {
          if (response.files) {
            return response.files.map((f, i) => {
              const file = new FileData({ ...file_data[i], path: f });
              return normalise_file(file, root, null);
            });
          }
          return [];
        }
      }
    )
  );
}
async function prepare_files(files, is_stream) {
  return files.map(
    (f, i) => new FileData({
      path: f.name,
      orig_name: f.name,
      blob: f,
      size: f.size,
      mime_type: f.type,
      is_stream
    })
  );
}
class FileData {
  constructor({
    path,
    url,
    orig_name,
    size,
    blob,
    is_stream,
    mime_type,
    alt_text
  }) {
    this.path = path;
    this.url = url;
    this.orig_name = orig_name;
    this.size = size;
    this.blob = url ? void 0 : blob;
    this.is_stream = is_stream;
    this.mime_type = mime_type;
    this.alt_text = alt_text;
  }
}
const QUEUE_FULL_MSG = "This application is too busy. Keep trying!";
const BROKEN_CONNECTION_MSG = "Connection errored out.";
let NodeBlob;
function api_factory(fetch_implementation, EventSource_factory) {
  return { post_data: post_data2, upload_files: upload_files2, client: client2, handle_blob: handle_blob2 };
  async function post_data2(url, body, token) {
    const headers = { "Content-Type": "application/json" };
    if (token) {
      headers.Authorization = `Bearer ${token}`;
    }
    try {
      var response = await fetch_implementation(url, {
        method: "POST",
        body: JSON.stringify(body),
        headers
      });
    } catch (e) {
      return [{ error: BROKEN_CONNECTION_MSG }, 500];
    }
    let output;
    let status;
    try {
      output = await response.json();
      status = response.status;
    } catch (e) {
      output = { error: `Could not parse server response: ${e}` };
      status = 500;
    }
    return [output, status];
  }
  async function upload_files2(root, files, token, upload_id) {
    const headers = {};
    if (token) {
      headers.Authorization = `Bearer ${token}`;
    }
    const chunkSize = 1e3;
    const uploadResponses = [];
    for (let i = 0; i < files.length; i += chunkSize) {
      const chunk = files.slice(i, i + chunkSize);
      const formData = new FormData();
      chunk.forEach((file) => {
        formData.append("files", file);
      });
      try {
        const upload_url = upload_id ? `${root}/upload?upload_id=${upload_id}` : `${root}/upload`;
        var response = await fetch_implementation(upload_url, {
          method: "POST",
          body: formData,
          headers
        });
      } catch (e) {
        return { error: BROKEN_CONNECTION_MSG };
      }
      const output = await response.json();
      uploadResponses.push(...output);
    }
    return { files: uploadResponses };
  }
  async function client2(app_reference, options = { normalise_files: true }) {
    return new Promise(async (res) => {
      const { status_callback, hf_token, normalise_files } = options;
      const return_obj = {
        predict,
        submit,
        view_api,
        component_server
      };
      const transform_files = normalise_files ?? true;
      if ((typeof window === "undefined" || !("WebSocket" in window)) && !global.Websocket) {
        const ws = await __vitePreload(() => import('./wrapper-6f348d45-eeeb8bd1.js'),true?["./wrapper-6f348d45-eeeb8bd1.js","./__vite-browser-external-cb19bde2.js"]:void 0,import.meta.url);
        NodeBlob = (await __vitePreload(() => import('./__vite-browser-external-cb19bde2.js').then(n => n._),true?[]:void 0,import.meta.url)).Blob;
        global.WebSocket = ws.WebSocket;
      }
      const { ws_protocol, http_protocol, host, space_id } = await process_endpoint(app_reference, hf_token);
      const session_hash = Math.random().toString(36).substring(2);
      const last_status = {};
      let stream_open = false;
      let pending_stream_messages = {};
      let event_stream = null;
      const event_callbacks = {};
      let config;
      let api_map = {};
      let jwt = false;
      if (hf_token && space_id) {
        jwt = await get_jwt(space_id, hf_token);
      }
      async function config_success(_config) {
        config = _config;
        api_map = map_names_to_ids((_config == null ? void 0 : _config.dependencies) || []);
        if (config.auth_required) {
          return {
            config,
            ...return_obj
          };
        }
        try {
          api = await view_api(config);
        } catch (e) {
          console.error(`Could not get api details: ${e.message}`);
        }
        return {
          config,
          ...return_obj
        };
      }
      let api;
      async function handle_space_sucess(status) {
        if (status_callback)
          status_callback(status);
        if (status.status === "running")
          try {
            config = await resolve_config(
              fetch_implementation,
              `${http_protocol}//${host}`,
              hf_token
            );
            const _config = await config_success(config);
            res(_config);
          } catch (e) {
            console.error(e);
            if (status_callback) {
              status_callback({
                status: "error",
                message: "Could not load this space.",
                load_status: "error",
                detail: "NOT_FOUND"
              });
            }
          }
      }
      try {
        config = await resolve_config(
          fetch_implementation,
          `${http_protocol}//${host}`,
          hf_token
        );
        const _config = await config_success(config);
        res(_config);
      } catch (e) {
        console.error(e);
        if (space_id) {
          check_space_status(
            space_id,
            RE_SPACE_NAME.test(space_id) ? "space_name" : "subdomain",
            handle_space_sucess
          );
        } else {
          if (status_callback)
            status_callback({
              status: "error",
              message: "Could not load this space.",
              load_status: "error",
              detail: "NOT_FOUND"
            });
        }
      }
      function predict(endpoint, data, event_data) {
        let data_returned = false;
        let status_complete = false;
        let dependency;
        if (typeof endpoint === "number") {
          dependency = config.dependencies[endpoint];
        } else {
          const trimmed_endpoint = endpoint.replace(/^\//, "");
          dependency = config.dependencies[api_map[trimmed_endpoint]];
        }
        if (dependency.types.continuous) {
          throw new Error(
            "Cannot call predict on this function as it may run forever. Use submit instead"
          );
        }
        return new Promise((res2, rej) => {
          const app = submit(endpoint, data, event_data);
          let result;
          app.on("data", (d) => {
            if (status_complete) {
              app.destroy();
              res2(d);
            }
            data_returned = true;
            result = d;
          }).on("status", (status) => {
            if (status.stage === "error")
              rej(status);
            if (status.stage === "complete") {
              status_complete = true;
              if (data_returned) {
                app.destroy();
                res2(result);
              }
            }
          });
        });
      }
      function submit(endpoint, data, event_data, trigger_id = null) {
        let fn_index;
        let api_info;
        if (typeof endpoint === "number") {
          fn_index = endpoint;
          api_info = api.unnamed_endpoints[fn_index];
        } else {
          const trimmed_endpoint = endpoint.replace(/^\//, "");
          fn_index = api_map[trimmed_endpoint];
          api_info = api.named_endpoints[endpoint.trim()];
        }
        if (typeof fn_index !== "number") {
          throw new Error(
            "There is no endpoint matching that name of fn_index matching that number."
          );
        }
        let websocket;
        let eventSource;
        let protocol = config.protocol ?? "ws";
        const _endpoint = typeof endpoint === "number" ? "/predict" : endpoint;
        let payload;
        let event_id = null;
        let complete = false;
        const listener_map = {};
        let url_params = "";
        if (typeof window !== "undefined") {
          url_params = new URLSearchParams(window.location.search).toString();
        }
        handle_blob2(`${config.root}`, data, api_info, hf_token).then(
          (_payload) => {
            payload = {
              data: _payload || [],
              event_data,
              fn_index,
              trigger_id
            };
            if (skip_queue(fn_index, config)) {
              fire_event({
                type: "status",
                endpoint: _endpoint,
                stage: "pending",
                queue: false,
                fn_index,
                time: /* @__PURE__ */ new Date()
              });
              post_data2(
                `${config.root}/run${_endpoint.startsWith("/") ? _endpoint : `/${_endpoint}`}${url_params ? "?" + url_params : ""}`,
                {
                  ...payload,
                  session_hash
                },
                hf_token
              ).then(([output, status_code]) => {
                const data2 = transform_files ? transform_output(
                  output.data,
                  api_info,
                  config.root,
                  config.root_url
                ) : output.data;
                if (status_code == 200) {
                  fire_event({
                    type: "data",
                    endpoint: _endpoint,
                    fn_index,
                    data: data2,
                    time: /* @__PURE__ */ new Date()
                  });
                  fire_event({
                    type: "status",
                    endpoint: _endpoint,
                    fn_index,
                    stage: "complete",
                    eta: output.average_duration,
                    queue: false,
                    time: /* @__PURE__ */ new Date()
                  });
                } else {
                  fire_event({
                    type: "status",
                    stage: "error",
                    endpoint: _endpoint,
                    fn_index,
                    message: output.error,
                    queue: false,
                    time: /* @__PURE__ */ new Date()
                  });
                }
              }).catch((e) => {
                fire_event({
                  type: "status",
                  stage: "error",
                  message: e.message,
                  endpoint: _endpoint,
                  fn_index,
                  queue: false,
                  time: /* @__PURE__ */ new Date()
                });
              });
            } else if (protocol == "ws") {
              fire_event({
                type: "status",
                stage: "pending",
                queue: true,
                endpoint: _endpoint,
                fn_index,
                time: /* @__PURE__ */ new Date()
              });
              let url = new URL(`${ws_protocol}://${resolve_root(
                host,
                config.path,
                true
              )}
							/queue/join${url_params ? "?" + url_params : ""}`);
              if (jwt) {
                url.searchParams.set("__sign", jwt);
              }
              websocket = new WebSocket(url);
              websocket.onclose = (evt) => {
                if (!evt.wasClean) {
                  fire_event({
                    type: "status",
                    stage: "error",
                    broken: true,
                    message: BROKEN_CONNECTION_MSG,
                    queue: true,
                    endpoint: _endpoint,
                    fn_index,
                    time: /* @__PURE__ */ new Date()
                  });
                }
              };
              websocket.onmessage = function(event) {
                const _data = JSON.parse(event.data);
                const { type, status, data: data2 } = handle_message(
                  _data,
                  last_status[fn_index]
                );
                if (type === "update" && status && !complete) {
                  fire_event({
                    type: "status",
                    endpoint: _endpoint,
                    fn_index,
                    time: /* @__PURE__ */ new Date(),
                    ...status
                  });
                  if (status.stage === "error") {
                    websocket.close();
                  }
                } else if (type === "hash") {
                  websocket.send(JSON.stringify({ fn_index, session_hash }));
                  return;
                } else if (type === "data") {
                  websocket.send(JSON.stringify({ ...payload, session_hash }));
                } else if (type === "complete") {
                  complete = status;
                } else if (type === "log") {
                  fire_event({
                    type: "log",
                    log: data2.log,
                    level: data2.level,
                    endpoint: _endpoint,
                    fn_index
                  });
                } else if (type === "generating") {
                  fire_event({
                    type: "status",
                    time: /* @__PURE__ */ new Date(),
                    ...status,
                    stage: status == null ? void 0 : status.stage,
                    queue: true,
                    endpoint: _endpoint,
                    fn_index
                  });
                }
                if (data2) {
                  fire_event({
                    type: "data",
                    time: /* @__PURE__ */ new Date(),
                    data: transform_files ? transform_output(
                      data2.data,
                      api_info,
                      config.root,
                      config.root_url
                    ) : data2.data,
                    endpoint: _endpoint,
                    fn_index
                  });
                  if (complete) {
                    fire_event({
                      type: "status",
                      time: /* @__PURE__ */ new Date(),
                      ...complete,
                      stage: status == null ? void 0 : status.stage,
                      queue: true,
                      endpoint: _endpoint,
                      fn_index
                    });
                    websocket.close();
                  }
                }
              };
              if (semiver(config.version || "2.0.0", "3.6") < 0) {
                addEventListener(
                  "open",
                  () => websocket.send(JSON.stringify({ hash: session_hash }))
                );
              }
            } else if (protocol == "sse") {
              fire_event({
                type: "status",
                stage: "pending",
                queue: true,
                endpoint: _endpoint,
                fn_index,
                time: /* @__PURE__ */ new Date()
              });
              var params = new URLSearchParams({
                fn_index: fn_index.toString(),
                session_hash
              }).toString();
              let url = new URL(
                `${config.root}/queue/join?${url_params ? url_params + "&" : ""}${params}`
              );
              eventSource = EventSource_factory(url);
              eventSource.onmessage = async function(event) {
                const _data = JSON.parse(event.data);
                const { type, status, data: data2 } = handle_message(
                  _data,
                  last_status[fn_index]
                );
                if (type === "update" && status && !complete) {
                  fire_event({
                    type: "status",
                    endpoint: _endpoint,
                    fn_index,
                    time: /* @__PURE__ */ new Date(),
                    ...status
                  });
                  if (status.stage === "error") {
                    eventSource.close();
                  }
                } else if (type === "data") {
                  event_id = _data.event_id;
                  let [_, status2] = await post_data2(
                    `${config.root}/queue/data`,
                    {
                      ...payload,
                      session_hash,
                      event_id
                    },
                    hf_token
                  );
                  if (status2 !== 200) {
                    fire_event({
                      type: "status",
                      stage: "error",
                      message: BROKEN_CONNECTION_MSG,
                      queue: true,
                      endpoint: _endpoint,
                      fn_index,
                      time: /* @__PURE__ */ new Date()
                    });
                    eventSource.close();
                  }
                } else if (type === "complete") {
                  complete = status;
                } else if (type === "log") {
                  fire_event({
                    type: "log",
                    log: data2.log,
                    level: data2.level,
                    endpoint: _endpoint,
                    fn_index
                  });
                } else if (type === "generating") {
                  fire_event({
                    type: "status",
                    time: /* @__PURE__ */ new Date(),
                    ...status,
                    stage: status == null ? void 0 : status.stage,
                    queue: true,
                    endpoint: _endpoint,
                    fn_index
                  });
                }
                if (data2) {
                  fire_event({
                    type: "data",
                    time: /* @__PURE__ */ new Date(),
                    data: transform_files ? transform_output(
                      data2.data,
                      api_info,
                      config.root,
                      config.root_url
                    ) : data2.data,
                    endpoint: _endpoint,
                    fn_index
                  });
                  if (complete) {
                    fire_event({
                      type: "status",
                      time: /* @__PURE__ */ new Date(),
                      ...complete,
                      stage: status == null ? void 0 : status.stage,
                      queue: true,
                      endpoint: _endpoint,
                      fn_index
                    });
                    eventSource.close();
                  }
                }
              };
            } else if (protocol == "sse_v1") {
              fire_event({
                type: "status",
                stage: "pending",
                queue: true,
                endpoint: _endpoint,
                fn_index,
                time: /* @__PURE__ */ new Date()
              });
              post_data2(
                `${config.root}/queue/join?${url_params}`,
                {
                  ...payload,
                  session_hash
                },
                hf_token
              ).then(([response, status]) => {
                if (status === 503) {
                  fire_event({
                    type: "status",
                    stage: "error",
                    message: QUEUE_FULL_MSG,
                    queue: true,
                    endpoint: _endpoint,
                    fn_index,
                    time: /* @__PURE__ */ new Date()
                  });
                } else if (status !== 200) {
                  fire_event({
                    type: "status",
                    stage: "error",
                    message: BROKEN_CONNECTION_MSG,
                    queue: true,
                    endpoint: _endpoint,
                    fn_index,
                    time: /* @__PURE__ */ new Date()
                  });
                } else {
                  event_id = response.event_id;
                  let callback = async function(_data) {
                    try {
                      const { type, status: status2, data: data2 } = handle_message(
                        _data,
                        last_status[fn_index]
                      );
                      if (type == "heartbeat") {
                        return;
                      }
                      if (type === "update" && status2 && !complete) {
                        fire_event({
                          type: "status",
                          endpoint: _endpoint,
                          fn_index,
                          time: /* @__PURE__ */ new Date(),
                          ...status2
                        });
                      } else if (type === "complete") {
                        complete = status2;
                      } else if (type == "unexpected_error") {
                        console.error("Unexpected error", status2 == null ? void 0 : status2.message);
                        fire_event({
                          type: "status",
                          stage: "error",
                          message: "An Unexpected Error Occurred!",
                          queue: true,
                          endpoint: _endpoint,
                          fn_index,
                          time: /* @__PURE__ */ new Date()
                        });
                      } else if (type === "log") {
                        fire_event({
                          type: "log",
                          log: data2.log,
                          level: data2.level,
                          endpoint: _endpoint,
                          fn_index
                        });
                        return;
                      } else if (type === "generating") {
                        fire_event({
                          type: "status",
                          time: /* @__PURE__ */ new Date(),
                          ...status2,
                          stage: status2 == null ? void 0 : status2.stage,
                          queue: true,
                          endpoint: _endpoint,
                          fn_index
                        });
                      }
                      if (data2) {
                        fire_event({
                          type: "data",
                          time: /* @__PURE__ */ new Date(),
                          data: transform_files ? transform_output(
                            data2.data,
                            api_info,
                            config.root,
                            config.root_url
                          ) : data2.data,
                          endpoint: _endpoint,
                          fn_index
                        });
                        if (complete) {
                          fire_event({
                            type: "status",
                            time: /* @__PURE__ */ new Date(),
                            ...complete,
                            stage: status2 == null ? void 0 : status2.stage,
                            queue: true,
                            endpoint: _endpoint,
                            fn_index
                          });
                        }
                      }
                      if ((status2 == null ? void 0 : status2.stage) === "complete" || (status2 == null ? void 0 : status2.stage) === "error") {
                        if (event_callbacks[event_id]) {
                          delete event_callbacks[event_id];
                          if (Object.keys(event_callbacks).length === 0) {
                            close_stream();
                          }
                        }
                      }
                    } catch (e) {
                      console.error("Unexpected client exception", e);
                      fire_event({
                        type: "status",
                        stage: "error",
                        message: "An Unexpected Error Occurred!",
                        queue: true,
                        endpoint: _endpoint,
                        fn_index,
                        time: /* @__PURE__ */ new Date()
                      });
                      close_stream();
                    }
                  };
                  if (event_id in pending_stream_messages) {
                    pending_stream_messages[event_id].forEach(
                      (msg) => callback(msg)
                    );
                    delete pending_stream_messages[event_id];
                  }
                  event_callbacks[event_id] = callback;
                  if (!stream_open) {
                    open_stream();
                  }
                }
              });
            }
          }
        );
        function fire_event(event) {
          const narrowed_listener_map = listener_map;
          const listeners = narrowed_listener_map[event.type] || [];
          listeners == null ? void 0 : listeners.forEach((l) => l(event));
        }
        function on(eventType, listener) {
          const narrowed_listener_map = listener_map;
          const listeners = narrowed_listener_map[eventType] || [];
          narrowed_listener_map[eventType] = listeners;
          listeners == null ? void 0 : listeners.push(listener);
          return { on, off, cancel, destroy };
        }
        function off(eventType, listener) {
          const narrowed_listener_map = listener_map;
          let listeners = narrowed_listener_map[eventType] || [];
          listeners = listeners == null ? void 0 : listeners.filter((l) => l !== listener);
          narrowed_listener_map[eventType] = listeners;
          return { on, off, cancel, destroy };
        }
        async function cancel() {
          const _status = {
            stage: "complete",
            queue: false,
            time: /* @__PURE__ */ new Date()
          };
          complete = _status;
          fire_event({
            ..._status,
            type: "status",
            endpoint: _endpoint,
            fn_index
          });
          let cancel_request = {};
          if (protocol === "ws") {
            if (websocket && websocket.readyState === 0) {
              websocket.addEventListener("open", () => {
                websocket.close();
              });
            } else {
              websocket.close();
            }
            cancel_request = { fn_index, session_hash };
          } else {
            eventSource.close();
            cancel_request = { event_id };
          }
          try {
            await fetch_implementation(`${config.root}/reset`, {
              headers: { "Content-Type": "application/json" },
              method: "POST",
              body: JSON.stringify(cancel_request)
            });
          } catch (e) {
            console.warn(
              "The `/reset` endpoint could not be called. Subsequent endpoint results may be unreliable."
            );
          }
        }
        function destroy() {
          for (const event_type in listener_map) {
            listener_map[event_type].forEach((fn2) => {
              off(event_type, fn2);
            });
          }
        }
        return {
          on,
          off,
          cancel,
          destroy
        };
      }
      function open_stream() {
        stream_open = true;
        let params = new URLSearchParams({
          session_hash
        }).toString();
        let url = new URL(`${config.root}/queue/data?${params}`);
        event_stream = new EventSource(url);
        event_stream.onmessage = async function(event) {
          let _data = JSON.parse(event.data);
          const event_id = _data.event_id;
          if (!event_id) {
            await Promise.all(
              Object.keys(event_callbacks).map(
                (event_id2) => event_callbacks[event_id2](_data)
              )
            );
          } else if (event_callbacks[event_id]) {
            await event_callbacks[event_id](_data);
          } else {
            if (!pending_stream_messages[event_id]) {
              pending_stream_messages[event_id] = [];
            }
            pending_stream_messages[event_id].push(_data);
          }
        };
      }
      function close_stream() {
        stream_open = false;
        event_stream == null ? void 0 : event_stream.close();
      }
      async function component_server(component_id, fn_name, data) {
        var _a;
        const headers = { "Content-Type": "application/json" };
        if (hf_token) {
          headers.Authorization = `Bearer ${hf_token}`;
        }
        let root_url;
        let component = config.components.find(
          (comp) => comp.id === component_id
        );
        if ((_a = component == null ? void 0 : component.props) == null ? void 0 : _a.root_url) {
          root_url = component.props.root_url;
        } else {
          root_url = config.root;
        }
        const response = await fetch_implementation(
          `${root_url}/component_server/`,
          {
            method: "POST",
            body: JSON.stringify({
              data,
              component_id,
              fn_name,
              session_hash
            }),
            headers
          }
        );
        if (!response.ok) {
          throw new Error(
            "Could not connect to component server: " + response.statusText
          );
        }
        const output = await response.json();
        return output;
      }
      async function view_api(config2) {
        if (api)
          return api;
        const headers = { "Content-Type": "application/json" };
        if (hf_token) {
          headers.Authorization = `Bearer ${hf_token}`;
        }
        let response;
        if (semiver(config2.version || "2.0.0", "3.30") < 0) {
          response = await fetch_implementation(
            "https://gradio-space-api-fetcher-v2.hf.space/api",
            {
              method: "POST",
              body: JSON.stringify({
                serialize: false,
                config: JSON.stringify(config2)
              }),
              headers
            }
          );
        } else {
          response = await fetch_implementation(`${config2.root}/info`, {
            headers
          });
        }
        if (!response.ok) {
          throw new Error(BROKEN_CONNECTION_MSG);
        }
        let api_info = await response.json();
        if ("api" in api_info) {
          api_info = api_info.api;
        }
        if (api_info.named_endpoints["/predict"] && !api_info.unnamed_endpoints["0"]) {
          api_info.unnamed_endpoints[0] = api_info.named_endpoints["/predict"];
        }
        const x = transform_api_info(api_info, config2, api_map);
        return x;
      }
    });
  }
  async function handle_blob2(endpoint, data, api_info, token) {
    const blob_refs = await walk_and_store_blobs(
      data,
      void 0,
      [],
      true,
      api_info
    );
    return Promise.all(
      blob_refs.map(async ({ path, blob, type }) => {
        if (blob) {
          const file_url = (await upload_files2(endpoint, [blob], token)).files[0];
          return { path, file_url, type, name: blob == null ? void 0 : blob.name };
        }
        return { path, type };
      })
    ).then((r) => {
      r.forEach(({ path, file_url, type, name }) => {
        if (type === "Gallery") {
          update_object(data, file_url, path);
        } else if (file_url) {
          const file = new FileData({ path: file_url, orig_name: name });
          update_object(data, file, path);
        }
      });
      return data;
    });
  }
}
const { post_data, upload_files, client, handle_blob } = api_factory(
  fetch,
  (...args) => new EventSource(...args)
);
function transform_output(data, api_info, root_url, remote_url) {
  return data.map((d, i) => {
    var _a, _b, _c, _d;
    if (((_b = (_a = api_info == null ? void 0 : api_info.returns) == null ? void 0 : _a[i]) == null ? void 0 : _b.component) === "File") {
      return normalise_file(d, root_url, remote_url);
    } else if (((_d = (_c = api_info == null ? void 0 : api_info.returns) == null ? void 0 : _c[i]) == null ? void 0 : _d.component) === "Gallery") {
      return d.map((img) => {
        return Array.isArray(img) ? [normalise_file(img[0], root_url, remote_url), img[1]] : [normalise_file(img, root_url, remote_url), null];
      });
    } else if (typeof d === "object" && d.path) {
      return normalise_file(d, root_url, remote_url);
    }
    return d;
  });
}
function get_type(type, component, serializer, signature_type) {
  switch (type.type) {
    case "string":
      return "string";
    case "boolean":
      return "boolean";
    case "number":
      return "number";
  }
  if (serializer === "JSONSerializable" || serializer === "StringSerializable") {
    return "any";
  } else if (serializer === "ListStringSerializable") {
    return "string[]";
  } else if (component === "Image") {
    return signature_type === "parameter" ? "Blob | File | Buffer" : "string";
  } else if (serializer === "FileSerializable") {
    if ((type == null ? void 0 : type.type) === "array") {
      return signature_type === "parameter" ? "(Blob | File | Buffer)[]" : `{ name: string; data: string; size?: number; is_file?: boolean; orig_name?: string}[]`;
    }
    return signature_type === "parameter" ? "Blob | File | Buffer" : `{ name: string; data: string; size?: number; is_file?: boolean; orig_name?: string}`;
  } else if (serializer === "GallerySerializable") {
    return signature_type === "parameter" ? "[(Blob | File | Buffer), (string | null)][]" : `[{ name: string; data: string; size?: number; is_file?: boolean; orig_name?: string}, (string | null))][]`;
  }
}
function get_description(type, serializer) {
  if (serializer === "GallerySerializable") {
    return "array of [file, label] tuples";
  } else if (serializer === "ListStringSerializable") {
    return "array of strings";
  } else if (serializer === "FileSerializable") {
    return "array of files or single file";
  }
  return type.description;
}
function transform_api_info(api_info, config, api_map) {
  const new_data = {
    named_endpoints: {},
    unnamed_endpoints: {}
  };
  for (const key in api_info) {
    const cat = api_info[key];
    for (const endpoint in cat) {
      const dep_index = config.dependencies[endpoint] ? endpoint : api_map[endpoint.replace("/", "")];
      const info = cat[endpoint];
      new_data[key][endpoint] = {};
      new_data[key][endpoint].parameters = {};
      new_data[key][endpoint].returns = {};
      new_data[key][endpoint].type = config.dependencies[dep_index].types;
      new_data[key][endpoint].parameters = info.parameters.map(
        ({ label, component, type, serializer }) => ({
          label,
          component,
          type: get_type(type, component, serializer, "parameter"),
          description: get_description(type, serializer)
        })
      );
      new_data[key][endpoint].returns = info.returns.map(
        ({ label, component, type, serializer }) => ({
          label,
          component,
          type: get_type(type, component, serializer, "return"),
          description: get_description(type, serializer)
        })
      );
    }
  }
  return new_data;
}
async function get_jwt(space, token) {
  try {
    const r = await fetch(`https://huggingface.co/api/spaces/${space}/jwt`, {
      headers: {
        Authorization: `Bearer ${token}`
      }
    });
    const jwt = (await r.json()).token;
    return jwt || false;
  } catch (e) {
    console.error(e);
    return false;
  }
}
function update_object(object, newValue, stack) {
  while (stack.length > 1) {
    object = object[stack.shift()];
  }
  object[stack.shift()] = newValue;
}
async function walk_and_store_blobs(param, type = void 0, path = [], root = false, api_info = void 0) {
  if (Array.isArray(param)) {
    let blob_refs = [];
    await Promise.all(
      param.map(async (v, i) => {
        var _a;
        let new_path = path.slice();
        new_path.push(i);
        const array_refs = await walk_and_store_blobs(
          param[i],
          root ? ((_a = api_info == null ? void 0 : api_info.parameters[i]) == null ? void 0 : _a.component) || void 0 : type,
          new_path,
          false,
          api_info
        );
        blob_refs = blob_refs.concat(array_refs);
      })
    );
    return blob_refs;
  } else if (globalThis.Buffer && param instanceof globalThis.Buffer) {
    const is_image = type === "Image";
    return [
      {
        path,
        blob: is_image ? false : new NodeBlob([param]),
        type
      }
    ];
  } else if (typeof param === "object") {
    let blob_refs = [];
    for (let key in param) {
      if (param.hasOwnProperty(key)) {
        let new_path = path.slice();
        new_path.push(key);
        blob_refs = blob_refs.concat(
          await walk_and_store_blobs(
            param[key],
            void 0,
            new_path,
            false,
            api_info
          )
        );
      }
    }
    return blob_refs;
  }
  return [];
}
function skip_queue(id, config) {
  var _a, _b, _c, _d;
  return !(((_b = (_a = config == null ? void 0 : config.dependencies) == null ? void 0 : _a[id]) == null ? void 0 : _b.queue) === null ? config.enable_queue : (_d = (_c = config == null ? void 0 : config.dependencies) == null ? void 0 : _c[id]) == null ? void 0 : _d.queue) || false;
}
async function resolve_config(fetch_implementation, endpoint, token) {
  const headers = {};
  if (token) {
    headers.Authorization = `Bearer ${token}`;
  }
  if (typeof window !== "undefined" && window.gradio_config && location.origin !== "http://localhost:9876" && !window.gradio_config.dev_mode) {
    const path = window.gradio_config.root;
    const config = window.gradio_config;
    config.root = resolve_root(endpoint, config.root, false);
    return { ...config, path };
  } else if (endpoint) {
    let response = await fetch_implementation(`${endpoint}/config`, {
      headers
    });
    if (response.status === 200) {
      const config = await response.json();
      config.path = config.path ?? "";
      config.root = endpoint;
      return config;
    }
    throw new Error("Could not get config.");
  }
  throw new Error("No config or app endpoint found");
}
async function check_space_status(id, type, status_callback) {
  let endpoint = type === "subdomain" ? `https://huggingface.co/api/spaces/by-subdomain/${id}` : `https://huggingface.co/api/spaces/${id}`;
  let response;
  let _status;
  try {
    response = await fetch(endpoint);
    _status = response.status;
    if (_status !== 200) {
      throw new Error();
    }
    response = await response.json();
  } catch (e) {
    status_callback({
      status: "error",
      load_status: "error",
      message: "Could not get space status",
      detail: "NOT_FOUND"
    });
    return;
  }
  if (!response || _status !== 200)
    return;
  const {
    runtime: { stage },
    id: space_name
  } = response;
  switch (stage) {
    case "STOPPED":
    case "SLEEPING":
      status_callback({
        status: "sleeping",
        load_status: "pending",
        message: "Space is asleep. Waking it up...",
        detail: stage
      });
      setTimeout(() => {
        check_space_status(id, type, status_callback);
      }, 1e3);
      break;
    case "PAUSED":
      status_callback({
        status: "paused",
        load_status: "error",
        message: "This space has been paused by the author. If you would like to try this demo, consider duplicating the space.",
        detail: stage,
        discussions_enabled: await discussions_enabled(space_name)
      });
      break;
    case "RUNNING":
    case "RUNNING_BUILDING":
      status_callback({
        status: "running",
        load_status: "complete",
        message: "",
        detail: stage
      });
      break;
    case "BUILDING":
      status_callback({
        status: "building",
        load_status: "pending",
        message: "Space is building...",
        detail: stage
      });
      setTimeout(() => {
        check_space_status(id, type, status_callback);
      }, 1e3);
      break;
    default:
      status_callback({
        status: "space_error",
        load_status: "error",
        message: "This space is experiencing an issue.",
        detail: stage,
        discussions_enabled: await discussions_enabled(space_name)
      });
      break;
  }
}
function handle_message(data, last_status) {
  const queue = true;
  switch (data.msg) {
    case "send_data":
      return { type: "data" };
    case "send_hash":
      return { type: "hash" };
    case "queue_full":
      return {
        type: "update",
        status: {
          queue,
          message: QUEUE_FULL_MSG,
          stage: "error",
          code: data.code,
          success: data.success
        }
      };
    case "heartbeat":
      return {
        type: "heartbeat"
      };
    case "unexpected_error":
      return {
        type: "unexpected_error",
        status: {
          queue,
          message: data.message,
          stage: "error",
          success: false
        }
      };
    case "estimation":
      return {
        type: "update",
        status: {
          queue,
          stage: last_status || "pending",
          code: data.code,
          size: data.queue_size,
          position: data.rank,
          eta: data.rank_eta,
          success: data.success
        }
      };
    case "progress":
      return {
        type: "update",
        status: {
          queue,
          stage: "pending",
          code: data.code,
          progress_data: data.progress_data,
          success: data.success
        }
      };
    case "log":
      return { type: "log", data };
    case "process_generating":
      return {
        type: "generating",
        status: {
          queue,
          message: !data.success ? data.output.error : null,
          stage: data.success ? "generating" : "error",
          code: data.code,
          progress_data: data.progress_data,
          eta: data.average_duration
        },
        data: data.success ? data.output : null
      };
    case "process_completed":
      if ("error" in data.output) {
        return {
          type: "update",
          status: {
            queue,
            message: data.output.error,
            stage: "error",
            code: data.code,
            success: data.success
          }
        };
      }
      return {
        type: "complete",
        status: {
          queue,
          message: !data.success ? data.output.error : void 0,
          stage: data.success ? "complete" : "error",
          code: data.code,
          progress_data: data.progress_data
        },
        data: data.success ? data.output : null
      };
    case "process_starts":
      return {
        type: "update",
        status: {
          queue,
          stage: "pending",
          code: data.code,
          size: data.rank,
          position: 0,
          success: data.success,
          eta: data.eta
        }
      };
  }
  return { type: "none", status: { stage: "error", queue } };
}

function mount_css(url, target) {
  const base = new URL(import.meta.url).origin;
  const _url = new URL(url, base).href;
  const existing_link = document.querySelector(`link[href='${_url}']`);
  if (existing_link)
    return Promise.resolve();
  const link = document.createElement("link");
  link.rel = "stylesheet";
  link.href = _url;
  return new Promise((res, rej) => {
    link.addEventListener("load", () => res());
    link.addEventListener("error", () => {
      console.error(`Unable to preload CSS for ${_url}`);
      res();
    });
    target.appendChild(link);
  });
}
function prefix_css(string, version, style_element = document.createElement("style")) {
  style_element.remove();
  const stylesheet = new CSSStyleSheet();
  stylesheet.replaceSync(string);
  const rules = stylesheet.cssRules;
  let css_string = "";
  for (let i = 0; i < rules.length; i++) {
    const rule = rules[i];
    if (rule instanceof CSSStyleRule) {
      const selector = rule.selectorText;
      if (selector) {
        const new_selector = selector.split(",").map(
          (s) => `gradio-app .gradio-container.gradio-container-${version} .contain ${s.trim()}`
        ).join(",");
        css_string += rule.cssText;
        css_string += rule.cssText.replace(selector, new_selector);
      }
    }
  }
  style_element.textContent = css_string;
  document.head.appendChild(style_element);
  return style_element;
}

const ENTRY_CSS = "./assets/index-078de39b.css";
let FONTS;
FONTS = [];
let IndexComponent;
let _res;
let pending = new Promise((res) => {
  _res = res;
});
async function get_index() {
  IndexComponent = (await __vitePreload(() => import('./Index-a77cc637.js').then(n => n.I),true?["./Index-a77cc637.js","./Index-99651b20.css"]:void 0,import.meta.url)).default;
  _res();
}
function create_custom_element() {
  const o = {
    SvelteComponent: svelte.SvelteComponent
  };
  for (const key in svelte) {
    if (key === "SvelteComponent")
      continue;
    if (key === "SvelteComponentDev") {
      o[key] = o["SvelteComponent"];
    } else {
      o[key] = svelte[key];
    }
  }
  window.__gradio__svelte__internal = o;
  class GradioApp extends HTMLElement {
    constructor() {
      super();
      this.host = this.getAttribute("host");
      this.space = this.getAttribute("space");
      this.src = this.getAttribute("src");
      this.control_page_title = this.getAttribute("control_page_title");
      this.initial_height = this.getAttribute("initial_height") ?? "300px";
      this.is_embed = this.getAttribute("embed") ?? "true";
      this.container = this.getAttribute("container") ?? "true";
      this.info = this.getAttribute("info") ?? true;
      this.autoscroll = this.getAttribute("autoscroll");
      this.eager = this.getAttribute("eager");
      this.theme_mode = this.getAttribute("theme_mode");
      this.updating = false;
      this.loading = false;
    }
    async connectedCallback() {
      await get_index();
      this.loading = true;
      if (this.app) {
        this.app.$destroy();
      }
      if (typeof FONTS !== "string") {
        FONTS.forEach((f) => mount_css(f, document.head));
      }
      await mount_css(ENTRY_CSS, document.head);
      const event = new CustomEvent("domchange", {
        bubbles: true,
        cancelable: false,
        composed: true
      });
      const observer = new MutationObserver((mutations) => {
        this.dispatchEvent(event);
      });
      observer.observe(this, { childList: true });
      this.app = new IndexComponent({
        target: this,
        props: {
          // embed source
          space: this.space ? this.space.trim() : this.space,
          src: this.src ? this.src.trim() : this.src,
          host: this.host ? this.host.trim() : this.host,
          // embed info
          info: this.info === "false" ? false : true,
          container: this.container === "false" ? false : true,
          is_embed: this.is_embed === "false" ? false : true,
          initial_height: this.initial_height,
          eager: this.eager === "true" ? true : false,
          // gradio meta info
          version: "4-12-0",
          theme_mode: this.theme_mode,
          // misc global behaviour
          autoscroll: this.autoscroll === "true" ? true : false,
          control_page_title: this.control_page_title === "true" ? true : false,
          // injectables
          client,
          upload_files,
          // for gradio docs
          // TODO: Remove -- i think this is just for autoscroll behavhiour, app vs embeds
          app_mode: window.__gradio_mode__ === "app"
        }
      });
      if (this.updating) {
        this.setAttribute(this.updating.name, this.updating.value);
      }
      this.loading = false;
    }
    static get observedAttributes() {
      return ["src", "space", "host"];
    }
    async attributeChangedCallback(name, old_val, new_val) {
      await pending;
      if ((name === "host" || name === "space" || name === "src") && new_val !== old_val) {
        this.updating = { name, value: new_val };
        if (this.loading)
          return;
        if (this.app) {
          this.app.$destroy();
        }
        this.space = null;
        this.host = null;
        this.src = null;
        if (name === "host") {
          this.host = new_val;
        } else if (name === "space") {
          this.space = new_val;
        } else if (name === "src") {
          this.src = new_val;
        }
        this.app = new IndexComponent({
          target: this,
          props: {
            // embed source
            space: this.space ? this.space.trim() : this.space,
            src: this.src ? this.src.trim() : this.src,
            host: this.host ? this.host.trim() : this.host,
            // embed info
            info: this.info === "false" ? false : true,
            container: this.container === "false" ? false : true,
            is_embed: this.is_embed === "false" ? false : true,
            initial_height: this.initial_height,
            eager: this.eager === "true" ? true : false,
            // gradio meta info
            version: "4-12-0",
            theme_mode: this.theme_mode,
            // misc global behaviour
            autoscroll: this.autoscroll === "true" ? true : false,
            control_page_title: this.control_page_title === "true" ? true : false,
            // injectables
            client,
            upload_files,
            // for gradio docs
            // TODO: Remove -- i think this is just for autoscroll behavhiour, app vs embeds
            app_mode: window.__gradio_mode__ === "app"
          }
        });
        this.updating = false;
      }
    }
  }
  if (!customElements.get("gradio-app"))
    customElements.define("gradio-app", GradioApp);
}
create_custom_element();

export { __vitePreload as _, prefix_css as a, get_fetchable_url_or_file as g, mount_css as m, normalise_file as n, prepare_files as p, upload as u };
//# sourceMappingURL=index-9dc32a9d.js.map
