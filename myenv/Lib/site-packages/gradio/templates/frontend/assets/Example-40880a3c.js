import { r as resolve_wasm_src } from './file-url-a54881a3.js';

const ERROR_RESPONSE_BODY_READER = new Error("failed to get response body reader");
const ERROR_INCOMPLETED_DOWNLOAD = new Error("failed to complete download");

const HeaderContentLength = "Content-Length";

/**
 * Download content of a URL with progress.
 *
 * Progress only works when Content-Length is provided by the server.
 *
 */
const downloadWithProgress = async (url, cb) => {
    const resp = await fetch(url);
    let buf;
    try {
        // Set total to -1 to indicate that there is not Content-Type Header.
        const total = parseInt(resp.headers.get(HeaderContentLength) || "-1");
        const reader = resp.body?.getReader();
        if (!reader)
            throw ERROR_RESPONSE_BODY_READER;
        const chunks = [];
        let received = 0;
        for (;;) {
            const { done, value } = await reader.read();
            const delta = value ? value.length : 0;
            if (done) {
                if (total != -1 && total !== received)
                    throw ERROR_INCOMPLETED_DOWNLOAD;
                cb && cb({ url, total, received, delta, done });
                break;
            }
            chunks.push(value);
            received += delta;
            cb && cb({ url, total, received, delta, done });
        }
        const data = new Uint8Array(received);
        let position = 0;
        for (const chunk of chunks) {
            data.set(chunk, position);
            position += chunk.length;
        }
        buf = data.buffer;
    }
    catch (e) {
        console.log(`failed to send download progress event: `, e);
        // Fetch arrayBuffer directly when it is not possible to get progress.
        buf = await resp.arrayBuffer();
        cb &&
            cb({
                url,
                total: buf.byteLength,
                received: buf.byteLength,
                delta: 0,
                done: true,
            });
    }
    return buf;
};
/**
 * toBlobURL fetches data from an URL and return a blob URL.
 *
 * Example:
 *
 * ```ts
 * await toBlobURL("http://localhost:3000/ffmpeg.js", "text/javascript");
 * ```
 */
const toBlobURL = async (url, mimeType, progress = false, cb) => {
    const buf = progress
        ? await downloadWithProgress(url, cb)
        : await (await fetch(url)).arrayBuffer();
    const blob = new Blob([buf], { type: mimeType });
    return URL.createObjectURL(blob);
};

var FFMessageType;
(function (FFMessageType) {
    FFMessageType["LOAD"] = "LOAD";
    FFMessageType["EXEC"] = "EXEC";
    FFMessageType["WRITE_FILE"] = "WRITE_FILE";
    FFMessageType["READ_FILE"] = "READ_FILE";
    FFMessageType["DELETE_FILE"] = "DELETE_FILE";
    FFMessageType["RENAME"] = "RENAME";
    FFMessageType["CREATE_DIR"] = "CREATE_DIR";
    FFMessageType["LIST_DIR"] = "LIST_DIR";
    FFMessageType["DELETE_DIR"] = "DELETE_DIR";
    FFMessageType["ERROR"] = "ERROR";
    FFMessageType["DOWNLOAD"] = "DOWNLOAD";
    FFMessageType["PROGRESS"] = "PROGRESS";
    FFMessageType["LOG"] = "LOG";
    FFMessageType["MOUNT"] = "MOUNT";
    FFMessageType["UNMOUNT"] = "UNMOUNT";
})(FFMessageType || (FFMessageType = {}));

/**
 * Generate an unique message ID.
 */
const getMessageID = (() => {
    let messageID = 0;
    return () => messageID++;
})();

const ERROR_NOT_LOADED = new Error("ffmpeg is not loaded, call `await ffmpeg.load()` first");
const ERROR_TERMINATED = new Error("called FFmpeg.terminate()");

/**
 * Provides APIs to interact with ffmpeg web worker.
 *
 * @example
 * ```ts
 * const ffmpeg = new FFmpeg();
 * ```
 */
class FFmpeg {
    #worker = null;
    /**
     * #resolves and #rejects tracks Promise resolves and rejects to
     * be called when we receive message from web worker.
     */
    #resolves = {};
    #rejects = {};
    #logEventCallbacks = [];
    #progressEventCallbacks = [];
    loaded = false;
    /**
     * register worker message event handlers.
     */
    #registerHandlers = () => {
        if (this.#worker) {
            this.#worker.onmessage = ({ data: { id, type, data }, }) => {
                switch (type) {
                    case FFMessageType.LOAD:
                        this.loaded = true;
                        this.#resolves[id](data);
                        break;
                    case FFMessageType.MOUNT:
                    case FFMessageType.UNMOUNT:
                    case FFMessageType.EXEC:
                    case FFMessageType.WRITE_FILE:
                    case FFMessageType.READ_FILE:
                    case FFMessageType.DELETE_FILE:
                    case FFMessageType.RENAME:
                    case FFMessageType.CREATE_DIR:
                    case FFMessageType.LIST_DIR:
                    case FFMessageType.DELETE_DIR:
                        this.#resolves[id](data);
                        break;
                    case FFMessageType.LOG:
                        this.#logEventCallbacks.forEach((f) => f(data));
                        break;
                    case FFMessageType.PROGRESS:
                        this.#progressEventCallbacks.forEach((f) => f(data));
                        break;
                    case FFMessageType.ERROR:
                        this.#rejects[id](data);
                        break;
                }
                delete this.#resolves[id];
                delete this.#rejects[id];
            };
        }
    };
    /**
     * Generic function to send messages to web worker.
     */
    #send = ({ type, data }, trans = [], signal) => {
        if (!this.#worker) {
            return Promise.reject(ERROR_NOT_LOADED);
        }
        return new Promise((resolve, reject) => {
            const id = getMessageID();
            this.#worker && this.#worker.postMessage({ id, type, data }, trans);
            this.#resolves[id] = resolve;
            this.#rejects[id] = reject;
            signal?.addEventListener("abort", () => {
                reject(new DOMException(`Message # ${id} was aborted`, "AbortError"));
            }, { once: true });
        });
    };
    on(event, callback) {
        if (event === "log") {
            this.#logEventCallbacks.push(callback);
        }
        else if (event === "progress") {
            this.#progressEventCallbacks.push(callback);
        }
    }
    off(event, callback) {
        if (event === "log") {
            this.#logEventCallbacks = this.#logEventCallbacks.filter((f) => f !== callback);
        }
        else if (event === "progress") {
            this.#progressEventCallbacks = this.#progressEventCallbacks.filter((f) => f !== callback);
        }
    }
    /**
     * Loads ffmpeg-core inside web worker. It is required to call this method first
     * as it initializes WebAssembly and other essential variables.
     *
     * @category FFmpeg
     * @returns `true` if ffmpeg core is loaded for the first time.
     */
    load = (config = {}, { signal } = {}) => {
        if (!this.#worker) {
            this.#worker = new Worker(new URL(""+new URL('worker-6d6dd1a7.js', import.meta.url).href+"", self.location), {
                type: "module",
            });
            this.#registerHandlers();
        }
        return this.#send({
            type: FFMessageType.LOAD,
            data: config,
        }, undefined, signal);
    };
    /**
     * Execute ffmpeg command.
     *
     * @remarks
     * To avoid common I/O issues, ["-nostdin", "-y"] are prepended to the args
     * by default.
     *
     * @example
     * ```ts
     * const ffmpeg = new FFmpeg();
     * await ffmpeg.load();
     * await ffmpeg.writeFile("video.avi", ...);
     * // ffmpeg -i video.avi video.mp4
     * await ffmpeg.exec(["-i", "video.avi", "video.mp4"]);
     * const data = ffmpeg.readFile("video.mp4");
     * ```
     *
     * @returns `0` if no error, `!= 0` if timeout (1) or error.
     * @category FFmpeg
     */
    exec = (
    /** ffmpeg command line args */
    args, 
    /**
     * milliseconds to wait before stopping the command execution.
     *
     * @defaultValue -1
     */
    timeout = -1, { signal } = {}) => this.#send({
        type: FFMessageType.EXEC,
        data: { args, timeout },
    }, undefined, signal);
    /**
     * Terminate all ongoing API calls and terminate web worker.
     * `FFmpeg.load()` must be called again before calling any other APIs.
     *
     * @category FFmpeg
     */
    terminate = () => {
        const ids = Object.keys(this.#rejects);
        // rejects all incomplete Promises.
        for (const id of ids) {
            this.#rejects[id](ERROR_TERMINATED);
            delete this.#rejects[id];
            delete this.#resolves[id];
        }
        if (this.#worker) {
            this.#worker.terminate();
            this.#worker = null;
            this.loaded = false;
        }
    };
    /**
     * Write data to ffmpeg.wasm.
     *
     * @example
     * ```ts
     * const ffmpeg = new FFmpeg();
     * await ffmpeg.load();
     * await ffmpeg.writeFile("video.avi", await fetchFile("../video.avi"));
     * await ffmpeg.writeFile("text.txt", "hello world");
     * ```
     *
     * @category File System
     */
    writeFile = (path, data, { signal } = {}) => {
        const trans = [];
        if (data instanceof Uint8Array) {
            trans.push(data.buffer);
        }
        return this.#send({
            type: FFMessageType.WRITE_FILE,
            data: { path, data },
        }, trans, signal);
    };
    mount = (fsType, options, mountPoint) => {
        const trans = [];
        return this.#send({
            type: FFMessageType.MOUNT,
            data: { fsType, options, mountPoint },
        }, trans);
    };
    unmount = (mountPoint) => {
        const trans = [];
        return this.#send({
            type: FFMessageType.UNMOUNT,
            data: { mountPoint },
        }, trans);
    };
    /**
     * Read data from ffmpeg.wasm.
     *
     * @example
     * ```ts
     * const ffmpeg = new FFmpeg();
     * await ffmpeg.load();
     * const data = await ffmpeg.readFile("video.mp4");
     * ```
     *
     * @category File System
     */
    readFile = (path, 
    /**
     * File content encoding, supports two encodings:
     * - utf8: read file as text file, return data in string type.
     * - binary: read file as binary file, return data in Uint8Array type.
     *
     * @defaultValue binary
     */
    encoding = "binary", { signal } = {}) => this.#send({
        type: FFMessageType.READ_FILE,
        data: { path, encoding },
    }, undefined, signal);
    /**
     * Delete a file.
     *
     * @category File System
     */
    deleteFile = (path, { signal } = {}) => this.#send({
        type: FFMessageType.DELETE_FILE,
        data: { path },
    }, undefined, signal);
    /**
     * Rename a file or directory.
     *
     * @category File System
     */
    rename = (oldPath, newPath, { signal } = {}) => this.#send({
        type: FFMessageType.RENAME,
        data: { oldPath, newPath },
    }, undefined, signal);
    /**
     * Create a directory.
     *
     * @category File System
     */
    createDir = (path, { signal } = {}) => this.#send({
        type: FFMessageType.CREATE_DIR,
        data: { path },
    }, undefined, signal);
    /**
     * List directory contents.
     *
     * @category File System
     */
    listDir = (path, { signal } = {}) => this.#send({
        type: FFMessageType.LIST_DIR,
        data: { path },
    }, undefined, signal);
    /**
     * Delete an empty directory.
     *
     * @category File System
     */
    deleteDir = (path, { signal } = {}) => this.#send({
        type: FFMessageType.DELETE_DIR,
        data: { path },
    }, undefined, signal);
}

const mimes = {
  "3g2": "video/3gpp2",
  "3gp": "video/3gpp",
  "3gpp": "video/3gpp",
  "3mf": "model/3mf",
  "aac": "audio/aac",
  "ac": "application/pkix-attr-cert",
  "adp": "audio/adpcm",
  "adts": "audio/aac",
  "ai": "application/postscript",
  "aml": "application/automationml-aml+xml",
  "amlx": "application/automationml-amlx+zip",
  "amr": "audio/amr",
  "apng": "image/apng",
  "appcache": "text/cache-manifest",
  "appinstaller": "application/appinstaller",
  "appx": "application/appx",
  "appxbundle": "application/appxbundle",
  "asc": "application/pgp-keys",
  "atom": "application/atom+xml",
  "atomcat": "application/atomcat+xml",
  "atomdeleted": "application/atomdeleted+xml",
  "atomsvc": "application/atomsvc+xml",
  "au": "audio/basic",
  "avci": "image/avci",
  "avcs": "image/avcs",
  "avif": "image/avif",
  "aw": "application/applixware",
  "bdoc": "application/bdoc",
  "bin": "application/octet-stream",
  "bmp": "image/bmp",
  "bpk": "application/octet-stream",
  "btf": "image/prs.btif",
  "btif": "image/prs.btif",
  "buffer": "application/octet-stream",
  "ccxml": "application/ccxml+xml",
  "cdfx": "application/cdfx+xml",
  "cdmia": "application/cdmi-capability",
  "cdmic": "application/cdmi-container",
  "cdmid": "application/cdmi-domain",
  "cdmio": "application/cdmi-object",
  "cdmiq": "application/cdmi-queue",
  "cer": "application/pkix-cert",
  "cgm": "image/cgm",
  "cjs": "application/node",
  "class": "application/java-vm",
  "coffee": "text/coffeescript",
  "conf": "text/plain",
  "cpl": "application/cpl+xml",
  "cpt": "application/mac-compactpro",
  "crl": "application/pkix-crl",
  "css": "text/css",
  "csv": "text/csv",
  "cu": "application/cu-seeme",
  "cwl": "application/cwl",
  "cww": "application/prs.cww",
  "davmount": "application/davmount+xml",
  "dbk": "application/docbook+xml",
  "deb": "application/octet-stream",
  "def": "text/plain",
  "deploy": "application/octet-stream",
  "dib": "image/bmp",
  "disposition-notification": "message/disposition-notification",
  "dist": "application/octet-stream",
  "distz": "application/octet-stream",
  "dll": "application/octet-stream",
  "dmg": "application/octet-stream",
  "dms": "application/octet-stream",
  "doc": "application/msword",
  "dot": "application/msword",
  "dpx": "image/dpx",
  "drle": "image/dicom-rle",
  "dsc": "text/prs.lines.tag",
  "dssc": "application/dssc+der",
  "dtd": "application/xml-dtd",
  "dump": "application/octet-stream",
  "dwd": "application/atsc-dwd+xml",
  "ear": "application/java-archive",
  "ecma": "application/ecmascript",
  "elc": "application/octet-stream",
  "emf": "image/emf",
  "eml": "message/rfc822",
  "emma": "application/emma+xml",
  "emotionml": "application/emotionml+xml",
  "eps": "application/postscript",
  "epub": "application/epub+zip",
  "exe": "application/octet-stream",
  "exi": "application/exi",
  "exp": "application/express",
  "exr": "image/aces",
  "ez": "application/andrew-inset",
  "fdf": "application/fdf",
  "fdt": "application/fdt+xml",
  "fits": "image/fits",
  "g3": "image/g3fax",
  "gbr": "application/rpki-ghostbusters",
  "geojson": "application/geo+json",
  "gif": "image/gif",
  "glb": "model/gltf-binary",
  "gltf": "model/gltf+json",
  "gml": "application/gml+xml",
  "gpx": "application/gpx+xml",
  "gram": "application/srgs",
  "grxml": "application/srgs+xml",
  "gxf": "application/gxf",
  "gz": "application/gzip",
  "h261": "video/h261",
  "h263": "video/h263",
  "h264": "video/h264",
  "heic": "image/heic",
  "heics": "image/heic-sequence",
  "heif": "image/heif",
  "heifs": "image/heif-sequence",
  "hej2": "image/hej2k",
  "held": "application/atsc-held+xml",
  "hjson": "application/hjson",
  "hlp": "application/winhlp",
  "hqx": "application/mac-binhex40",
  "hsj2": "image/hsj2",
  "htm": "text/html",
  "html": "text/html",
  "ics": "text/calendar",
  "ief": "image/ief",
  "ifb": "text/calendar",
  "iges": "model/iges",
  "igs": "model/iges",
  "img": "application/octet-stream",
  "in": "text/plain",
  "ini": "text/plain",
  "ink": "application/inkml+xml",
  "inkml": "application/inkml+xml",
  "ipfix": "application/ipfix",
  "iso": "application/octet-stream",
  "its": "application/its+xml",
  "jade": "text/jade",
  "jar": "application/java-archive",
  "jhc": "image/jphc",
  "jls": "image/jls",
  "jp2": "image/jp2",
  "jpe": "image/jpeg",
  "jpeg": "image/jpeg",
  "jpf": "image/jpx",
  "jpg": "image/jpeg",
  "jpg2": "image/jp2",
  "jpgm": "image/jpm",
  "jpgv": "video/jpeg",
  "jph": "image/jph",
  "jpm": "image/jpm",
  "jpx": "image/jpx",
  "js": "text/javascript",
  "json": "application/json",
  "json5": "application/json5",
  "jsonld": "application/ld+json",
  "jsonml": "application/jsonml+json",
  "jsx": "text/jsx",
  "jt": "model/jt",
  "jxr": "image/jxr",
  "jxra": "image/jxra",
  "jxrs": "image/jxrs",
  "jxs": "image/jxs",
  "jxsc": "image/jxsc",
  "jxsi": "image/jxsi",
  "jxss": "image/jxss",
  "kar": "audio/midi",
  "ktx": "image/ktx",
  "ktx2": "image/ktx2",
  "less": "text/less",
  "lgr": "application/lgr+xml",
  "list": "text/plain",
  "litcoffee": "text/coffeescript",
  "log": "text/plain",
  "lostxml": "application/lost+xml",
  "lrf": "application/octet-stream",
  "m1v": "video/mpeg",
  "m21": "application/mp21",
  "m2a": "audio/mpeg",
  "m2v": "video/mpeg",
  "m3a": "audio/mpeg",
  "m4a": "audio/mp4",
  "m4p": "application/mp4",
  "m4s": "video/iso.segment",
  "ma": "application/mathematica",
  "mads": "application/mads+xml",
  "maei": "application/mmt-aei+xml",
  "man": "text/troff",
  "manifest": "text/cache-manifest",
  "map": "application/json",
  "mar": "application/octet-stream",
  "markdown": "text/markdown",
  "mathml": "application/mathml+xml",
  "mb": "application/mathematica",
  "mbox": "application/mbox",
  "md": "text/markdown",
  "mdx": "text/mdx",
  "me": "text/troff",
  "mesh": "model/mesh",
  "meta4": "application/metalink4+xml",
  "metalink": "application/metalink+xml",
  "mets": "application/mets+xml",
  "mft": "application/rpki-manifest",
  "mid": "audio/midi",
  "midi": "audio/midi",
  "mime": "message/rfc822",
  "mj2": "video/mj2",
  "mjp2": "video/mj2",
  "mjs": "text/javascript",
  "mml": "text/mathml",
  "mods": "application/mods+xml",
  "mov": "video/quicktime",
  "mp2": "audio/mpeg",
  "mp21": "application/mp21",
  "mp2a": "audio/mpeg",
  "mp3": "audio/mpeg",
  "mp4": "video/mp4",
  "mp4a": "audio/mp4",
  "mp4s": "application/mp4",
  "mp4v": "video/mp4",
  "mpd": "application/dash+xml",
  "mpe": "video/mpeg",
  "mpeg": "video/mpeg",
  "mpf": "application/media-policy-dataset+xml",
  "mpg": "video/mpeg",
  "mpg4": "video/mp4",
  "mpga": "audio/mpeg",
  "mpp": "application/dash-patch+xml",
  "mrc": "application/marc",
  "mrcx": "application/marcxml+xml",
  "ms": "text/troff",
  "mscml": "application/mediaservercontrol+xml",
  "msh": "model/mesh",
  "msi": "application/octet-stream",
  "msix": "application/msix",
  "msixbundle": "application/msixbundle",
  "msm": "application/octet-stream",
  "msp": "application/octet-stream",
  "mtl": "model/mtl",
  "musd": "application/mmt-usd+xml",
  "mxf": "application/mxf",
  "mxmf": "audio/mobile-xmf",
  "mxml": "application/xv+xml",
  "n3": "text/n3",
  "nb": "application/mathematica",
  "nq": "application/n-quads",
  "nt": "application/n-triples",
  "obj": "model/obj",
  "oda": "application/oda",
  "oga": "audio/ogg",
  "ogg": "audio/ogg",
  "ogv": "video/ogg",
  "ogx": "application/ogg",
  "omdoc": "application/omdoc+xml",
  "onepkg": "application/onenote",
  "onetmp": "application/onenote",
  "onetoc": "application/onenote",
  "onetoc2": "application/onenote",
  "opf": "application/oebps-package+xml",
  "opus": "audio/ogg",
  "otf": "font/otf",
  "owl": "application/rdf+xml",
  "oxps": "application/oxps",
  "p10": "application/pkcs10",
  "p7c": "application/pkcs7-mime",
  "p7m": "application/pkcs7-mime",
  "p7s": "application/pkcs7-signature",
  "p8": "application/pkcs8",
  "pdf": "application/pdf",
  "pfr": "application/font-tdpfr",
  "pgp": "application/pgp-encrypted",
  "pkg": "application/octet-stream",
  "pki": "application/pkixcmp",
  "pkipath": "application/pkix-pkipath",
  "pls": "application/pls+xml",
  "png": "image/png",
  "prc": "model/prc",
  "prf": "application/pics-rules",
  "provx": "application/provenance+xml",
  "ps": "application/postscript",
  "pskcxml": "application/pskc+xml",
  "pti": "image/prs.pti",
  "qt": "video/quicktime",
  "raml": "application/raml+yaml",
  "rapd": "application/route-apd+xml",
  "rdf": "application/rdf+xml",
  "relo": "application/p2p-overlay+xml",
  "rif": "application/reginfo+xml",
  "rl": "application/resource-lists+xml",
  "rld": "application/resource-lists-diff+xml",
  "rmi": "audio/midi",
  "rnc": "application/relax-ng-compact-syntax",
  "rng": "application/xml",
  "roa": "application/rpki-roa",
  "roff": "text/troff",
  "rq": "application/sparql-query",
  "rs": "application/rls-services+xml",
  "rsat": "application/atsc-rsat+xml",
  "rsd": "application/rsd+xml",
  "rsheet": "application/urc-ressheet+xml",
  "rss": "application/rss+xml",
  "rtf": "text/rtf",
  "rtx": "text/richtext",
  "rusd": "application/route-usd+xml",
  "s3m": "audio/s3m",
  "sbml": "application/sbml+xml",
  "scq": "application/scvp-cv-request",
  "scs": "application/scvp-cv-response",
  "sdp": "application/sdp",
  "senmlx": "application/senml+xml",
  "sensmlx": "application/sensml+xml",
  "ser": "application/java-serialized-object",
  "setpay": "application/set-payment-initiation",
  "setreg": "application/set-registration-initiation",
  "sgi": "image/sgi",
  "sgm": "text/sgml",
  "sgml": "text/sgml",
  "shex": "text/shex",
  "shf": "application/shf+xml",
  "shtml": "text/html",
  "sieve": "application/sieve",
  "sig": "application/pgp-signature",
  "sil": "audio/silk",
  "silo": "model/mesh",
  "siv": "application/sieve",
  "slim": "text/slim",
  "slm": "text/slim",
  "sls": "application/route-s-tsid+xml",
  "smi": "application/smil+xml",
  "smil": "application/smil+xml",
  "snd": "audio/basic",
  "so": "application/octet-stream",
  "spdx": "text/spdx",
  "spp": "application/scvp-vp-response",
  "spq": "application/scvp-vp-request",
  "spx": "audio/ogg",
  "sql": "application/sql",
  "sru": "application/sru+xml",
  "srx": "application/sparql-results+xml",
  "ssdl": "application/ssdl+xml",
  "ssml": "application/ssml+xml",
  "stk": "application/hyperstudio",
  "stl": "model/stl",
  "stpx": "model/step+xml",
  "stpxz": "model/step-xml+zip",
  "stpz": "model/step+zip",
  "styl": "text/stylus",
  "stylus": "text/stylus",
  "svg": "image/svg+xml",
  "svgz": "image/svg+xml",
  "swidtag": "application/swid+xml",
  "t": "text/troff",
  "t38": "image/t38",
  "td": "application/urc-targetdesc+xml",
  "tei": "application/tei+xml",
  "teicorpus": "application/tei+xml",
  "text": "text/plain",
  "tfi": "application/thraud+xml",
  "tfx": "image/tiff-fx",
  "tif": "image/tiff",
  "tiff": "image/tiff",
  "toml": "application/toml",
  "tr": "text/troff",
  "trig": "application/trig",
  "ts": "video/mp2t",
  "tsd": "application/timestamped-data",
  "tsv": "text/tab-separated-values",
  "ttc": "font/collection",
  "ttf": "font/ttf",
  "ttl": "text/turtle",
  "ttml": "application/ttml+xml",
  "txt": "text/plain",
  "u3d": "model/u3d",
  "u8dsn": "message/global-delivery-status",
  "u8hdr": "message/global-headers",
  "u8mdn": "message/global-disposition-notification",
  "u8msg": "message/global",
  "ubj": "application/ubjson",
  "uri": "text/uri-list",
  "uris": "text/uri-list",
  "urls": "text/uri-list",
  "vcard": "text/vcard",
  "vrml": "model/vrml",
  "vtt": "text/vtt",
  "vxml": "application/voicexml+xml",
  "war": "application/java-archive",
  "wasm": "application/wasm",
  "wav": "audio/wav",
  "weba": "audio/webm",
  "webm": "video/webm",
  "webmanifest": "application/manifest+json",
  "webp": "image/webp",
  "wgsl": "text/wgsl",
  "wgt": "application/widget",
  "wif": "application/watcherinfo+xml",
  "wmf": "image/wmf",
  "woff": "font/woff",
  "woff2": "font/woff2",
  "wrl": "model/vrml",
  "wsdl": "application/wsdl+xml",
  "wspolicy": "application/wspolicy+xml",
  "x3d": "model/x3d+xml",
  "x3db": "model/x3d+fastinfoset",
  "x3dbz": "model/x3d+binary",
  "x3dv": "model/x3d-vrml",
  "x3dvz": "model/x3d+vrml",
  "x3dz": "model/x3d+xml",
  "xaml": "application/xaml+xml",
  "xav": "application/xcap-att+xml",
  "xca": "application/xcap-caps+xml",
  "xcs": "application/calendar+xml",
  "xdf": "application/xcap-diff+xml",
  "xdssc": "application/dssc+xml",
  "xel": "application/xcap-el+xml",
  "xenc": "application/xenc+xml",
  "xer": "application/patch-ops-error+xml",
  "xfdf": "application/xfdf",
  "xht": "application/xhtml+xml",
  "xhtml": "application/xhtml+xml",
  "xhvml": "application/xv+xml",
  "xlf": "application/xliff+xml",
  "xm": "audio/xm",
  "xml": "text/xml",
  "xns": "application/xcap-ns+xml",
  "xop": "application/xop+xml",
  "xpl": "application/xproc+xml",
  "xsd": "application/xml",
  "xsf": "application/prs.xsf+xml",
  "xsl": "application/xml",
  "xslt": "application/xml",
  "xspf": "application/xspf+xml",
  "xvm": "application/xv+xml",
  "xvml": "application/xv+xml",
  "yaml": "text/yaml",
  "yang": "application/yang",
  "yin": "application/yin+xml",
  "yml": "text/yaml",
  "zip": "application/zip"
};

function lookup(extn) {
	let tmp = ('' + extn).trim().toLowerCase();
	let idx = tmp.lastIndexOf('.');
	return mimes[!~idx ? tmp : tmp.substring(++idx)];
}

const prettyBytes = (bytes) => {
  let units = ["B", "KB", "MB", "GB", "PB"];
  let i = 0;
  while (bytes > 1024) {
    bytes /= 1024;
    i++;
  }
  let unit = units[i];
  return bytes.toFixed(1) + " " + unit;
};
const playable = () => {
  return true;
};
function loaded(node, { autoplay }) {
  async function handle_playback() {
    if (!autoplay)
      return;
    await node.play();
  }
  node.addEventListener("loadeddata", handle_playback);
  return {
    destroy() {
      node.removeEventListener("loadeddata", handle_playback);
    }
  };
}
async function loadFfmpeg() {
  const ffmpeg = new FFmpeg();
  const baseURL = "https://unpkg.com/@ffmpeg/core@0.12.4/dist/esm";
  await ffmpeg.load({
    coreURL: await toBlobURL(`${baseURL}/ffmpeg-core.js`, "text/javascript"),
    wasmURL: await toBlobURL(`${baseURL}/ffmpeg-core.wasm`, "application/wasm")
  });
  return ffmpeg;
}
async function trimVideo(ffmpeg, startTime, endTime, videoElement) {
  const videoUrl = videoElement.src;
  const mimeType = lookup(videoElement.src) || "video/mp4";
  const blobUrl = await toBlobURL(videoUrl, mimeType);
  const response = await fetch(blobUrl);
  const vidBlob = await response.blob();
  const type = getVideoExtensionFromMimeType(mimeType) || "mp4";
  const inputName = `input.${type}`;
  const outputName = `output.${type}`;
  try {
    if (startTime === 0 && endTime === 0) {
      return vidBlob;
    }
    await ffmpeg.writeFile(
      inputName,
      new Uint8Array(await vidBlob.arrayBuffer())
    );
    let command = [
      "-i",
      inputName,
      ...startTime !== 0 ? ["-ss", startTime.toString()] : [],
      ...endTime !== 0 ? ["-to", endTime.toString()] : [],
      "-c:a",
      "copy",
      outputName
    ];
    await ffmpeg.exec(command);
    const outputData = await ffmpeg.readFile(outputName);
    const outputBlob = new Blob([outputData], {
      type: `video/${type}`
    });
    return outputBlob;
  } catch (error) {
    console.error("Error initializing FFmpeg:", error);
    return vidBlob;
  }
}
const getVideoExtensionFromMimeType = (mimeType) => {
  const videoMimeToExtensionMap = {
    "video/mp4": "mp4",
    "video/webm": "webm",
    "video/ogg": "ogv",
    "video/quicktime": "mov",
    "video/x-msvideo": "avi",
    "video/x-matroska": "mkv",
    "video/mpeg": "mpeg",
    "video/3gpp": "3gp",
    "video/3gpp2": "3g2",
    "video/h261": "h261",
    "video/h263": "h263",
    "video/h264": "h264",
    "video/jpeg": "jpgv",
    "video/jpm": "jpm",
    "video/mj2": "mj2",
    "video/mpv": "mpv",
    "video/vnd.ms-playready.media.pyv": "pyv",
    "video/vnd.uvvu.mp4": "uvu",
    "video/vnd.vivo": "viv",
    "video/x-f4v": "f4v",
    "video/x-fli": "fli",
    "video/x-flv": "flv",
    "video/x-m4v": "m4v",
    "video/x-ms-asf": "asf",
    "video/x-ms-wm": "wm",
    "video/x-ms-wmv": "wmv",
    "video/x-ms-wmx": "wmx",
    "video/x-ms-wvx": "wvx",
    "video/x-sgi-movie": "movie",
    "video/x-smv": "smv"
  };
  return videoMimeToExtensionMap[mimeType] || null;
};

const Video_svelte_svelte_type_style_lang = '';

/* home/runner/work/gradio/gradio/js/video/shared/Video.svelte generated by Svelte v4.2.2 */
const {
	SvelteComponent: SvelteComponent$1,
	action_destroyer,
	add_render_callback,
	assign,
	attr: attr$1,
	binding_callbacks: binding_callbacks$1,
	create_slot,
	detach: detach$1,
	element: element$1,
	exclude_internal_props,
	get_all_dirty_from_scope,
	get_slot_changes,
	init,
	insert: insert$1,
	is_function: is_function$1,
	listen,
	raf,
	run_all,
	safe_not_equal: safe_not_equal$1,
	space,
	src_url_equal,
	toggle_class: toggle_class$1,
	transition_in: transition_in$1,
	transition_out: transition_out$1,
	update_slot_base
} = window.__gradio__svelte__internal;
const { createEventDispatcher } = window.__gradio__svelte__internal;
function create_fragment$1(ctx) {
	let div;
	let t;
	let video;
	let video_src_value;
	let video_data_testid_value;
	let video_updating = false;
	let video_animationframe;
	let video_is_paused = true;
	let loaded_action;
	let current;
	let mounted;
	let dispose;
	const default_slot_template = /*#slots*/ ctx[16].default;
	const default_slot = create_slot(default_slot_template, ctx, /*$$scope*/ ctx[15], null);

	function video_timeupdate_handler() {
		cancelAnimationFrame(video_animationframe);

		if (!video.paused) {
			video_animationframe = raf(video_timeupdate_handler);
			video_updating = true;
		}

		/*video_timeupdate_handler*/ ctx[17].call(video);
	}

	return {
		c() {
			div = element$1("div");
			div.innerHTML = `<span class="load-wrap svelte-1y0s5gv"><span class="loader svelte-1y0s5gv"></span></span>`;
			t = space();
			video = element$1("video");
			if (default_slot) default_slot.c();
			attr$1(div, "class", "overlay svelte-1y0s5gv");
			toggle_class$1(div, "hidden", !/*processingVideo*/ ctx[9]);
			if (!src_url_equal(video.src, video_src_value = /*resolved_src*/ ctx[10])) attr$1(video, "src", video_src_value);
			video.muted = /*muted*/ ctx[4];
			video.playsInline = /*playsinline*/ ctx[5];
			attr$1(video, "preload", /*preload*/ ctx[6]);
			video.autoplay = /*autoplay*/ ctx[7];
			video.controls = /*controls*/ ctx[8];
			attr$1(video, "data-testid", video_data_testid_value = /*$$props*/ ctx[12]["data-testid"]);
			attr$1(video, "crossorigin", "anonymous");
			if (/*duration*/ ctx[1] === void 0) add_render_callback(() => /*video_durationchange_handler*/ ctx[18].call(video));
		},
		m(target, anchor) {
			insert$1(target, div, anchor);
			insert$1(target, t, anchor);
			insert$1(target, video, anchor);

			if (default_slot) {
				default_slot.m(video, null);
			}

			/*video_binding*/ ctx[20](video);
			current = true;

			if (!mounted) {
				dispose = [
					listen(video, "loadeddata", /*dispatch*/ ctx[11].bind(null, "loadeddata")),
					listen(video, "click", /*dispatch*/ ctx[11].bind(null, "click")),
					listen(video, "play", /*dispatch*/ ctx[11].bind(null, "play")),
					listen(video, "pause", /*dispatch*/ ctx[11].bind(null, "pause")),
					listen(video, "ended", /*dispatch*/ ctx[11].bind(null, "ended")),
					listen(video, "mouseover", /*dispatch*/ ctx[11].bind(null, "mouseover")),
					listen(video, "mouseout", /*dispatch*/ ctx[11].bind(null, "mouseout")),
					listen(video, "focus", /*dispatch*/ ctx[11].bind(null, "focus")),
					listen(video, "blur", /*dispatch*/ ctx[11].bind(null, "blur")),
					listen(video, "timeupdate", video_timeupdate_handler),
					listen(video, "durationchange", /*video_durationchange_handler*/ ctx[18]),
					listen(video, "play", /*video_play_pause_handler*/ ctx[19]),
					listen(video, "pause", /*video_play_pause_handler*/ ctx[19]),
					action_destroyer(loaded_action = loaded.call(null, video, { autoplay: /*autoplay*/ ctx[7] ?? false }))
				];

				mounted = true;
			}
		},
		p(ctx, [dirty]) {
			if (!current || dirty & /*processingVideo*/ 512) {
				toggle_class$1(div, "hidden", !/*processingVideo*/ ctx[9]);
			}

			if (default_slot) {
				if (default_slot.p && (!current || dirty & /*$$scope*/ 32768)) {
					update_slot_base(
						default_slot,
						default_slot_template,
						ctx,
						/*$$scope*/ ctx[15],
						!current
						? get_all_dirty_from_scope(/*$$scope*/ ctx[15])
						: get_slot_changes(default_slot_template, /*$$scope*/ ctx[15], dirty, null),
						null
					);
				}
			}

			if (!current || dirty & /*resolved_src*/ 1024 && !src_url_equal(video.src, video_src_value = /*resolved_src*/ ctx[10])) {
				attr$1(video, "src", video_src_value);
			}

			if (!current || dirty & /*muted*/ 16) {
				video.muted = /*muted*/ ctx[4];
			}

			if (!current || dirty & /*playsinline*/ 32) {
				video.playsInline = /*playsinline*/ ctx[5];
			}

			if (!current || dirty & /*preload*/ 64) {
				attr$1(video, "preload", /*preload*/ ctx[6]);
			}

			if (!current || dirty & /*autoplay*/ 128) {
				video.autoplay = /*autoplay*/ ctx[7];
			}

			if (!current || dirty & /*controls*/ 256) {
				video.controls = /*controls*/ ctx[8];
			}

			if (!current || dirty & /*$$props*/ 4096 && video_data_testid_value !== (video_data_testid_value = /*$$props*/ ctx[12]["data-testid"])) {
				attr$1(video, "data-testid", video_data_testid_value);
			}

			if (!video_updating && dirty & /*currentTime*/ 1 && !isNaN(/*currentTime*/ ctx[0])) {
				video.currentTime = /*currentTime*/ ctx[0];
			}

			video_updating = false;

			if (dirty & /*paused*/ 4 && video_is_paused !== (video_is_paused = /*paused*/ ctx[2])) {
				video[video_is_paused ? "pause" : "play"]();
			}

			if (loaded_action && is_function$1(loaded_action.update) && dirty & /*autoplay*/ 128) loaded_action.update.call(null, { autoplay: /*autoplay*/ ctx[7] ?? false });
		},
		i(local) {
			if (current) return;
			transition_in$1(default_slot, local);
			current = true;
		},
		o(local) {
			transition_out$1(default_slot, local);
			current = false;
		},
		d(detaching) {
			if (detaching) {
				detach$1(div);
				detach$1(t);
				detach$1(video);
			}

			if (default_slot) default_slot.d(detaching);
			/*video_binding*/ ctx[20](null);
			mounted = false;
			run_all(dispose);
		}
	};
}

function instance$1($$self, $$props, $$invalidate) {
	let { $$slots: slots = {}, $$scope } = $$props;
	let { src = undefined } = $$props;
	let { muted = undefined } = $$props;
	let { playsinline = undefined } = $$props;
	let { preload = undefined } = $$props;
	let { autoplay = undefined } = $$props;
	let { controls = undefined } = $$props;
	let { currentTime = undefined } = $$props;
	let { duration = undefined } = $$props;
	let { paused = undefined } = $$props;
	let { node = undefined } = $$props;
	let { processingVideo = false } = $$props;
	const dispatch = createEventDispatcher();
	let resolved_src;

	// The `src` prop can be updated before the Promise from `resolve_wasm_src` is resolved.
	// In such a case, the resolved value for the old `src` has to be discarded,
	// This variable `latest_src` is used to pick up only the value resolved for the latest `src` prop.
	let latest_src;

	function video_timeupdate_handler() {
		currentTime = this.currentTime;
		$$invalidate(0, currentTime);
	}

	function video_durationchange_handler() {
		duration = this.duration;
		$$invalidate(1, duration);
	}

	function video_play_pause_handler() {
		paused = this.paused;
		$$invalidate(2, paused);
	}

	function video_binding($$value) {
		binding_callbacks$1[$$value ? 'unshift' : 'push'](() => {
			node = $$value;
			$$invalidate(3, node);
		});
	}

	$$self.$$set = $$new_props => {
		$$invalidate(12, $$props = assign(assign({}, $$props), exclude_internal_props($$new_props)));
		if ('src' in $$new_props) $$invalidate(13, src = $$new_props.src);
		if ('muted' in $$new_props) $$invalidate(4, muted = $$new_props.muted);
		if ('playsinline' in $$new_props) $$invalidate(5, playsinline = $$new_props.playsinline);
		if ('preload' in $$new_props) $$invalidate(6, preload = $$new_props.preload);
		if ('autoplay' in $$new_props) $$invalidate(7, autoplay = $$new_props.autoplay);
		if ('controls' in $$new_props) $$invalidate(8, controls = $$new_props.controls);
		if ('currentTime' in $$new_props) $$invalidate(0, currentTime = $$new_props.currentTime);
		if ('duration' in $$new_props) $$invalidate(1, duration = $$new_props.duration);
		if ('paused' in $$new_props) $$invalidate(2, paused = $$new_props.paused);
		if ('node' in $$new_props) $$invalidate(3, node = $$new_props.node);
		if ('processingVideo' in $$new_props) $$invalidate(9, processingVideo = $$new_props.processingVideo);
		if ('$$scope' in $$new_props) $$invalidate(15, $$scope = $$new_props.$$scope);
	};

	$$self.$$.update = () => {
		if ($$self.$$.dirty & /*src, latest_src*/ 24576) {
			{
				// In normal (non-Wasm) Gradio, the `<img>` element should be rendered with the passed `src` props immediately
				// without waiting for `resolve_wasm_src()` to resolve.
				// If it waits, a black image is displayed until the async task finishes
				// and it leads to undesirable flickering.
				// So set `src` to `resolved_src` here.
				$$invalidate(10, resolved_src = src);

				$$invalidate(14, latest_src = src);
				const resolving_src = src;

				resolve_wasm_src(resolving_src).then(s => {
					if (latest_src === resolving_src) {
						$$invalidate(10, resolved_src = s);
					}
				});
			}
		}
	};

	$$props = exclude_internal_props($$props);

	return [
		currentTime,
		duration,
		paused,
		node,
		muted,
		playsinline,
		preload,
		autoplay,
		controls,
		processingVideo,
		resolved_src,
		dispatch,
		$$props,
		src,
		latest_src,
		$$scope,
		slots,
		video_timeupdate_handler,
		video_durationchange_handler,
		video_play_pause_handler,
		video_binding
	];
}

class Video extends SvelteComponent$1 {
	constructor(options) {
		super();

		init(this, options, instance$1, create_fragment$1, safe_not_equal$1, {
			src: 13,
			muted: 4,
			playsinline: 5,
			preload: 6,
			autoplay: 7,
			controls: 8,
			currentTime: 0,
			duration: 1,
			paused: 2,
			node: 3,
			processingVideo: 9
		});
	}
}

const Example_svelte_svelte_type_style_lang = '';

/* home/runner/work/gradio/gradio/js/video/Example.svelte generated by Svelte v4.2.2 */
const {
	SvelteComponent,
	add_flush_callback,
	append,
	attr,
	bind,
	binding_callbacks,
	create_component,
	destroy_component,
	detach,
	element,
	empty,
	init: init_1,
	insert,
	is_function,
	mount_component,
	noop,
	safe_not_equal,
	set_data,
	text,
	toggle_class,
	transition_in,
	transition_out
} = window.__gradio__svelte__internal;
function create_else_block(ctx) {
	let div;
	let t;

	return {
		c() {
			div = element("div");
			t = text(/*value*/ ctx[2]);
		},
		m(target, anchor) {
			insert(target, div, anchor);
			append(div, t);
		},
		p(ctx, dirty) {
			if (dirty & /*value*/ 4) set_data(t, /*value*/ ctx[2]);
		},
		i: noop,
		o: noop,
		d(detaching) {
			if (detaching) {
				detach(div);
			}
		}
	};
}

// (18:0) {#if playable()}
function create_if_block(ctx) {
	let div;
	let video_1;
	let updating_node;
	let current;

	function video_1_node_binding(value) {
		/*video_1_node_binding*/ ctx[6](value);
	}

	let video_1_props = {
		muted: true,
		playsinline: true,
		src: /*samples_dir*/ ctx[3] + /*value*/ ctx[2]
	};

	if (/*video*/ ctx[4] !== void 0) {
		video_1_props.node = /*video*/ ctx[4];
	}

	video_1 = new Video({ props: video_1_props });
	binding_callbacks.push(() => bind(video_1, 'node', video_1_node_binding));
	video_1.$on("loadeddata", /*init*/ ctx[5]);

	video_1.$on("mouseover", function () {
		if (is_function(/*video*/ ctx[4].play.bind(/*video*/ ctx[4]))) /*video*/ ctx[4].play.bind(/*video*/ ctx[4]).apply(this, arguments);
	});

	video_1.$on("mouseout", function () {
		if (is_function(/*video*/ ctx[4].pause.bind(/*video*/ ctx[4]))) /*video*/ ctx[4].pause.bind(/*video*/ ctx[4]).apply(this, arguments);
	});

	return {
		c() {
			div = element("div");
			create_component(video_1.$$.fragment);
			attr(div, "class", "container svelte-1de9zxs");
			toggle_class(div, "table", /*type*/ ctx[0] === "table");
			toggle_class(div, "gallery", /*type*/ ctx[0] === "gallery");
			toggle_class(div, "selected", /*selected*/ ctx[1]);
		},
		m(target, anchor) {
			insert(target, div, anchor);
			mount_component(video_1, div, null);
			current = true;
		},
		p(new_ctx, dirty) {
			ctx = new_ctx;
			const video_1_changes = {};
			if (dirty & /*samples_dir, value*/ 12) video_1_changes.src = /*samples_dir*/ ctx[3] + /*value*/ ctx[2];

			if (!updating_node && dirty & /*video*/ 16) {
				updating_node = true;
				video_1_changes.node = /*video*/ ctx[4];
				add_flush_callback(() => updating_node = false);
			}

			video_1.$set(video_1_changes);

			if (!current || dirty & /*type*/ 1) {
				toggle_class(div, "table", /*type*/ ctx[0] === "table");
			}

			if (!current || dirty & /*type*/ 1) {
				toggle_class(div, "gallery", /*type*/ ctx[0] === "gallery");
			}

			if (!current || dirty & /*selected*/ 2) {
				toggle_class(div, "selected", /*selected*/ ctx[1]);
			}
		},
		i(local) {
			if (current) return;
			transition_in(video_1.$$.fragment, local);
			current = true;
		},
		o(local) {
			transition_out(video_1.$$.fragment, local);
			current = false;
		},
		d(detaching) {
			if (detaching) {
				detach(div);
			}

			destroy_component(video_1);
		}
	};
}

function create_fragment(ctx) {
	let current_block_type_index;
	let if_block;
	let if_block_anchor;
	let current;
	const if_block_creators = [create_if_block, create_else_block];
	const if_blocks = [];

	function select_block_type(ctx, dirty) {
		return 0;
	}

	current_block_type_index = select_block_type();
	if_block = if_blocks[current_block_type_index] = if_block_creators[current_block_type_index](ctx);

	return {
		c() {
			if_block.c();
			if_block_anchor = empty();
		},
		m(target, anchor) {
			if_blocks[current_block_type_index].m(target, anchor);
			insert(target, if_block_anchor, anchor);
			current = true;
		},
		p(ctx, [dirty]) {
			if_block.p(ctx, dirty);
		},
		i(local) {
			if (current) return;
			transition_in(if_block);
			current = true;
		},
		o(local) {
			transition_out(if_block);
			current = false;
		},
		d(detaching) {
			if (detaching) {
				detach(if_block_anchor);
			}

			if_blocks[current_block_type_index].d(detaching);
		}
	};
}

function instance($$self, $$props, $$invalidate) {
	let { type } = $$props;
	let { selected = false } = $$props;
	let { value } = $$props;
	let { samples_dir } = $$props;
	let video;

	async function init() {
		$$invalidate(4, video.muted = true, video);
		$$invalidate(4, video.playsInline = true, video);
		$$invalidate(4, video.controls = false, video);
		video.setAttribute("muted", "");
		await video.play();
		video.pause();
	}

	function video_1_node_binding(value) {
		video = value;
		$$invalidate(4, video);
	}

	$$self.$$set = $$props => {
		if ('type' in $$props) $$invalidate(0, type = $$props.type);
		if ('selected' in $$props) $$invalidate(1, selected = $$props.selected);
		if ('value' in $$props) $$invalidate(2, value = $$props.value);
		if ('samples_dir' in $$props) $$invalidate(3, samples_dir = $$props.samples_dir);
	};

	return [type, selected, value, samples_dir, video, init, video_1_node_binding];
}

class Example extends SvelteComponent {
	constructor(options) {
		super();

		init_1(this, options, instance, create_fragment, safe_not_equal, {
			type: 0,
			selected: 1,
			value: 2,
			samples_dir: 3
		});
	}
}

const Example$1 = /*#__PURE__*/Object.freeze(/*#__PURE__*/Object.defineProperty({
    __proto__: null,
    default: Example
}, Symbol.toStringTag, { value: 'Module' }));

export { Example as E, Video as V, playable as a, loaded as b, Example$1 as c, loadFfmpeg as l, prettyBytes as p, trimVideo as t };
//# sourceMappingURL=Example-40880a3c.js.map
