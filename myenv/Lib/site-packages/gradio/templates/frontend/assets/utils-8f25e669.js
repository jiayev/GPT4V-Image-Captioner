class ShareError extends Error {
  constructor(message) {
    super(message);
    this.name = "ShareError";
  }
}
async function uploadToHuggingFace(data, type) {
  if (window.__gradio_space__ == null) {
    throw new ShareError("Must be on Spaces to share.");
  }
  let blob;
  let contentType;
  let filename;
  if (type === "url") {
    const response = await fetch(data);
    blob = await response.blob();
    contentType = response.headers.get("content-type") || "";
    filename = response.headers.get("content-disposition") || "";
  } else {
    blob = dataURLtoBlob(data);
    contentType = data.split(";")[0].split(":")[1];
    filename = "file" + contentType.split("/")[1];
  }
  const file = new File([blob], filename, { type: contentType });
  const uploadResponse = await fetch("https://huggingface.co/uploads", {
    method: "POST",
    body: file,
    headers: {
      "Content-Type": file.type,
      "X-Requested-With": "XMLHttpRequest"
    }
  });
  if (!uploadResponse.ok) {
    if (uploadResponse.headers.get("content-type")?.includes("application/json")) {
      const error = await uploadResponse.json();
      throw new ShareError(`Upload failed: ${error.error}`);
    }
    throw new ShareError(`Upload failed.`);
  }
  const result = await uploadResponse.text();
  return result;
}
function dataURLtoBlob(dataurl) {
  var arr = dataurl.split(","), mime = arr[0].match(/:(.*?);/)[1], bstr = atob(arr[1]), n = bstr.length, u8arr = new Uint8Array(n);
  while (n--) {
    u8arr[n] = bstr.charCodeAt(n);
  }
  return new Blob([u8arr], { type: mime });
}
function copy(node) {
  node.addEventListener("click", handle_copy);
  async function handle_copy(event) {
    const path = event.composedPath();
    const [copy_button] = path.filter(
      (e) => e?.tagName === "BUTTON" && e.classList.contains("copy_code_button")
    );
    if (copy_button) {
      let copy_feedback = function(_copy_sucess_button) {
        _copy_sucess_button.style.opacity = "1";
        setTimeout(() => {
          _copy_sucess_button.style.opacity = "0";
        }, 2e3);
      };
      event.stopImmediatePropagation();
      const copy_text = copy_button.parentElement.innerText.trim();
      const copy_sucess_button = Array.from(
        copy_button.children
      )[1];
      const copied = await copy_to_clipboard(copy_text);
      if (copied)
        copy_feedback(copy_sucess_button);
    }
  }
  return {
    destroy() {
      node.removeEventListener("click", handle_copy);
    }
  };
}
async function copy_to_clipboard(value) {
  let copied = false;
  if ("clipboard" in navigator) {
    await navigator.clipboard.writeText(value);
    copied = true;
  } else {
    const textArea = document.createElement("textarea");
    textArea.value = value;
    textArea.style.position = "absolute";
    textArea.style.left = "-999999px";
    document.body.prepend(textArea);
    textArea.select();
    try {
      document.execCommand("copy");
      copied = true;
    } catch (error) {
      console.error(error);
      copied = false;
    } finally {
      textArea.remove();
    }
  }
  return copied;
}

export { ShareError as S, copy as c, uploadToHuggingFace as u };
//# sourceMappingURL=utils-8f25e669.js.map
