/*
 * "Ask Darshan" floating chat widget for any site (e.g. the GitHub Pages portfolio).
 * Usage — add one line before </body>:
 *   <script src="https://YOUR-SPACE.hf.space/static/widget.js" defer></script>
 * The chat runs inside an iframe served by the Space, so no CORS setup is needed.
 */
(() => {
  const script = document.currentScript;
  const origin = new URL(script.src).origin;
  const label = script.dataset.label || "Ask about Darshan";

  const css = `
    .ad-btn{position:fixed;right:20px;bottom:20px;z-index:2147483000;border:0;border-radius:999px;
      background:#0f766e;color:#fff;font:600 14px/1 system-ui,sans-serif;padding:13px 18px;cursor:pointer;
      box-shadow:0 6px 20px rgba(0,0,0,.18)}
    .ad-frame{position:fixed;right:20px;bottom:76px;z-index:2147483000;width:380px;height:560px;
      max-width:calc(100vw - 32px);max-height:calc(100vh - 100px);border:1px solid rgba(0,0,0,.12);
      border-radius:16px;box-shadow:0 12px 40px rgba(0,0,0,.2);background:#fff;display:none}
    .ad-frame.open{display:block}
    @media (max-width:480px){.ad-frame{right:16px;left:16px;width:auto;bottom:72px}}`;
  const style = document.createElement("style");
  style.textContent = css;
  document.head.appendChild(style);

  const frame = document.createElement("iframe");
  frame.className = "ad-frame";
  frame.title = label;
  frame.loading = "lazy";
  const btn = document.createElement("button");
  btn.className = "ad-btn";
  btn.textContent = label;
  btn.setAttribute("aria-expanded", "false");
  btn.addEventListener("click", () => {
    if (!frame.src) frame.src = origin + "/?embed=1";   // load (and wake the Space) on first open
    const open = frame.classList.toggle("open");
    btn.textContent = open ? "Close" : label;
    btn.setAttribute("aria-expanded", String(open));
  });
  document.body.append(frame, btn);
})();
