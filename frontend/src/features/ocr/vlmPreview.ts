/**
 * Builds the sandboxed preview document for experimental VLM OCR output.
 *
 * Model output is untrusted: it is sanitised with DOMPurify, every URL-bearing
 * attribute is removed, and the result is rendered inside an <iframe sandbox="">
 * (no scripts, no same-origin access) with a CSP that blocks all network loads.
 */
import DOMPurify from 'dompurify'

const CSP = "default-src 'none'; style-src 'unsafe-inline'; img-src data:"

export function sanitizeVlmHtml(html: string): string {
  const clean = DOMPurify.sanitize(html, {
    ADD_TAGS: ['chem', 'big'],
    ADD_ATTR: ['data-bbox', 'data-label', 'colspan', 'rowspan', 'display', 'checked', 'align', 'border'],
    FORBID_TAGS: ['style', 'script', 'iframe', 'object', 'embed', 'link', 'meta', 'base', 'form', 'svg', 'video', 'audio'],
    FORBID_ATTR: ['src', 'srcset', 'href', 'action', 'formaction', 'xlink:href', 'background', 'poster'],
  })
  // Chandra describes images in the alt text of <img> tags without a src: show that text instead.
  const doc = new DOMParser().parseFromString(`<body>${clean}</body>`, 'text/html')
  doc.querySelectorAll('img').forEach(img => {
    const placeholder = doc.createElement('div')
    placeholder.className = 'img-alt'
    placeholder.textContent = img.getAttribute('alt') || 'Image'
    img.replaceWith(placeholder)
  })
  doc.querySelectorAll('input').forEach(input => input.setAttribute('disabled', ''))
  return doc.body.innerHTML
}

export function buildPreviewDocument(html: string, opts: { outlines: boolean; dark: boolean }): string {
  const c = opts.dark
    ? { bg: '#0f1b2d', fg: '#dbe9f4', muted: '#8aa4b8', border: '#2c3e55', subtle: '#16263c', accent: '#ff8a5c' }
    : { bg: '#ffffff', fg: '#1a2233', muted: '#667085', border: '#d7dce3', subtle: '#f4f5f7', accent: '#e8590c' }
  const css = `
    :root { color-scheme: ${opts.dark ? 'dark' : 'light'}; }
    body { margin: 0; padding: 22px 26px; background: ${c.bg}; color: ${c.fg};
           font: 14px/1.6 Inter, system-ui, -apple-system, 'Segoe UI', sans-serif; word-wrap: break-word; }
    h1, h2, h3, h4, h5 { line-height: 1.25; margin: 1em 0 .45em; }
    h1 { font-size: 1.5em; } h2 { font-size: 1.3em; } h3 { font-size: 1.15em; } h4, h5 { font-size: 1em; }
    p { margin: .5em 0; }
    ul, ol { padding-left: 1.6em; }
    table { border-collapse: collapse; margin: .7em 0; font-size: 13px; max-width: 100%; }
    td, th { border: 1px solid ${c.border}; padding: 5px 9px; vertical-align: top; }
    th { background: ${c.subtle}; font-weight: 600; }
    math, chem, code, pre { font-family: 'JetBrains Mono', Consolas, monospace; font-size: .9em;
           background: ${c.subtle}; border-radius: 4px; padding: 0 4px; }
    math[display] { display: block; padding: 6px 10px; margin: .6em 0; overflow-x: auto; }
    pre { padding: 10px 12px; overflow-x: auto; }
    hr { border: none; border-top: 1px solid ${c.border}; margin: 1em 0; }
    .img-alt { border: 1px dashed ${c.border}; border-radius: 6px; padding: 8px 12px; margin: .6em 0;
           color: ${c.muted}; font-style: italic; }
    .img-alt::before { content: 'Image description: '; font-style: normal; font-weight: 600; }
    body.outlines [data-label] { position: relative; outline: 1px dashed ${c.border}; outline-offset: 4px;
           margin: 18px 0 12px; border-radius: 2px; }
    body.outlines [data-label]::before { content: attr(data-label); position: absolute; top: -17px; left: -4px;
           font: 600 9.5px/1.4 Inter, system-ui, sans-serif; letter-spacing: .05em; text-transform: uppercase;
           color: ${c.accent}; background: ${c.bg}; padding: 0 4px; }
  `
  return `<!doctype html><html><head><meta charset="utf-8">`
    + `<meta http-equiv="Content-Security-Policy" content="${CSP}">`
    + `<style>${css}</style></head>`
    + `<body class="${opts.outlines ? 'outlines' : ''}">${sanitizeVlmHtml(html)}</body></html>`
}
