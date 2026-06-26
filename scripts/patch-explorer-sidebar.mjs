import fs from "node:fs"

const files = [
  ".quartz/plugins/explorer/dist/index.js",
  ".quartz/plugins/explorer/dist/components/index.js",
]

const replacements = [
  [
    'function N(u,e,D,F,r,t=""){',
    'function N(u,e,D,F,r,x,t=""){',
  ],
  [
    "let C=r[u.slug]!==void 0?r[u.slug]:!0",
    "let C=r[u.slug]!==void 0?r[u.slug]:x",
  ],
  [
    "N(i,o,D,F,r,E)",
    "N(i,o,D,F,r,x,E)",
  ],
  [
    'let r={};try{JSON.parse(localStorage.getItem("fileTree")||"[]").forEach(n=>{r[n.path]=n.collapsed})}catch(t){console.error("[Explorer] Error loading saved state:",t)}for(let t of F){let n=t.querySelector(".explorer-ul");',
    'for(let t of F){let q=t.dataset.savestate!=="false",r={},z=t.dataset.collapsed!=="open";try{q&&JSON.parse(localStorage.getItem("fileTree")||"[]").forEach(n=>{r[n.path]=n.collapsed})}catch(n){console.error("[Explorer] Error loading saved state:",n)}let n=t.querySelector(".explorer-ul");',
  ],
  [
    'N(o,n,D,E,r,"")',
    'N(o,n,D,E,r,z,"")',
  ],
  [
    'let i=!f.classList.contains("open"),B=c.dataset.folderpath,h=JSON.parse(localStorage.getItem("fileTree")||"[]"),g=h.findIndex(x=>x.path===B);g>=0?h[g].collapsed=i:h.push({path:B,collapsed:i}),localStorage.setItem("fileTree",JSON.stringify(h))',
    'let i=!f.classList.contains("open"),B=c.dataset.folderpath;if(q){let h=JSON.parse(localStorage.getItem("fileTree")||"[]"),g=h.findIndex(x=>x.path===B);g>=0?h[g].collapsed=i:h.push({path:B,collapsed:i}),localStorage.setItem("fileTree",JSON.stringify(h))}',
  ],
  [
    'let h=!i.classList.contains("open"),g=JSON.parse(localStorage.getItem("fileTree")||"[]"),x=g.findIndex(w=>w.path===B);x>=0?g[x].collapsed=h:g.push({path:B,collapsed:h}),localStorage.setItem("fileTree",JSON.stringify(g))',
    'let h=!i.classList.contains("open");if(q){let g=JSON.parse(localStorage.getItem("fileTree")||"[]"),x=g.findIndex(w=>w.path===B);x>=0?g[x].collapsed=h:g.push({path:B,collapsed:h}),localStorage.setItem("fileTree",JSON.stringify(g))}',
  ],
  [
    'l.addEventListener("click",o),s.push(()=>l.removeEventListener("click",o))',
    "l.onclick=o",
  ],
  [
    "overflow: hidden;\\n  margin-left: 6px;",
    "overflow: hidden;\\n  min-height: 0;\\n  margin-left: 6px;",
  ],
  [
    "overflow:hidden;margin-left:6px;",
    "overflow:hidden;min-height:0;margin-left:6px;",
  ],
  [
    '  if (node.isFolder) {\\n    const clone = folderTemplate.content.cloneNode(true);',
    '  if (node.isFolder && node.data && (!node.children || node.children.length === 0)) {\\n    const clone = fileTemplate.content.cloneNode(true);\\n    const link = clone.querySelector(\\"a\\");\\n    if (link) {\\n      link.href = resolveBasePath(simplifySlug(node.slug));\\n      link.textContent = node.displayName || node.slugSegment;\\n      if (simplifySlug(node.slug) === simplifiedCurrentSlug) {\\n        link.classList.add(\\"active\\", \\"is-active\\");\\n      }\\n    }\\n    container.appendChild(clone);\\n  } else if (node.isFolder) {\\n    const clone = folderTemplate.content.cloneNode(true);',
  ],
  [
    '  if (node.isFolder && node.data && (!node.children || node.children.length === 0)) {\\n    const clone = fileTemplate.content.cloneNode(true);\\n    const link = clone.querySelector("a");\\n    if (link) {\\n      link.href = resolveBasePath(simplifySlug(node.slug));\\n      link.textContent = node.displayName || node.slugSegment;\\n      if (simplifySlug(node.slug) === simplifiedCurrentSlug) {\\n        link.classList.add("active", "is-active");\\n      }\\n    }\\n    container.appendChild(clone);\\n  } else if (node.isFolder) {\\n    const clone = folderTemplate.content.cloneNode(true);',
    '  if (node.isFolder && node.data && (!node.children || node.children.length === 0)) {\\n    const clone = fileTemplate.content.cloneNode(true);\\n    const link = clone.querySelector(\\"a\\");\\n    if (link) {\\n      link.href = resolveBasePath(simplifySlug(node.slug));\\n      link.textContent = node.displayName || node.slugSegment;\\n      if (simplifySlug(node.slug) === simplifiedCurrentSlug) {\\n        link.classList.add(\\"active\\", \\"is-active\\");\\n      }\\n    }\\n    container.appendChild(clone);\\n  } else if (node.isFolder) {\\n    const clone = folderTemplate.content.cloneNode(true);',
  ],
  [
    'if(u.isFolder){let s=n.content.cloneNode(!0),a=s.querySelector(".folder-container")',
    'if(u.isFolder&&u.data&&(!u.children||u.children.length===0)){let s=d.content.cloneNode(!0),a=s.querySelector("a");a&&(a.href=v(S(u.slug)),a.textContent=u.displayName||u.slugSegment,S(u.slug)===A&&a.classList.add("active","is-active")),e.appendChild(s)}else if(u.isFolder){let s=n.content.cloneNode(!0),a=s.querySelector(".folder-container")',
  ],
  [
    ".explorer-content .folder-outer.open {\\n  visibility: visible;\\n  grid-template-rows: 1fr;\\n}\\n\\n.explorer-content .folder-outer > ul {",
    ".explorer-content .folder-outer.open {\\n  visibility: visible;\\n  grid-template-rows: 1fr;\\n}\\n\\n.explorer-content .folder-outer:not(.open) {\\n  height: 0;\\n  overflow: hidden;\\n}\\n\\n.explorer-content .folder-outer:not(.open) > ul {\\n  display: none;\\n}\\n\\n.explorer-content .folder-outer > ul {",
  ],
  [
    ".explorer-content .folder-outer.open{visibility:visible;grid-template-rows:1fr}.explorer-content .folder-outer>ul{",
    ".explorer-content .folder-outer.open{visibility:visible;grid-template-rows:1fr}.explorer-content .folder-outer:not(.open){height:0;overflow:hidden}.explorer-content .folder-outer:not(.open)>ul{display:none}.explorer-content .folder-outer>ul{",
  ],
]

for (const file of files) {
  if (!fs.existsSync(file)) {
    throw new Error(`Explorer plugin file is missing: ${file}`)
  }

  let text = fs.readFileSync(file, "utf8")
  const alreadyPatched =
    (text.includes('dataset.collapsed!=="open"') &&
      text.includes("l.onclick=o") &&
      text.includes("min-height") &&
      text.includes("folder-outer:not")) ||
    (text.includes('explorer.dataset.collapsed !== "open"') &&
      text.includes("button.onclick = clickHandler") &&
      text.includes("min-height") &&
      text.includes("folder-outer:not")) ||
    (text.includes('explorer.dataset.collapsed !== \\"open\\"') &&
      text.includes("button.onclick = clickHandler") &&
      text.includes("min-height") &&
      text.includes("folder-outer:not"))

  for (const [from, to] of replacements) {
    const count = text.split(from).length - 1
    if (count === 0 && !text.includes(to)) {
      const optionalCssAlternative =
        from.includes("overflow: hidden") ||
        from.includes("overflow:hidden") ||
        from.includes("folder-outer.open")
      const optionalLeafAlternative =
        from.includes("node.isFolder") || from.includes("u.isFolder")
      if (alreadyPatched) {
        continue
      }
      if (optionalCssAlternative || optionalLeafAlternative) {
        continue
      }
      throw new Error(`Explorer patch target not found in ${file}: ${from.slice(0, 80)}`)
    }
    text = text.split(from).join(to)
  }

  if (text.includes("folder-outer") && !text.includes("folder-outer:not(.open)")) {
    throw new Error(`Explorer CSS collapse patch was not applied in ${file}`)
  }

  if (
    text.includes("template-folder") &&
    !text.includes("node.isFolder && node.data && (!node.children || node.children.length === 0)") &&
    !text.includes("u.isFolder&&u.data&&(!u.children||u.children.length===0)")
  ) {
    throw new Error(`Explorer leaf-folder link patch was not applied in ${file}`)
  }

  fs.writeFileSync(file, text)
}

console.log("Explorer sidebar patch applied")
