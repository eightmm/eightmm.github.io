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
]

for (const file of files) {
  if (!fs.existsSync(file)) {
    throw new Error(`Explorer plugin file is missing: ${file}`)
  }

  let text = fs.readFileSync(file, "utf8")
  const alreadyPatched =
    (text.includes('dataset.collapsed!=="open"') && text.includes("l.onclick=o")) ||
    (text.includes('explorer.dataset.collapsed !== "open"') &&
      text.includes("button.onclick = clickHandler")) ||
    (text.includes('explorer.dataset.collapsed !== \\"open\\"') &&
      text.includes("button.onclick = clickHandler"))

  for (const [from, to] of replacements) {
    const count = text.split(from).length - 1
    if (count === 0 && !text.includes(to)) {
      if (alreadyPatched) {
        continue
      }
      throw new Error(`Explorer patch target not found in ${file}: ${from.slice(0, 80)}`)
    }
    text = text.split(from).join(to)
  }

  fs.writeFileSync(file, text)
}

console.log("Explorer sidebar patch applied")
