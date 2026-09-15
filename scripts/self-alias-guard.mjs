#!/usr/bin/env node

import fs from "node:fs"
import path from "node:path"
import { execFileSync } from "node:child_process"

const args = process.argv.slice(2)
const writeMode = args.includes("--write")
const changedIndex = args.indexOf("--check-changed")

function normalizeSlug(value) {
  let slug = value.trim().replace(/^['"]|['"]$/g, "")
  slug = slug.replace(/\\/g, "/").replace(/^\/+/, "").replace(/\/+$/, "")
  slug = slug.replace(/\.md$/i, "")
  if (slug.endsWith("/index")) slug = slug.slice(0, -6)
  return slug
}

function canonicalSlug(file) {
  let slug = file.replace(/\\/g, "/").replace(/^content\//, "").replace(/\.md$/i, "")
  if (slug.endsWith("/index")) slug = slug.slice(0, -6)
  return normalizeSlug(slug)
}

function walk(dir) {
  const out = []
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name)
    if (entry.isDirectory()) out.push(...walk(full))
    else if (entry.isFile() && entry.name.endsWith(".md")) out.push(full)
  }
  return out
}

function changedPaperFiles(baseRef) {
  const output = execFileSync(
    "git",
    ["diff", "--name-only", "--diff-filter=ACMRT", `${baseRef}...HEAD`, "--", "content/papers"],
    { encoding: "utf8" },
  )
  return output
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => line.endsWith(".md") && fs.existsSync(line))
}

function inspectAndMaybeFix(file) {
  const original = fs.readFileSync(file, "utf8")
  const lines = original.split(/\r?\n/)
  if (lines[0] !== "---") return { changed: false, collisions: [] }

  const end = lines.indexOf("---", 1)
  if (end < 0) return { changed: false, collisions: [] }

  const canonical = canonicalSlug(file)
  const remove = new Set()
  const collisions = []
  let inAliases = false
  let aliasesKey = -1
  let aliasEntries = []

  for (let i = 1; i < end; i += 1) {
    const line = lines[i]

    if (/^aliases:\s*$/.test(line)) {
      inAliases = true
      aliasesKey = i
      aliasEntries = []
      continue
    }

    const scalarAlias = line.match(/^alias:\s*(.+?)\s*$/)
    if (scalarAlias) {
      const value = normalizeSlug(scalarAlias[1])
      if (value === canonical) {
        collisions.push({ file, canonical, alias: scalarAlias[1].trim() })
        if (writeMode) remove.add(i)
      }
      inAliases = false
      continue
    }

    if (inAliases) {
      const item = line.match(/^\s*-\s*(.+?)\s*$/)
      if (item) {
        aliasEntries.push(i)
        const value = normalizeSlug(item[1])
        if (value === canonical) {
          collisions.push({ file, canonical, alias: item[1].trim() })
          if (writeMode) remove.add(i)
        }
        continue
      }

      if (/^\S/.test(line)) {
        if (writeMode && aliasesKey >= 0 && aliasEntries.length > 0 && aliasEntries.every((idx) => remove.has(idx))) {
          remove.add(aliasesKey)
        }
        inAliases = false
        aliasesKey = -1
        aliasEntries = []
      }
    }
  }

  if (writeMode && inAliases && aliasesKey >= 0 && aliasEntries.length > 0 && aliasEntries.every((idx) => remove.has(idx))) {
    remove.add(aliasesKey)
  }

  if (writeMode && remove.size > 0) {
    const updated = lines.filter((_, index) => !remove.has(index)).join("\n")
    fs.writeFileSync(file, updated)
    return { changed: true, collisions }
  }

  return { changed: false, collisions }
}

let files
if (changedIndex >= 0) {
  const baseRef = args[changedIndex + 1]
  if (!baseRef) {
    console.error("--check-changed requires a base ref, e.g. origin/v5")
    process.exit(2)
  }
  files = changedPaperFiles(baseRef)
} else {
  files = walk("content/papers")
}

let collisionCount = 0
let changedCount = 0
for (const file of files) {
  const result = inspectAndMaybeFix(file)
  collisionCount += result.collisions.length
  if (result.changed) changedCount += 1

  if (!writeMode) {
    for (const collision of result.collisions) {
      console.error(
        `Self-alias collision: ${collision.file}: alias '${collision.alias}' equals canonical slug '${collision.canonical}'`,
      )
    }
  }
}

if (writeMode) {
  console.log(`Self-alias sanitizer: removed ${collisionCount} collision(s) from ${changedCount} file(s) for this build.`)
  process.exit(0)
}

if (collisionCount > 0) {
  console.error(`Found ${collisionCount} self-alias collision(s). Remove canonical slugs from aliases.`)
  process.exit(1)
}

console.log(`Self-alias guard: ${files.length} paper file(s) checked; no collisions found.`)
