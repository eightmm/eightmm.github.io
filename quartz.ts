import { loadQuartzConfig, loadQuartzLayout } from "./quartz/plugins/loader/config-loader"
import * as ExternalPlugin from "./.quartz/plugins"

ExternalPlugin.Explorer({
  filterFn: (node) => {
    // Explorer serializes callbacks into HTML, so keep required values inside the callback.
    const explorerRoots = ["ai", "bio-ai", "math", "infra", "research", "papers", "agents", "projects", "posts"]
    const slug = node.slug ?? ""
    const slugSegment = node.slugSegment ?? ""
    if (slugSegment === "tags") return false
    const segments = slug.split("/")
    const firstSegment = segments[0]
    return firstSegment !== undefined && explorerRoots.includes(firstSegment)
  },
  sortFn: (a, b) => {
    // Explorer serializes callbacks into HTML, so keep required values inside the callback.
    const explorerRoots = ["ai", "bio-ai", "math", "infra", "research", "papers", "agents", "projects", "posts"]
    const agentGroups = ["core", "tools", "workflows", "verification"]
    const aSegments = (a.slug ?? "").split("/")
    const bSegments = (b.slug ?? "").split("/")
    if (aSegments.at(-1) === "index") aSegments.pop()
    if (bSegments.at(-1) === "index") bSegments.pop()
    const aIndex = explorerRoots.indexOf(aSegments[0] ?? "")
    const bIndex = explorerRoots.indexOf(bSegments[0] ?? "")

    if (aSegments.length === 1 && bSegments.length === 1 && aIndex !== bIndex) {
      return aIndex - bIndex
    }

    if (aSegments[0] === "agents" && bSegments[0] === "agents") {
      const aGroupIndex = agentGroups.indexOf(aSegments[1] ?? "")
      const bGroupIndex = agentGroups.indexOf(bSegments[1] ?? "")
      if (aGroupIndex !== -1 && bGroupIndex !== -1 && aGroupIndex !== bGroupIndex) {
        return aGroupIndex - bGroupIndex
      }
    }

    if ((!a.isFolder && !b.isFolder) || (a.isFolder && b.isFolder)) {
      return (a.displayName || "").localeCompare(b.displayName || "", undefined, {
        numeric: true,
        sensitivity: "base",
      })
    }

    return a.isFolder ? -1 : 1
  },
})

const config = await loadQuartzConfig()
export default config
export const layout = await loadQuartzLayout()
