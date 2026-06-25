import { loadQuartzConfig, loadQuartzLayout } from "./quartz/plugins/loader/config-loader"
import * as ExternalPlugin from "./.quartz/plugins"

const explorerRoots = ["ai", "bio-ai", "math", "infra", "research", "papers", "agents", "projects", "posts"]
const agentGroups = ["core", "tools", "workflows", "verification"]

ExternalPlugin.Explorer({
  filterFn: (node) => {
    if (node.slugSegment === "tags") return false
    const firstSegment = node.slugSegments?.[0]
    return firstSegment !== undefined && explorerRoots.includes(firstSegment)
  },
  sortFn: (a, b) => {
    const aIndex = explorerRoots.indexOf(a.slugSegments?.[0] ?? "")
    const bIndex = explorerRoots.indexOf(b.slugSegments?.[0] ?? "")

    if (a.slugSegments?.length === 1 && b.slugSegments?.length === 1 && aIndex !== bIndex) {
      return aIndex - bIndex
    }

    if (a.slugSegments?.[0] === "agents" && b.slugSegments?.[0] === "agents") {
      const aGroupIndex = agentGroups.indexOf(a.slugSegments?.[1] ?? "")
      const bGroupIndex = agentGroups.indexOf(b.slugSegments?.[1] ?? "")
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
