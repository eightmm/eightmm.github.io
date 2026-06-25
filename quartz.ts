import { loadQuartzConfig, loadQuartzLayout } from "./quartz/plugins/loader/config-loader"
import * as ExternalPlugin from "./.quartz/plugins"

ExternalPlugin.Explorer({
  filterFn: (node) => {
    const explorerRoots = ["ai", "bio-ai", "infra", "research", "agents", "posts"]
    if (node.slugSegment === "tags") return false
    const firstSegment = node.slugSegments?.[0]
    return firstSegment !== undefined && explorerRoots.includes(firstSegment)
  },
  sortFn: (a, b) => {
    const explorerRoots = ["ai", "bio-ai", "infra", "research", "agents", "posts"]
    const aIndex = explorerRoots.indexOf(a.slugSegments?.[0] ?? "")
    const bIndex = explorerRoots.indexOf(b.slugSegments?.[0] ?? "")

    if (a.slugSegments?.length === 1 && b.slugSegments?.length === 1 && aIndex !== bIndex) {
      return aIndex - bIndex
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
