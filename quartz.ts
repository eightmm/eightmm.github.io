import { loadQuartzConfig, loadQuartzLayout } from "./quartz/plugins/loader/config-loader"
import * as ExternalPlugin from "./.quartz/plugins"

ExternalPlugin.Explorer({
  filterFn: (node) => {
    // Explorer serializes callbacks into HTML, so keep required values inside the callback.
    const explorerRoots = ["ai", "bio-ai", "math", "infra", "research", "papers", "agents", "projects", "posts"]
    const hiddenExplorerSubtrees = [
      "papers/analysis",
      "papers/reproducibility",
      "papers/workflows",
      "papers/learning-methods",
      "papers/systems",
    ]
    const slugSegments = Array.isArray(node.slugSegments) ? node.slugSegments : []
    const dataSlug = typeof node.data?.slug === "string" ? node.data.slug : ""
    const slug = slugSegments.length > 0 ? slugSegments.join("/") : dataSlug || node.slug || ""
    const slugSegment = node.slugSegment ?? slugSegments.at(-1) ?? ""
    if (slugSegment === "tags") return false
    if (hiddenExplorerSubtrees.some((prefix) => slug === prefix || slug.startsWith(prefix + "/"))) {
      return false
    }
    const segments = slug.split("/")
    const firstSegment = segments[0]
    return firstSegment !== undefined && explorerRoots.includes(firstSegment)
  },
  sortFn: (a, b) => {
    // Explorer serializes callbacks into HTML, so keep required values inside the callback.
    const explorerRoots = ["ai", "bio-ai", "math", "infra", "research", "papers", "agents", "projects", "posts"]
    const bioAiGroups = [
      "entities",
      "molecules",
      "proteins",
      "structure-based-ai",
      "data-evaluation",
      "geometry",
      "genome",
    ]
    const mathGroups = [
      "linear-algebra",
      "calculus-gradients",
      "probability-statistics",
      "information-likelihood",
      "geometry-symmetry",
      "discrete-graphs",
      "dynamical-systems",
      "numerical-computing",
      "evaluation-math",
    ]
    const agentGroups = ["core", "tools", "workflows", "verification"]
    const aSlugSegments = Array.isArray(a.slugSegments) ? a.slugSegments : []
    const bSlugSegments = Array.isArray(b.slugSegments) ? b.slugSegments : []
    const aDataSlug = typeof a.data?.slug === "string" ? a.data.slug : ""
    const bDataSlug = typeof b.data?.slug === "string" ? b.data.slug : ""
    const aSegments = (aSlugSegments.length > 0 ? aSlugSegments.join("/") : aDataSlug || a.slug || "").split("/")
    const bSegments = (bSlugSegments.length > 0 ? bSlugSegments.join("/") : bDataSlug || b.slug || "").split("/")
    if (aSegments.at(-1) === "index") aSegments.pop()
    if (bSegments.at(-1) === "index") bSegments.pop()
    const aIndex = explorerRoots.indexOf(aSegments[0] ?? "")
    const bIndex = explorerRoots.indexOf(bSegments[0] ?? "")

    if (aSegments.length === 1 && bSegments.length === 1 && aIndex !== bIndex) {
      return aIndex - bIndex
    }

    if (aSegments[0] === "bio-ai" && bSegments[0] === "bio-ai") {
      const aGroupIndex = bioAiGroups.indexOf(aSegments[1] ?? "")
      const bGroupIndex = bioAiGroups.indexOf(bSegments[1] ?? "")
      if (aGroupIndex !== -1 && bGroupIndex !== -1 && aGroupIndex !== bGroupIndex) {
        return aGroupIndex - bGroupIndex
      }
    }

    if (aSegments[0] === "math" && bSegments[0] === "math") {
      const aGroupIndex = mathGroups.indexOf(aSegments[1] ?? "")
      const bGroupIndex = mathGroups.indexOf(bSegments[1] ?? "")
      if (aGroupIndex !== -1 && bGroupIndex !== -1 && aGroupIndex !== bGroupIndex) {
        return aGroupIndex - bGroupIndex
      }
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
