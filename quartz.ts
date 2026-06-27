import { loadQuartzConfig, loadQuartzLayout } from "./quartz/plugins/loader/config-loader"
import * as ExternalPlugin from "./.quartz/plugins"

ExternalPlugin.Explorer({
  filterFn: (node) => {
    // Explorer serializes callbacks into HTML, so keep required values inside the callback.
    const explorerRoots = [
      "ai",
      "math",
      "molecular-modeling",
      "infra",
      "research",
      "papers",
      "agents",
      "projects",
      "posts",
    ]
    const hiddenExplorerSubtrees = [
      "ai/paper-intake",
      "ai/paper-claim-patterns",
      "molecular-modeling/computational-biology",
      "molecular-modeling/docking",
      "molecular-modeling/geometry",
      "molecular-modeling/genome",
      "molecular-modeling/modeling-scope",
      "molecular-modeling/molecules",
      "molecular-modeling/paper-intake",
      "molecular-modeling/paper-claim-patterns",
      "molecular-modeling/protein-modeling",
      "molecular-modeling/proteins",
      "molecular-modeling/structure-based/protein-ligand-docking",
      "math/formula-intake",
      "math/formula-patterns",
      "math/formula-explanation-ladder",
      "papers/analysis",
      "papers/protein-modeling",
      "papers/reproducibility",
      "papers/sbdd",
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
  mapFn: (node) => {
    if (node.slugSegment === "molecular-modeling") {
      node.displayName = "Computational Biology"
    }
    if (node.slugSegment === "entities") {
      const slugSegments = Array.isArray(node.slugSegments) ? node.slugSegments : []
      if (slugSegments.join("/") === "molecular-modeling/entities") {
        node.displayName = "Objects and Entities"
      }
    }
  },
  sortFn: (a, b) => {
    // Explorer serializes callbacks into HTML, so keep required values inside the callback.
    const explorerRoots = [
      "ai",
      "math",
      "molecular-modeling",
      "infra",
      "research",
      "papers",
      "agents",
      "projects",
      "posts",
    ]
    const aiGroups = [
      "machine-learning",
      "architectures",
      "learning-methods",
      "generative-models",
      "evaluation",
    ]
    const molecularGroups = [
      "entities",
      "sequence-based",
      "molecular-ligand",
      "interactions",
      "structure-based",
      "data-evaluation",
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
      "formula-patterns",
      "formula-intake",
      "formula-explanation-ladder",
    ]
    const agentGroups = ["core", "tools", "workflows", "verification"]
    const paperGroups = [
      "architectures",
      "llm",
      "generative-models",
      "computational-biology",
    ]
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

    if (aSegments[0] === "ai" && bSegments[0] === "ai") {
      const aGroupIndex = aiGroups.indexOf(aSegments[1] ?? "")
      const bGroupIndex = aiGroups.indexOf(bSegments[1] ?? "")
      if (aGroupIndex !== -1 && bGroupIndex !== -1 && aGroupIndex !== bGroupIndex) {
        return aGroupIndex - bGroupIndex
      }
    }

    if (aSegments[0] === "molecular-modeling" && bSegments[0] === "molecular-modeling") {
      const aGroupIndex = molecularGroups.indexOf(aSegments[1] ?? "")
      const bGroupIndex = molecularGroups.indexOf(bSegments[1] ?? "")
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

    if (aSegments[0] === "papers" && bSegments[0] === "papers") {
      const aGroupIndex = paperGroups.indexOf(aSegments[1] ?? "")
      const bGroupIndex = paperGroups.indexOf(bSegments[1] ?? "")
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
