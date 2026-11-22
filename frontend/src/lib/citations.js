// Same marker format the backend uses (app/services/rag.py): [1], [2, 3], [1; 4]
const CITATION = /\[(\d+(?:\s*[,;]\s*\d+)*)\]/g

/**
 * Split an answer into text parts and citation parts so citations can be rendered as buttons.
 * Returns e.g. [{type:'text', value:'Hold RESET '}, {type:'cite', numbers:[1]}, ...]
 */
export function splitCitations(answer) {
  const parts = []
  let last = 0
  for (const match of answer.matchAll(CITATION)) {
    if (match.index > last) parts.push({ type: 'text', value: answer.slice(last, match.index) })
    parts.push({ type: 'cite', numbers: match[1].split(/\s*[,;]\s*/).map(Number) })
    last = match.index + match[0].length
  }
  if (last < answer.length) parts.push({ type: 'text', value: answer.slice(last) })
  return parts
}
