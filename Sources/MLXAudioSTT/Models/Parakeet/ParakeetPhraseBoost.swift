import Foundation
import MLX

/// Token-prefix tree for TDT greedy phrase boosting. A failed prefix falls back
/// to its longest matching suffix (including the start of another phrase).
struct ParakeetPhraseBoost {
    private struct Node {
        var children: [Int: Int] = [:]
        var fallback = 0
    }
    private var nodes = [Node()]
    let biases: [MLXArray]

    init?(phrases: [String], vocabulary: [String], strength: Float) {
        guard !phrases.isEmpty, strength.isFinite, strength > 0 else { return nil }
        var ids = [String: Int]()
        for (id, piece) in vocabulary.enumerated() where !ParakeetTokenizer.isSpecialToken(id, vocabulary: vocabulary) {
            ids[piece] = id
        }

        // Exported checkpoints contain the output pieces even when they omit
        // the original SentencePiece model. Find a shortest exact segmentation.
        for phrase in phrases.prefix(64) {
            let words = phrase.split(whereSeparator: { $0.isWhitespace })
            guard !words.isEmpty, phrase.count <= 80 else { continue }
            let spelling = "▁" + words.joined(separator: "▁")
            let chars = Array(spelling)
            var paths = [[Int]?](repeating: nil, count: chars.count + 1)
            paths[0] = []
            for i in chars.indices {
                guard let path = paths[i] else { continue }
                var piece = ""
                for j in i..<chars.count {
                    piece.append(chars[j])
                    if let id = ids[piece], path.count + 1 < (paths[j + 1]?.count ?? .max) {
                        paths[j + 1] = path + [id]
                    }
                }
            }
            guard let tokens = paths.last ?? nil, nodes.count + tokens.count <= 512 else { continue }
            var state = 0
            for token in tokens {
                if let child = nodes[state].children[token] {
                    state = child
                } else {
                    let child = nodes.count
                    nodes.append(Node())
                    nodes[state].children[token] = child
                    state = child
                }
            }
        }
        guard nodes.count > 1 else { return nil }
        var queue = Array(nodes[0].children.values)
        var index = 0
        while index < queue.count {
            let parent = queue[index]
            index += 1
            for (token, child) in nodes[parent].children {
                var suffix = nodes[parent].fallback
                while suffix != 0 && nodes[suffix].children[token] == nil {
                    suffix = nodes[suffix].fallback
                }
                nodes[child].fallback = nodes[suffix].children[token] ?? 0
                queue.append(child)
            }
        }

        // Precompute one device vector per branching state. Leaves reuse their
        // suffix's vector. The blank and all duration logits receive no bonus.
        var rows = [MLXArray?](repeating: nil, count: nodes.count)
        for state in [0] + queue {
            if state != 0 && nodes[state].children.isEmpty {
                rows[state] = rows[nodes[state].fallback]
                continue
            }
            var scores = [Float](repeating: 0, count: vocabulary.count + 1)
            var suffix = state
            while true {
                let bonus = suffix == 0 ? strength * 0.35 : strength
                for token in nodes[suffix].children.keys { scores[token] = max(scores[token], bonus) }
                if suffix == 0 { break }
                suffix = nodes[suffix].fallback
            }
            rows[state] = MLXArray(scores)
        }
        biases = rows.map { $0! }
    }

    func advance(_ state: Int, token: Int) -> Int {
        var suffix = state
        while suffix != 0 && nodes[suffix].children[token] == nil {
            suffix = nodes[suffix].fallback
        }
        return nodes[suffix].children[token] ?? 0
    }
}
