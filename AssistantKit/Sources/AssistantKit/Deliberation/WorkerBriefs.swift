import Foundation

// The texts deep mode sends to Claude. They ask for conclusions and considerations, never for a
// write-up of the reasoning, because requests to reproduce reasoning get `reasoning_extraction`
// refusals.

extension WorkerBrief {
    /// Finds the current facts the answer depends on, with web search and the read-only tools.
    /// Effort `low`; its search activity is shown to the user.
    public static let researcher = WorkerBrief(
        id: "researcher",
        instruction: "You are the research step of a deliberation about the user's last message. Use web search to find current facts the answer depends on: numbers, dates, recent events, availability. Write compact notes, each fact with its date and source name. No final answer. At most 150 words.",
        effort: "low",
        usesTools: true,
        forwardsActivity: true
    )

    /// Works out the best answer from what the model knows. Effort `high`; no tools.
    public static let reasoner = WorkerBrief(
        id: "reasoner",
        instruction: "You are the reasoning step of a deliberation. Don't search the web. Work out the best answer from what you know. Write the conclusion and the 3–5 considerations it rests on, as compact notes. At most 150 words.",
        effort: "high",
        usesTools: false,
        forwardsActivity: false
    )

    /// Lists what a quick answer would get wrong. Effort `medium`; no tools.
    public static let critic = WorkerBrief(
        id: "critic",
        instruction: "You are the critic step of a deliberation. Don't search the web. List what a quick answer to the user's last message would most likely get wrong: hidden assumptions, missing context about the user, edge cases, risks, and the one question you'd ask them. At most 120 words.",
        effort: "medium",
        usesTools: false,
        forwardsActivity: false
    )
}

extension DeliberationConfiguration {
    /// The merger's brief. The workers' notes reach it as an `<analyst_notes>` block at the end of
    /// the user's last message.
    public static let defaultMergerInstruction = "Notes from three analysts about the user's last message are attached to it. Use them to give the best answer: reconcile disagreements, prefer the researched facts for anything current, and mention a real uncertainty briefly. Never mention analysts, notes or a deliberation. Follow the spoken-reply rules when the input was spoken; you may use up to six sentences."
}

/// The block that carries the workers' notes to the merger:
/// `<analyst_notes><research>…</research><reasoning>…</reasoning><critique>…</critique></analyst_notes>`,
/// one element per worker that produced text, in the workers' order, each on its own lines.
enum AnalystNotes {
    /// - Parameter notes: each worker's brief id and its (non-empty) notes, in the workers' order.
    static func block(_ notes: [(id: String, text: String)]) -> String {
        var lines = ["<analyst_notes>"]
        for note in notes {
            let tag = tag(for: note.id)
            lines.append("<\(tag)>")
            lines.append(note.text)
            lines.append("</\(tag)>")
        }
        lines.append("</analyst_notes>")
        return lines.joined(separator: "\n")
    }

    /// The element name for a worker's notes: `research`, `reasoning` and `critique` for the three
    /// standard briefs; for any other brief, its id reduced to lowercase letters, digits and `_`.
    static func tag(for id: String) -> String {
        switch id {
        case WorkerBrief.researcher.id: return "research"
        case WorkerBrief.reasoner.id: return "reasoning"
        case WorkerBrief.critic.id: return "critique"
        default:
            var tag = ""
            for scalar in id.lowercased().unicodeScalars {
                switch scalar.value {
                case 0x30...0x39, 0x61...0x7A, 0x5F: // 0-9, a-z, "_"
                    tag.unicodeScalars.append(scalar)
                default:
                    tag += "_"
                }
            }
            guard let first = tag.unicodeScalars.first else { return "notes" }
            // XML names can't start with a digit.
            return (0x30...0x39).contains(first.value) ? "notes_" + tag : tag
        }
    }
}
