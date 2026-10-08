import Foundation

/// Measurements of one on-device reply.
public struct LocalGenerationStats: Equatable, Sendable {
    /// From the request to the first visible text, including any prefill.
    public var timeToFirstText: TimeInterval?
    public var promptTokens: Int
    public var promptTime: TimeInterval
    public var generatedTokens: Int
    public var generateTime: TimeInterval
    /// Whether the cached session was continued (only the new turn was prefilled).
    public var reusedSession: Bool
    public var draftTokens: Int?
    public var acceptedDraftTokens: Int?
    /// Peak MLX memory during the reply, in bytes.
    public var peakMemoryBytes: Int?
    /// Which generator answered: e.g. "alvin" (the engine) or "stock" (`ChatSession`).
    public var engine: String?
    /// The engine's time to first token, by phase.
    public var phases: EnginePhaseTimes?
    /// Tokens the engine prefilled for this reply.
    public var prefilledTokens: Int?
    /// Tokens of the live cache the engine kept instead of prefilling.
    public var reusedTokens: Int?
    /// The session planner's reason (`SessionPlan.Reason` raw value).
    public var planReason: String?
    /// The engine's speculative decoding, when it ran.
    public var speculation: SpeculationStats?
    /// The sampled tokens' top-1 probabilities.
    public var confidence: TokenConfidence?

    public init(
        timeToFirstText: TimeInterval? = nil,
        promptTokens: Int = 0,
        promptTime: TimeInterval = 0,
        generatedTokens: Int = 0,
        generateTime: TimeInterval = 0,
        reusedSession: Bool = false,
        draftTokens: Int? = nil,
        acceptedDraftTokens: Int? = nil,
        peakMemoryBytes: Int? = nil,
        engine: String? = nil,
        phases: EnginePhaseTimes? = nil,
        prefilledTokens: Int? = nil,
        reusedTokens: Int? = nil,
        planReason: String? = nil,
        speculation: SpeculationStats? = nil,
        confidence: TokenConfidence? = nil
    ) {
        self.timeToFirstText = timeToFirstText
        self.promptTokens = promptTokens
        self.promptTime = promptTime
        self.generatedTokens = generatedTokens
        self.generateTime = generateTime
        self.reusedSession = reusedSession
        self.draftTokens = draftTokens
        self.acceptedDraftTokens = acceptedDraftTokens
        self.peakMemoryBytes = peakMemoryBytes
        self.engine = engine
        self.phases = phases
        self.prefilledTokens = prefilledTokens
        self.reusedTokens = reusedTokens
        self.planReason = planReason
        self.speculation = speculation
        self.confidence = confidence
    }

    public var tokensPerSecond: Double? {
        generateTime > 0 ? Double(generatedTokens) / generateTime : nil
    }

    public var prefillTokensPerSecond: Double? {
        promptTime > 0 ? Double(promptTokens) / promptTime : nil
    }

    /// Share of drafted tokens the main model accepted, when speculative decoding ran.
    public var acceptanceRate: Double? {
        guard let draftTokens, let acceptedDraftTokens, draftTokens > 0 else { return nil }
        return Double(acceptedDraftTokens) / Double(draftTokens)
    }
}

/// The scripted conversations the on-device benchmark plays, and its report.
public enum LocalBenchmark {
    /// Voice-style turns in English and Chinese, played as one conversation so that later turns
    /// show the effect of reusing the cached session.
    public static let prompts = [
        "Hi! What can you help me with?",
        "Give me one tip for sleeping better.",
        "Why is the sky blue? Keep it short.",
        "And why are sunsets red then?",
        "用一句话介绍一下新加坡。",
        "我明天早上要早起，有什么建议吗？",
        "Write a two-line poem about coffee.",
        "Thanks, that's all for now.",
    ]

    public struct Row: Equatable, Sendable {
        public var prompt: String
        public var reply: String
        public var stats: LocalGenerationStats
        /// For a turn with a `ToolExpectation`: how its first tool call scored.
        public var toolScore: ToolScore?

        public init(prompt: String, reply: String, stats: LocalGenerationStats, toolScore: ToolScore? = nil) {
            self.prompt = prompt
            self.reply = reply
            self.stats = stats
            self.toolScore = toolScore
        }
    }

    /// A plain-text report to copy and share.
    public static func report(model: String, device: String, speculative: String, loadTime: TimeInterval?, rows: [Row], scenario: String? = nil) -> String {
        var lines = [
            "On-device benchmark",
            "Model: \(model)",
            "Device: \(device)",
            "Speculative decoding: \(speculative)",
        ]
        if let scenario { lines.append("Scenario: \(scenario)") }
        if let loadTime { lines.append("Load time: \(format(loadTime, digits: 1)) s") }
        lines.append("")
        lines.append("turn | first text s | prefill tok (tok/s) | gen tok/s | draft accept | cache | peak GB | engine | plan | prefilled/reused | tok/round | acc%")
        for (index, row) in rows.enumerated() {
            let stats = row.stats
            var columns: [String] = [
                "\(index + 1)",
                stats.timeToFirstText.map { format($0, digits: 2) } ?? "–",
                "\(stats.promptTokens) (\(stats.prefillTokensPerSecond.map { format($0, digits: 0) } ?? "–"))",
                stats.tokensPerSecond.map { format($0, digits: 1) } ?? "–",
                stats.acceptanceRate.map(percent) ?? "–",
                stats.reusedSession ? "reused" : "rebuilt",
                stats.peakMemoryBytes.map { format(Double($0) / 1e9, digits: 2) } ?? "–",
            ]
            columns += engineColumns(stats)
            lines.append(columns.joined(separator: " | "))
        }
        let summaries = [summary(rows), toolSummary(rows)].compactMap { $0 }
        if !summaries.isEmpty {
            lines.append("")
            lines += summaries
        }
        lines.append("")
        for (index, row) in rows.enumerated() {
            lines.append("\(index + 1). \(row.prompt)")
            lines.append("   → \(row.reply.trimmed.replacingOccurrences(of: "\n", with: " "))")
        }
        return lines.joined(separator: "\n")
    }

    /// Medians across turns, which matter more than any single turn.
    static func summary(_ rows: [Row]) -> String? {
        let firstText = rows.compactMap(\.stats.timeToFirstText)
        let speed = rows.compactMap(\.stats.tokensPerSecond)
        guard !firstText.isEmpty || !speed.isEmpty else { return nil }
        var parts: [String] = []
        if let value = median(firstText) { parts.append("median first text \(format(value, digits: 2)) s") }
        if let value = median(speed) { parts.append("median \(format(value, digits: 1)) tok/s") }
        let reused = rows.filter(\.stats.reusedSession).compactMap(\.stats.timeToFirstText)
        let rebuilt = rows.filter { !$0.stats.reusedSession }.compactMap(\.stats.timeToFirstText)
        if let reused = median(reused), let rebuilt = median(rebuilt) {
            parts.append("first text \(format(reused, digits: 2)) s with the cache vs \(format(rebuilt, digits: 2)) s without")
        }
        return "Summary: " + parts.joined(separator: "; ")
    }

    /// "Tool calls: name 10/12, arguments 8/12", when any row was scored.
    static func toolSummary(_ rows: [Row]) -> String? {
        let scores = rows.compactMap(\.toolScore)
        guard !scores.isEmpty else { return nil }
        let names = scores.filter(\.nameMatches).count
        let arguments = scores.filter(\.argumentsMatch).count
        return "Tool calls: name \(names)/\(scores.count), arguments \(arguments)/\(scores.count)"
    }

    /// The `engine | plan | prefilled/reused | tok/round | acc%` columns.
    static func engineColumns(_ stats: LocalGenerationStats) -> [String] {
        let tokensPerRound: String = stats.speculation?.meanTokensPerRound.map { format($0, digits: 2) } ?? "–"
        let acceptance: String = stats.speculation?.acceptanceRate.map(percent) ?? "–"
        return [stats.engine ?? "–", stats.planReason ?? "–", prefilledReused(stats), tokensPerRound, acceptance]
    }

    /// "48/812": tokens prefilled / tokens of the cache reused.
    static func prefilledReused(_ stats: LocalGenerationStats) -> String {
        guard stats.prefilledTokens != nil || stats.reusedTokens != nil else { return "–" }
        return "\(stats.prefilledTokens.map(String.init) ?? "–")/\(stats.reusedTokens.map(String.init) ?? "–")"
    }

    static func percent(_ value: Double) -> String {
        "\(Int((value * 100).rounded()))%"
    }

    static func median(_ values: [Double]) -> Double? {
        guard !values.isEmpty else { return nil }
        let sorted = values.sorted()
        let middle = sorted.count / 2
        return sorted.count.isMultiple(of: 2) ? (sorted[middle - 1] + sorted[middle]) / 2 : sorted[middle]
    }

    static func format(_ value: Double, digits: Int) -> String {
        String(format: "%.\(digits)f", value)
    }
}

// MARK: - Scenarios

extension LocalBenchmark {
    /// A scripted run of the benchmark.
    ///
    /// Steps run in order. `.cancelAfter` and `.storeSpokenPrefix` modify the reply to the `.user`
    /// step before them; `actions` folds them into it.
    public struct Scenario: Identifiable, Equatable, Sendable {
        public enum Step: Equatable, Sendable {
            /// The user says this; the model replies.
            case user(String)
            /// Cancel the reply after this many generated tokens (a barge-in).
            case cancelAfter(tokens: Int)
            /// Store only the reply's first words, as the app keeps what was spoken before a barge-in.
            case storeSpokenPrefix(words: Int)
            /// Unload and load the model again, with or without the system prefix saved on disk.
            case reloadModel(usePrefixCache: Bool)
            /// Start an empty conversation; the loaded model and its cache stay.
            case newConversation
        }

        /// A `.user` step with its modifiers and expectation.
        public struct Turn: Equatable, Sendable {
            public var prompt: String
            public var cancelAfterTokens: Int?
            public var storedWords: Int?
            public var expectation: ToolExpectation?

            public init(prompt: String, cancelAfterTokens: Int? = nil, storedWords: Int? = nil, expectation: ToolExpectation? = nil) {
                self.prompt = prompt
                self.cancelAfterTokens = cancelAfterTokens
                self.storedWords = storedWords
                self.expectation = expectation
            }
        }

        public enum Action: Equatable, Sendable {
            case turn(Turn)
            case reloadModel(usePrefixCache: Bool)
            case newConversation
        }

        public var id: String
        public var title: String
        public var steps: [Step]
        /// The context tag sent before every user turn (`ChatTurn.context`).
        public var context: String
        /// Expected tool calls, keyed by the zero-based number of the `.user` step they answer.
        public var expectations: [Int: ToolExpectation]

        public init(id: String, title: String, steps: [Step], context: String = LocalBenchmark.spokenContext, expectations: [Int: ToolExpectation] = [:]) {
            self.id = id
            self.title = title
            self.steps = steps
            self.context = context
            self.expectations = expectations
        }

        /// The steps, with each `.user` step's modifiers and expectation folded into its turn.
        /// A modifier that doesn't follow a `.user` step is ignored.
        public var actions: [Action] {
            var actions: [Action] = []
            var userSteps = 0
            for step in steps {
                switch step {
                case .user(let prompt):
                    actions.append(.turn(Turn(prompt: prompt, expectation: expectations[userSteps])))
                    userSteps += 1
                case .cancelAfter(let tokens):
                    if case .turn(var turn) = actions.last {
                        turn.cancelAfterTokens = tokens
                        actions[actions.count - 1] = .turn(turn)
                    }
                case .storeSpokenPrefix(let words):
                    if case .turn(var turn) = actions.last {
                        turn.storedWords = words
                        actions[actions.count - 1] = .turn(turn)
                    }
                case .reloadModel(let usePrefixCache):
                    actions.append(.reloadModel(usePrefixCache: usePrefixCache))
                case .newConversation:
                    actions.append(.newConversation)
                }
            }
            return actions
        }

        /// The number of `.user` steps.
        public var turnCount: Int {
            steps.filter { step in
                if case .user = step { return true }
                return false
            }.count
        }
    }

    /// The context tag of today's benchmark turns.
    public static let spokenContext = "<context>input: spoken</context>"
    /// The drafter lab's context tag (`scripts/lab_prompts.json`) at its fixed time, which the
    /// expected tool arguments assume: Wednesday 7 October 2026, so "tomorrow" is 2026-10-08.
    public static let labSpokenContext = "<context>time: Wednesday 7 October 2026, 16:05 Asia/Singapore; input: spoken</context>"
    public static let labTypedContext = "<context>time: Wednesday 7 October 2026, 16:05 Asia/Singapore; input: typed</context>"

    /// The first `words` whitespace-separated words of `text`: what the app keeps of a reply that
    /// was interrupted after that much was spoken.
    public static func spokenPrefix(of text: String, words: Int) -> String {
        text.split(whereSeparator: \.isWhitespace).prefix(max(0, words)).joined(separator: " ")
    }

    public static let scenarios: [Scenario] = [continued, bargeIn, coldPrefix, longChat, copyHeavy, tools]

    public static func scenario(id: String) -> Scenario? {
        scenarios.first { $0.id == id }
    }

    /// Today's eight prompts as one conversation: every turn after the first continues the cache.
    public static let continued = Scenario(id: "continued", title: "Continued chat", steps: prompts.map(Scenario.Step.user))

    /// The user interrupts turn 2 after 10 tokens and the app keeps its first 6 words, so turn 3
    /// replaces the cached reply.
    public static let bargeIn = Scenario(
        id: "bargeIn",
        title: "Barge-in",
        steps: [
            .user("Hi! What can you help me with?"),
            .user("Tell me the story of how coffee spread around the world."),
            .cancelAfter(tokens: 10),
            .storeSpokenPrefix(words: 6),
            .user("Sorry, shorter please. Just one sentence."),
            .user("And where is most coffee grown today?"),
            .user("谢谢！那茶最早是从哪里来的？"),
        ]
    )

    /// Turn 1 after a reload: first to make sure the system prefix is saved, then without and
    /// with the saved prefix.
    public static let coldPrefix = Scenario(
        id: "coldPrefix",
        title: "Cold start with and without the saved prefix",
        steps: [
            .reloadModel(usePrefixCache: true),
            .user(prompts[0]),
            .newConversation,
            .reloadModel(usePrefixCache: false),
            .user(prompts[0]),
            .newConversation,
            .reloadModel(usePrefixCache: true),
            .user(prompts[0]),
        ]
    )

    /// Thirty short turns as one conversation, to show the cost of a growing cache.
    public static let longChat = Scenario(
        id: "longChat",
        title: "Long chat",
        steps: [
            "Hi there!",
            "What's the capital of Japan?",
            "And its population, roughly?",
            "Suggest a name for a goldfish.",
            "Give me another one.",
            "How do you say thank you in French?",
            "And in German?",
            "用一句话说说为什么要多喝水。",
            "What's seven times eight?",
            "Divide that by four.",
            "Name a fruit that starts with K.",
            "Is it healthy?",
            "用一句话解释什么是光合作用。",
            "Shorter, please.",
            "What rhymes with orange?",
            "Tell me a fun fact about octopuses.",
            "One more.",
            "推荐一本好书。",
            "Why that one?",
            "How many minutes are in a day?",
            "And in a week?",
            "What's the opposite of ancient?",
            "Spell necessary.",
            "给我一个早餐的建议。",
            "Something without eggs?",
            "What colour do blue and yellow make?",
            "And red and white?",
            "How long should I boil an egg?",
            "用中文说晚安。",
            "Thanks, bye!",
        ].map(Scenario.Step.user)
    )

    /// Prompts whose answers repeat much of the prompt, where prompt-lookup drafting helps most,
    /// each in a new conversation. The drafter lab's `copy-01` to `copy-06`.
    public static let copyHeavy = Scenario(
        id: "copyHeavy",
        title: "Copy-heavy",
        steps: separateConversations([
            "Read my shopping list back to me: eggs, two cartons of oat milk, sourdough bread, three avocados, cherry tomatoes, Greek yoghurt, and a five kilo bag of jasmine rice.",
            "Fix the typos in this and give me only the corrected text: \"Hi Sarah, thanks for you're email. I'll definately send the quarterly report by tomorow morning, and we can dicuss the budget at the meeting on thursday.\"",
            "Turn this into a JSON array of objects with name, phone and city: Alice Tan 9123 4567 Singapore, Ben Lim 8234 5678 Kuala Lumpur, Chloe Ng 9345 6789 Singapore.",
            "Rewrite this message to sound more polite, keeping every detail: \"Send me the invoice for order 4471 by Friday. The amount was 1,250 dollars and it has to go to accounts@example.com, not my personal email.\"",
            "把我的购物清单再读一遍：鸡蛋、两盒燕麦奶、全麦面包、三个牛油果、小番茄、希腊酸奶，还有一袋五公斤的茉莉香米。",
            "帮我改正这段话里的错别字，只输出改好的内容：“我明天下午三点在公司门口等你，记得带上合同和身分证，不要迟倒，我们要在四点前把资料交给王经里。”",
        ]),
        context: labTypedContext
    )

    /// Requests for device tools, each in a new conversation, with the expected tool and key
    /// arguments: the drafter lab's `tool-01` to `tool-12`, at the lab's fixed time. Run it with a
    /// tool executor that records the calls instead of acting on them.
    public static let tools = Scenario(
        id: "tools",
        title: "Tool calls",
        steps: separateConversations(toolPrompts.map(\.prompt)),
        context: labSpokenContext,
        expectations: Dictionary(uniqueKeysWithValues: toolPrompts.enumerated().map { ($0.offset, $0.element.expectation) })
    )

    static let toolPrompts: [(prompt: String, expectation: ToolExpectation)] = [
        ("Remind me at 5 to call mum.", ToolExpectation(name: "create_reminder", arguments: ["title": "call mum", "due": "2026-10-07T17:00"])),
        ("Set a timer for ten minutes for the pasta.", ToolExpectation(name: "set_timer", arguments: ["seconds": 600])),
        ("What's on my calendar tomorrow?", ToolExpectation(name: "list_events", arguments: ["start": "2026-10-08"])),
        ("Put a dentist appointment in my calendar for Friday at 3pm.", ToolExpectation(name: "create_event", arguments: ["title": "dentist", "start": "2026-10-09T15:00"])),
        ("Which of my reminders are due today?", ToolExpectation(name: "list_reminders", arguments: ["scope": "today"])),
        ("Add buy milk to my to-do list.", ToolExpectation(name: "create_reminder", arguments: ["title": "buy milk"])),
        ("提醒我明天早上八点去取快递。", ToolExpectation(name: "create_reminder", arguments: ["title": "快递", "due": "2026-10-08T08:00"])),
        ("帮我设一个二十分钟的计时器。", ToolExpectation(name: "set_timer", arguments: ["seconds": 1200])),
        ("我今天下午有什么安排？", ToolExpectation(name: "list_events", arguments: ["start": "2026-10-07"])),
        ("下周一上午十点和王经理开会，帮我加到日历里。", ToolExpectation(name: "create_event", arguments: ["title": "王经理", "start": "2026-10-12T10:00"])),
        ("有哪些提醒已经过期了？", ToolExpectation(name: "list_reminders", arguments: ["scope": "overdue"])),
        ("我的计时器还剩多少时间？", ToolExpectation(name: "list_timers")),
    ]

    /// `.user` steps separated by `.newConversation`.
    static func separateConversations(_ prompts: [String]) -> [Scenario.Step] {
        var steps: [Scenario.Step] = []
        for (index, prompt) in prompts.enumerated() {
            if index > 0 { steps.append(.newConversation) }
            steps.append(.user(prompt))
        }
        return steps
    }
}

// MARK: - Tool expectations

extension LocalBenchmark {
    /// The tool call a benchmark turn should make, scored like the drafter lab scores it
    /// (`expect_matching` in `scripts/lab_prompts.json`).
    public struct ToolExpectation: Equatable, Sendable {
        public var name: String
        /// Key arguments; arguments not listed are not scored.
        public var arguments: [String: JSONValue]

        public init(name: String, arguments: [String: JSONValue] = [:]) {
            self.name = name
            self.arguments = arguments
        }

        /// Scores a reply's first tool call; pass nil for both when it made none.
        public func score(name actualName: String?, arguments actual: JSONValue?) -> ToolScore {
            let nameMatches = actualName == name
            var results: [String: Bool] = [:]
            if actualName != nil {
                for (key, expected) in arguments {
                    results[key] = Self.matches(expected, actual?[key])
                }
            }
            return ToolScore(nameMatches: nameMatches, argumentsMatch: nameMatches && results.values.allSatisfy { $0 }, argumentResults: results)
        }

        /// Whether one argument matches:
        /// - a list matches when any of its items matches;
        /// - `YYYY-MM-DD` matches any time on that date; `YYYY-MM-DDTHH:MM` matches that date, hour
        ///   and minute, ignoring seconds and offsets;
        /// - a number matches the same number, also when given as a string;
        /// - any other string matches when each of its words appears in the value, ignoring case and
        ///   punctuation.
        static func matches(_ expected: JSONValue, _ actual: JSONValue?) -> Bool {
            if case .array(let options) = expected {
                return options.contains { matches($0, actual) }
            }
            guard let actual, !actual.isNull else { return false }
            let text = Self.text(of: actual)
            switch expected {
            case .bool(let value):
                return text.trimmingCharacters(in: .whitespaces).lowercased() == (value ? "true" : "false")
            case .int(let value):
                return Double(text.trimmingCharacters(in: .whitespaces)) == Double(value)
            case .double(let value):
                return Double(text.trimmingCharacters(in: .whitespaces)) == value
            case .string(let value):
                if let want = DateMatch.parse(value, whole: true) {
                    guard let got = DateMatch.parse(text, whole: false) else { return false }
                    if want.hour == nil {
                        return got.year == want.year && got.month == want.month && got.day == want.day
                    }
                    return got == want
                }
                let haystack = normalized(text)
                return normalized(value).split(whereSeparator: \.isWhitespace).allSatisfy { haystack.contains(String($0)) }
            case .null, .array, .object:
                return text == Self.text(of: expected)
            }
        }

        static func text(of value: JSONValue) -> String {
            switch value {
            case .string(let string): return string
            case .int(let number): return String(number)
            case .double(let number): return String(number)
            case .bool(let flag): return flag ? "true" : "false"
            case .null: return ""
            case .array, .object:
                return (try? value.serialized()).map { String(decoding: $0, as: UTF8.self) } ?? ""
            }
        }

        /// NFKC, lowercased, with punctuation and symbols replaced by spaces.
        static func normalized(_ text: String) -> String {
            var out = String.UnicodeScalarView()
            for scalar in text.precomposedStringWithCompatibilityMapping.lowercased().unicodeScalars {
                switch scalar.properties.generalCategory {
                case .connectorPunctuation, .dashPunctuation, .openPunctuation, .closePunctuation,
                     .initialPunctuation, .finalPunctuation, .otherPunctuation,
                     .mathSymbol, .currencySymbol, .modifierSymbol, .otherSymbol:
                    out.append(" ")
                default:
                    out.append(scalar)
                }
            }
            return String(out)
        }

        /// A date, and optionally a time, as `YYYY-M-D` followed by an optional `THH:MM` or ` HH:MM`.
        struct DateMatch: Equatable {
            var year: Int
            var month: Int
            var day: Int
            var hour: Int?
            var minute: Int?

            /// With `whole`, `text` must be exactly `YYYY-MM-DD` or `YYYY-MM-DDTHH:MM`. Otherwise the
            /// first date anywhere in `text`.
            static func parse(_ text: String, whole: Bool) -> DateMatch? {
                let scalars = Array(text.unicodeScalars)
                if whole {
                    guard let found = match(scalars, at: 0, strict: true), found.end == scalars.count else { return nil }
                    return found.date
                }
                for start in scalars.indices {
                    if let found = match(scalars, at: start, strict: false) { return found.date }
                }
                return nil
            }

            /// `strict` requires two-digit fields and `T` before the time.
            private static func match(_ scalars: [Unicode.Scalar], at start: Int, strict: Bool) -> (date: DateMatch, end: Int)? {
                func number(at index: Int, min: Int, max: Int) -> (value: Int, end: Int)? {
                    var end = index
                    while end < scalars.count, end - index < max, ("0" ... "9").contains(scalars[end]) { end += 1 }
                    guard end - index >= min else { return nil }
                    var digits = String.UnicodeScalarView()
                    digits.append(contentsOf: scalars[index ..< end])
                    guard let value = Int(String(digits)) else { return nil }
                    return (value, end)
                }
                func symbol(at index: Int, _ options: Set<Unicode.Scalar>) -> Bool {
                    index < scalars.count && options.contains(scalars[index])
                }
                let short = strict ? 2 : 1
                guard let year = number(at: start, min: 4, max: 4), symbol(at: year.end, ["-"]),
                      let month = number(at: year.end + 1, min: short, max: 2), symbol(at: month.end, ["-"]),
                      let day = number(at: month.end + 1, min: short, max: 2)
                else { return nil }
                var date = DateMatch(year: year.value, month: month.value, day: day.value)
                var end = day.end
                if symbol(at: day.end, strict ? ["T"] : ["T", " "]),
                   let hour = number(at: day.end + 1, min: short, max: 2), symbol(at: hour.end, [":"]),
                   let minute = number(at: hour.end + 1, min: 2, max: 2)
                {
                    date.hour = hour.value
                    date.minute = minute.value
                    end = minute.end
                }
                return (date, end)
            }
        }
    }

    /// How a reply's first tool call compared with a `ToolExpectation`.
    public struct ToolScore: Equatable, Sendable {
        public var nameMatches: Bool
        /// The name and every key argument matched.
        public var argumentsMatch: Bool
        /// Per key argument; empty when no tool was called.
        public var argumentResults: [String: Bool]

        public init(nameMatches: Bool, argumentsMatch: Bool, argumentResults: [String: Bool] = [:]) {
            self.nameMatches = nameMatches
            self.argumentsMatch = argumentsMatch
            self.argumentResults = argumentResults
        }
    }
}
