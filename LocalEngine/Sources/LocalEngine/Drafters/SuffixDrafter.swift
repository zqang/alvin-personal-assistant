import AssistantKit
import Foundation

/// Drafters the speculative loop tells when a reply has ended (with the reply's emitted tokens,
/// stop tokens excluded).
protocol ReplyObservingDrafter: AnyObject {
    func replyFinished(_ tokens: [Int])
}

/// Cross-session drafting (plan §4.7): past replies and tool results, kept in a `SuffixCorpus`
/// of at most 64k tokens that is saved on the device at `url`.
///
/// - The corpus is loaded lazily, on a background queue, at the first `reset`; until then the
///   drafter proposes nothing.
/// - After each reply the loop hands over the reply's tokens; at the next `reset`, the tool
///   results the engine prefilled since the previous reset (the spans between
///   `<tool_response>` and `</tool_response>`) are added too.
/// - Every change is applied and saved on the background queue, off the critical path
///   (written to a temporary file, then moved into place).
///
/// The corpus holds token ids, so `url` must be specific to one tokenizer (one model family).
/// It holds conversation text, so deleting a conversation erases it (`eraseAllCorpora`).
public final class SuffixDrafter: Drafter, ReplyObservingDrafter {
    public var source: DraftSource { .corpus }
    public var costPerToken: Double { 0 }
    public var wantsHidden: Bool { false }

    /// Where the corpus is saved; nil keeps it in memory only.
    public var url: URL? { store.url }
    /// Tokens never proposed (turn markers, role and other special tokens).
    public let excluded: Set<Int>
    /// The ids of `<tool_response>` and `</tool_response>`, when the vocabulary has them.
    public let toolResponseTags: (start: Int, end: Int)?

    private let store: SuffixCorpusStore
    /// The ledger at the previous `reset` (its length and last tokens), to find what was fed
    /// since.
    private var previousReset: (count: Int, tail: [Int])?

    public init(url: URL?, excluded: Set<Int>, toolResponseTags: (start: Int, end: Int)? = nil, capacityTokens: Int = 65_536) {
        self.excluded = excluded
        self.toolResponseTags = toolResponseTags
        self.store = SuffixCorpusStore(url: url, capacityTokens: capacityTokens)
    }

    /// A drafter that excludes `renderer`'s special and role tokens and `stopTokens`.
    public convenience init(url: URL?, renderer: any ChatTemplateRendering, stopTokens: Set<Int>, capacityTokens: Int = 65_536) {
        let tags = renderer.tokenID("<tool_response>").flatMap { start in
            renderer.tokenID("</tool_response>").map { (start: start, end: $0) }
        }
        let excluded = SpeculativeLoop.specialTokens(renderer: renderer, stopTokens: stopTokens)
            .union(PromptLookupDrafter.excludedTokens(renderer: renderer))
        self.init(url: url, excluded: excluded, toolResponseTags: tags, capacityTokens: capacityTokens)
    }

    /// The corpus as it is now (nil until loaded).
    public var corpus: SuffixCorpus? { store.current }

    /// Blocks until every queued load, change and save has finished (tests, shutdown).
    public func waitForBackgroundWork() {
        store.waitUntilIdle()
    }

    /// Erases every drafting corpus: empties the corpus of every `SuffixDrafter` in the process
    /// and deletes its file (once the changes queued before it are done), and deletes every
    /// other corpus file in `directory` (named `corpus-*.bin`, as the app names them). For when
    /// a conversation is deleted: the corpora hold past replies and tool results, which may be
    /// its own. Returns without waiting for the drafters' queues.
    public static func eraseAllCorpora(savedIn directory: URL) {
        for store in SuffixCorpusStore.liveStores() {
            store.erase()
        }
        let fileManager = FileManager.default
        let files = (try? fileManager.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)) ?? []
        for file in files where file.lastPathComponent.hasPrefix("corpus-") && file.pathExtension == "bin" {
            SuffixCorpusStore.remove(file)
        }
    }

    public func reset(ledger: [Int], request: EngineRequest) {
        store.loadIfNeeded()
        if let previousReset, let tags = toolResponseTags, Self.extends(ledger, previousReset) {
            let results = Self.spans(in: ledger[previousReset.count...], start: tags.start, end: tags.end)
            store.add(results)
        }
        previousReset = (ledger.count, Array(ledger.suffix(8)))
    }

    public func propose(context: ArraySlice<Int>, maxTokens: Int) -> DraftProposal? {
        guard maxTokens > 0 else { return nil }
        return store.propose(context: context, maxTokens: maxTokens, excluded: excluded)
    }

    public func observe(_ round: RoundObservation) {}

    func replyFinished(_ tokens: [Int]) {
        store.add([tokens])
    }

    /// Whether `ledger` still starts with the ledger of the previous reset (checked by its
    /// length and last tokens).
    private static func extends(_ ledger: [Int], _ previous: (count: Int, tail: [Int])) -> Bool {
        guard ledger.count >= previous.count else { return false }
        return Array(ledger[(previous.count - previous.tail.count) ..< previous.count]) == previous.tail
    }

    /// The token runs strictly between each `start` and the next `end`.
    static func spans(in tokens: ArraySlice<Int>, start: Int, end: Int) -> [[Int]] {
        var spans: [[Int]] = []
        var open: Int?
        for index in tokens.indices {
            if tokens[index] == start {
                open = index + 1
            } else if tokens[index] == end, let from = open {
                spans.append(Array(tokens[from ..< index]))
                open = nil
            }
        }
        return spans
    }
}

/// The corpus behind a `SuffixDrafter`. Reads (`propose`, `current`) take a lock and may come
/// from any thread; every change, load and save runs on the store's serial background queue,
/// which builds the new corpus on a copy and swaps it in, so readers never wait for indexing or
/// file I/O.
final class SuffixCorpusStore: @unchecked Sendable {
    let url: URL?
    let capacityTokens: Int
    private let queue = DispatchQueue(label: "alvin.engine.corpus", qos: .utility)
    private let lock = NSLock()
    private var corpus: SuffixCorpus?
    private var loadRequested = false

    init(url: URL?, capacityTokens: Int) {
        self.url = url
        self.capacityTokens = max(1, capacityTokens)
        Self.registry.add(self)
    }

    /// Every store alive in the process, for `SuffixDrafter.eraseAllCorpora(savedIn:)`.
    private static let registry = Registry()

    static func liveStores() -> [SuffixCorpusStore] {
        registry.stores()
    }

    /// Queues emptying the corpus and deleting its file, after the loads, changes and saves
    /// queued before.
    func erase() {
        queue.async {
            self.lock.lock()
            self.corpus = SuffixCorpus(capacityTokens: self.capacityTokens)
            self.lock.unlock()
            if let url = self.url {
                Self.remove(url)
            }
        }
    }

    /// Deletes the corpus file at `url`, if there is one.
    static func remove(_ url: URL) {
        let fileManager = FileManager.default
        guard fileManager.fileExists(atPath: url.path) else { return }
        do {
            try fileManager.removeItem(at: url)
        } catch {
            print("LocalEngine: couldn't delete the drafting corpus: \(error)")
        }
    }

    /// Weak references to the live stores.
    private final class Registry: @unchecked Sendable {
        private struct Entry {
            weak var store: SuffixCorpusStore?
        }

        private let lock = NSLock()
        private var entries: [Entry] = []

        func add(_ store: SuffixCorpusStore) {
            lock.lock()
            defer { lock.unlock() }
            entries = entries.filter { $0.store != nil } + [Entry(store: store)]
        }

        func stores() -> [SuffixCorpusStore] {
            lock.lock()
            defer { lock.unlock() }
            return entries.compactMap(\.store)
        }
    }

    var current: SuffixCorpus? {
        lock.lock()
        defer { lock.unlock() }
        return corpus
    }

    func propose(context: ArraySlice<Int>, maxTokens: Int, excluded: Set<Int>) -> DraftProposal? {
        lock.lock()
        defer { lock.unlock() }
        return corpus?.propose(context: context, maxTokens: maxTokens, excluded: excluded)
    }

    /// Queues the load from `url` (once).
    func loadIfNeeded() {
        lock.lock()
        let first = !loadRequested
        loadRequested = true
        lock.unlock()
        if first {
            queue.async { self.loadOnQueue() }
        }
    }

    /// Queues adding `documents` (shorter than 3 tokens are dropped) and saving the corpus.
    func add(_ documents: [[Int]]) {
        let documents = documents.filter { $0.count >= 3 }
        guard !documents.isEmpty else { return }
        loadIfNeeded()
        queue.async {
            self.loadOnQueue()
            guard var updated = self.current else { return }
            for document in documents {
                updated.add(document: document)
            }
            self.lock.lock()
            self.corpus = updated
            self.lock.unlock()
            self.save(updated)
        }
    }

    func waitUntilIdle() {
        queue.sync {}
    }

    /// Reads the saved corpus (or starts an empty one) unless it is loaded. Queue only.
    private func loadOnQueue() {
        guard current == nil else { return }
        var loaded = SuffixCorpus(capacityTokens: capacityTokens)
        if let url, let data = try? Data(contentsOf: url), let saved = SuffixCorpus(data: data) {
            if saved.capacityTokens == capacityTokens {
                loaded = saved
            } else {
                for document in saved.documents {
                    loaded.add(document: document)
                }
            }
        }
        lock.lock()
        corpus = loaded
        lock.unlock()
    }

    /// Writes `corpus` to a temporary file next to `url`, then moves it into place. Queue only.
    private func save(_ corpus: SuffixCorpus) {
        guard let url else { return }
        let fileManager = FileManager.default
        let directory = url.deletingLastPathComponent()
        do {
            try fileManager.createDirectory(at: directory, withIntermediateDirectories: true)
            try corpus.encoded().write(to: url, options: .atomic)
        } catch {
            print("LocalEngine: couldn't save the drafting corpus: \(error)")
        }
    }
}
