import LocalEngine
import SwiftData
import SwiftUI

struct RootView: View {
    @Environment(\.modelContext) private var modelContext
    @Environment(SettingsStore.self) private var store
    @State private var conversation: Conversation?
    @State private var showingHistory = false
    @State private var showingSettings = false

    var body: some View {
        NavigationStack {
            Group {
                if let conversation {
                    ChatView(conversation: conversation, openSettings: { showingSettings = true })
                        .id(conversation.id)
                        .navigationTitle(conversation.title)
                } else {
                    ProgressView()
                }
            }
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Button {
                        showingHistory = true
                    } label: {
                        Image(systemName: "clock.arrow.circlepath")
                    }
                    .accessibilityLabel("History")
                }
                ToolbarItemGroup(placement: .topBarTrailing) {
                    Button {
                        startNewChat()
                    } label: {
                        Image(systemName: "square.and.pencil")
                    }
                    .accessibilityLabel("New chat")
                    Button {
                        showingSettings = true
                    } label: {
                        Image(systemName: "gearshape")
                    }
                    .accessibilityLabel("Settings")
                }
            }
        }
        .sheet(isPresented: $showingHistory) {
            HistoryView(
                current: conversation,
                onSelect: { selected in
                    conversation = selected
                    showingHistory = false
                },
                onDelete: { deleted in delete(deleted) }
            )
        }
        .sheet(isPresented: $showingSettings) {
            SettingsView(store: store)
        }
        .onAppear {
            guard conversation == nil else { return }
            removeEmptyConversations()
            startNewChat()
        }
    }

    private func startNewChat() {
        if let conversation, conversation.messages.isEmpty { return }
        let fresh = Conversation()
        modelContext.insert(fresh)
        conversation = fresh
    }

    private func delete(_ target: Conversation) {
        if target.id == conversation?.id {
            let fresh = Conversation()
            modelContext.insert(fresh)
            conversation = fresh
        }
        modelContext.delete(target)
        try? modelContext.save()
        // The on-device drafting corpus keeps past replies and tool results on disk, maybe this
        // conversation's: forget it too.
        SuffixDrafter.eraseAllCorpora(savedIn: EngineSetup.cacheDirectory())
    }

    /// New chats are created eagerly; drop the ones that were never used.
    private func removeEmptyConversations() {
        let all = (try? modelContext.fetch(FetchDescriptor<Conversation>())) ?? []
        for item in all where item.messages.isEmpty {
            modelContext.delete(item)
        }
    }
}

struct HistoryView: View {
    let current: Conversation?
    var onSelect: (Conversation) -> Void
    var onDelete: (Conversation) -> Void

    @Environment(\.dismiss) private var dismiss
    @Query(sort: \Conversation.updatedAt, order: .reverse) private var conversations: [Conversation]

    private var visible: [Conversation] {
        conversations.filter { !$0.messages.isEmpty }
    }

    var body: some View {
        NavigationStack {
            List {
                ForEach(visible) { conversation in
                    Button {
                        onSelect(conversation)
                    } label: {
                        VStack(alignment: .leading, spacing: 4) {
                            Text(conversation.title)
                                .font(.body.weight(conversation.id == current?.id ? .semibold : .regular))
                                .lineLimit(1)
                            Text(conversation.updatedAt, format: .relative(presentation: .named))
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                    }
                    .foregroundStyle(.primary)
                }
                .onDelete { offsets in
                    let targets = offsets.map { visible[$0] }
                    for target in targets {
                        onDelete(target)
                    }
                }
            }
            .overlay {
                if visible.isEmpty {
                    ContentUnavailableView(
                        "No conversations yet",
                        systemImage: "bubble.left.and.bubble.right",
                        description: Text("Your chats and voice conversations will appear here.")
                    )
                }
            }
            .navigationTitle("History")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                }
            }
        }
    }
}
