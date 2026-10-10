import AssistantKit
import SwiftUI
import UIKit

struct ChatView: View {
    let conversation: Conversation
    var openSettings: () -> Void

    @Environment(SettingsStore.self) private var store
    @State private var controller = ChatController()
    @State private var draft = ""
    /// "Think deeper" is on for the next message.
    @State private var thinkDeeper = false
    @State private var voiceSession: VoiceSession?
    @FocusState private var composerFocused: Bool

    var body: some View {
        let messages = conversation.orderedMessages
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 18) {
                    if messages.isEmpty {
                        WelcomeView(name: store.settings.userName, startVoice: { startVoice() })
                    }
                    ForEach(messages) { message in
                        MessageRow(
                            message: message,
                            activity: message.status == .streaming ? controller.activity : nil,
                            routedEngine: controller.engine(for: message),
                            // Only a failed reply offers Retry; this reads its tool rounds.
                            canRetry: message.status != .failed || controller.canRetry(message),
                            onRetry: { controller.retry(message, in: conversation, store: store) }
                        )
                        .id(message.id)
                    }
                }
                .padding(.horizontal)
                .padding(.vertical, 12)
            }
            .scrollDismissesKeyboard(.interactively)
            .defaultScrollAnchor(.bottom)
            .onChange(of: messages.last?.text) {
                if let last = messages.last?.id {
                    proxy.scrollTo(last, anchor: .bottom)
                }
            }
        }
        .safeAreaInset(edge: .bottom) { composer }
        .fullScreenCover(item: $voiceSession) { session in
            VoiceModeView(session: session)
        }
        .onChange(of: composerFocused) {
            // Opening the composer warms the connection, and the on-device model when it may answer.
            if composerFocused {
                pipeline.prewarm(inputIsVoice: false)
            }
        }
    }

    /// The reply pipeline, shared with the controller.
    private var pipeline: any ReplyPipeline {
        controller.pipeline(for: store)
    }

    /// Whether "Think deeper" is offered: the pipeline routes deep replies, Claude answers, and
    /// deep mode isn't off.
    private var canThinkDeeper: Bool {
        !(pipeline is LegacyReplyPipeline) && store.settings.provider == .anthropic && store.settings.deepMode != .off
    }

    private var composer: some View {
        VStack(spacing: 8) {
            if let problem = pipeline.missingSetup() {
                Button {
                    openSettings()
                } label: {
                    Label(problem, systemImage: "key.fill")
                        .font(.footnote.weight(.medium))
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(10)
                        .background(Color.orange.opacity(0.15), in: RoundedRectangle(cornerRadius: 12, style: .continuous))
                }
                .buttonStyle(.plain)
            }
            if canThinkDeeper {
                thinkDeeperToggle
            }
            HStack(alignment: .bottom, spacing: 10) {
                TextField("Message", text: $draft, axis: .vertical)
                    .lineLimit(1...6)
                    .focused($composerFocused)
                    .submitLabel(.send)
                    .onSubmit { send() }
                    .padding(.horizontal, 14)
                    .padding(.vertical, 10)
                    .background(Color(uiColor: .secondarySystemBackground), in: RoundedRectangle(cornerRadius: 22, style: .continuous))
                trailingButton
                    .font(.system(size: 34))
                    .symbolRenderingMode(.hierarchical)
            }
        }
        .padding(.horizontal)
        .padding(.vertical, 8)
        .background(.bar)
    }

    @ViewBuilder
    private var trailingButton: some View {
        if controller.isResponding {
            Button {
                controller.stop()
            } label: {
                Image(systemName: "stop.circle.fill")
            }
            .accessibilityLabel("Stop generating")
        } else if draft.trimmed.isEmpty {
            Button {
                startVoice()
            } label: {
                Image(systemName: "waveform.circle.fill")
            }
            .accessibilityLabel("Start voice chat")
        } else {
            Button {
                send()
            } label: {
                Image(systemName: "arrow.up.circle.fill")
            }
            .accessibilityLabel("Send")
        }
    }

    /// "Think deeper" for the next message, with what's left of today's deep replies.
    private var thinkDeeperToggle: some View {
        let left = store.deepRunsLeft()
        return HStack(spacing: 8) {
            Toggle(isOn: $thinkDeeper) {
                Label("Think deeper", systemImage: "brain")
                    .font(.footnote.weight(.medium))
            }
            .toggleStyle(.button)
            .buttonBorderShape(.capsule)
            .controlSize(.small)
            .disabled(left == 0)
            .accessibilityHint("The next message gets a slower, more thorough answer.")
            Text(left == 0 ? "None left today" : "\(left) left today")
                .font(.caption)
                .foregroundStyle(.secondary)
            Spacer(minLength: 0)
        }
    }

    private func send() {
        let text = draft
        guard !text.trimmed.isEmpty else { return }
        let deep = thinkDeeper && canThinkDeeper
        let sent = deep
            ? controller.sendDeep(text, in: conversation, store: store)
            : controller.send(text, in: conversation, store: store)
        guard sent else { return }
        draft = ""
        if deep {
            thinkDeeper = false
        }
    }

    private func startVoice() {
        composerFocused = false
        voiceSession = VoiceSession(conversation: conversation, store: store)
    }
}

struct MessageRow: View {
    let message: ChatMessage
    var activity: String?
    /// The engine that answered, when known; "On device" shows for the on-device model.
    var routedEngine: ReplyEngine? = nil
    /// False for a failed reply that called a tool that acts (a reminder, an event, a timer):
    /// answering again would repeat the action, so it offers no Retry.
    var canRetry: Bool = true
    var onRetry: () -> Void

    var body: some View {
        switch message.role {
        case .user:
            userBubble
        case .assistant:
            assistantBody
        }
    }

    private var userBubble: some View {
        HStack {
            Spacer(minLength: 48)
            VStack(alignment: .trailing, spacing: 4) {
                Text(message.text)
                    .padding(.horizontal, 14)
                    .padding(.vertical, 10)
                    .foregroundStyle(.white)
                    .background(Color.accentColor, in: RoundedRectangle(cornerRadius: 20, style: .continuous))
                if message.isVoice {
                    Label("Spoken", systemImage: "waveform")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
            }
        }
        .contextMenu { copyButton }
    }

    private var assistantBody: some View {
        let rounds = message.toolRounds
        return VStack(alignment: .leading, spacing: 8) {
            if !rounds.isEmpty {
                // What the reply did with tools, ahead of what it said about it.
                ToolRoundChip(rounds: rounds)
            }
            switch message.status {
            case .streaming where message.text.isEmpty:
                HStack(spacing: 8) {
                    Image(systemName: "ellipsis")
                        .symbolEffect(.variableColor.iterative, options: .repeating)
                    if let activity {
                        Text(activity)
                    }
                }
                .foregroundStyle(.secondary)
            case .refused:
                Label(message.errorText ?? "The model declined to answer this.", systemImage: "hand.raised")
                    .foregroundStyle(.secondary)
            case .failed:
                if !message.text.isEmpty {
                    MarkdownText(message.text)
                }
                Label(message.errorText ?? "Something went wrong.", systemImage: "exclamationmark.triangle")
                    .foregroundStyle(.red)
                if canRetry {
                    Button("Retry") { onRetry() }
                        .buttonStyle(.bordered)
                } else {
                    Text("Retrying would repeat the actions above. Send a message to carry on.")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }
            default:
                if !message.text.isEmpty {
                    MarkdownText(message.text)
                }
                if let activity {
                    Label(activity, systemImage: "globe")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }
                if message.status == .interrupted {
                    Text("Interrupted")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            if routedEngine == .local {
                Label("On device", systemImage: "iphone")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .contextMenu { copyButton }
    }

    private var copyButton: some View {
        Button {
            UIPasteboard.general.string = message.text
        } label: {
            Label("Copy", systemImage: "doc.on.doc")
        }
    }
}

/// Renders inline Markdown (bold, italics, code, links) and keeps line breaks.
struct MarkdownText: View {
    let text: String

    init(_ text: String) {
        self.text = text
    }

    var body: some View {
        Text(Self.attributed(text))
            .textSelection(.enabled)
    }

    private static func attributed(_ text: String) -> AttributedString {
        let options = AttributedString.MarkdownParsingOptions(interpretedSyntax: .inlineOnlyPreservingWhitespace)
        return (try? AttributedString(markdown: text, options: options)) ?? AttributedString(text)
    }
}

struct WelcomeView: View {
    var name: String
    var startVoice: () -> Void

    var body: some View {
        VStack(spacing: 18) {
            OrbView(level: 0.15, mood: .idle)
                .frame(width: 130, height: 130)
            Text(name.trimmed.isEmpty ? "Hi there" : "Hi, \(name.trimmed)")
                .font(.largeTitle.bold())
            Text("Talk to me or type a message. I can search the web, answer questions, and help you think things through.")
                .multilineTextAlignment(.center)
                .foregroundStyle(.secondary)
            Button {
                startVoice()
            } label: {
                Label("Start voice chat", systemImage: "waveform")
                    .font(.headline)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 6)
            }
            .buttonStyle(.borderedProminent)
            .buttonBorderShape(.capsule)
        }
        .frame(maxWidth: .infinity)
        .padding(.top, 48)
        .padding(.horizontal, 24)
    }
}
