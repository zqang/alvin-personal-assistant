import AssistantKit
import MiniCPM
import SwiftUI

/// Settings › Photo reading: MiniCPM-V 4.7 1B reads the photos attached in chats and voice mode,
/// on this iPhone. The photo button appears once it's ready.
struct PhotoReadingSection: View {
    var body: some View {
        Section {
            MiniCPMModelRows(model: .photoReader, name: "the photo reader", downloadBytes: MiniCPMArtifacts.totalBytes)
        } header: {
            Text("Photo reading")
        } footer: {
            Text("MiniCPM-V 4.7 1B copies the text in photos you attach and says briefly what they show. Once it's downloaded, a photo button appears in chats and voice mode. Photos are read on this iPhone and never sent to a model provider; they stay in the chat on this iPhone until you delete the chat. Only the text it reads goes with your message, and leaves the iPhone only when Claude or another online model answers.")
        }
    }
}

/// Settings › Most accurate on device: MiniCPM-V 4.7 35B answers on-device replies in place of the
/// on-device model, which still handles reminders, calendar and timers and takes over when the
/// iPhone is hot. Offered only on iPhones with 12 GB of memory.
struct AccurateOnDeviceSection: View {
    @Bindable var store: SettingsStore

    private var host: MiniCPMHost { .shared }

    var body: some View {
        Section {
            if MiniCPMHost.accurateEligible() {
                Toggle("Most accurate on device", isOn: $store.settings.accurateOnDevice)
                    // It needs the model that answers whenever the 35B can't; always switchable off.
                    .disabled(!store.settings.accurateOnDevice && !fallbackDownloaded)
                if showsDownload {
                    MiniCPMModelRows(model: .accurate, name: "MiniCPM-V 35B", downloadBytes: MiniCPM35BArtifacts.totalBytes)
                }
            } else {
                LabeledContent("Most accurate on device", value: "Not on this iPhone")
            }
        } header: {
            Text("Most accurate on device")
        } footer: {
            Text(footer)
        }
    }

    /// The mode is on, or the 35B has files or a download to show.
    private var showsDownload: Bool {
        store.settings.accurateOnDevice
            || (host.status[.accurate] ?? .notDownloaded) != .notDownloaded
            || host.bytesOnDisk(.accurate) > 0
    }

    private var fallbackDownloaded: Bool {
        LocalModelHost.shared.isDownloaded(store.settings.localModelID)
    }

    private var footer: String {
        guard MiniCPMHost.accurateEligible() else {
            return "Most accurate on device needs an iPhone with 12 GB of memory, such as iPhone 17 Pro."
        }
        let fallback = LocalModelCatalog.option(for: store.settings.localModelID)?.displayName ?? "the on-device model"
        let size = ByteCountFormatter.string(fromByteCount: MiniCPM35BArtifacts.totalBytes, countStyle: .file)
        // The speed figures are estimates until the iPhone runs D21–D24 (architecture plan §11).
        let text = "MiniCPM-V 4.7 35B is a much larger model than \(fallback) and may handle harder questions better, but it's slower: its first words take 2–5 s, then about 5–6 tokens a second, and in a long conversation, every few replies, it first re-reads the latest turns, which can take a minute. It warms the iPhone, so \(fallback) answers instead when it's hot or in Low Power Mode, and always handles reminders, calendar and timers. Photos are still read by the photo reader. The download is about \(size), over Wi-Fi only; keep the app open while it downloads, and finished files are kept if you leave."
        return fallbackDownloaded ? text : "Download \(fallback) first: it answers whenever the 35B can't. " + text
    }
}

/// A MiniCPM-V model's state on this iPhone, with Download, Cancel and Delete and the bytes its
/// files take.
private struct MiniCPMModelRows: View {
    let model: MiniCPMHost.Model
    /// How the delete confirmation names it.
    let name: String
    /// What the download takes, from the model's manifest.
    let downloadBytes: Int64

    /// What the model's files take now: read when the rows appear and whenever the status changes.
    @State private var bytesOnDisk: Int64 = 0
    @State private var confirmingDelete = false

    private var host: MiniCPMHost { .shared }
    private var status: MiniCPMHost.Status { host.status[model] ?? .notDownloaded }

    var body: some View {
        Text(statusText)
            .font(.footnote)
            .foregroundStyle(.secondary)
            .onAppear { bytesOnDisk = host.bytesOnDisk(model) }
            .onChange(of: status) { bytesOnDisk = host.bytesOnDisk(model) }
        switch status {
        case .downloading, .verifying:
            Button("Cancel download") { host.cancelDownload(model) }
        case .notDownloaded:
            Button(bytesOnDisk > 0 ? "Resume download" : "Download (\(Self.size(downloadBytes)))") { host.download(model) }
            deleteButton
        case .failed:
            Button("Try again") { host.download(model) }
            deleteButton
        case .ready, .loading:
            deleteButton
        }
    }

    @ViewBuilder
    private var deleteButton: some View {
        if bytesOnDisk > 0 {
            Button("Delete (\(Self.size(bytesOnDisk)))", role: .destructive) { confirmingDelete = true }
                .confirmationDialog("Delete \(name)?", isPresented: $confirmingDelete, titleVisibility: .visible) {
                    Button("Delete", role: .destructive) {
                        Task {
                            _ = await host.delete(model)
                            bytesOnDisk = host.bytesOnDisk(model)
                        }
                    }
                } message: {
                    Text("This frees \(Self.size(bytesOnDisk)). Using it again means downloading about \(Self.size(downloadBytes)) over Wi-Fi.")
                }
        }
    }

    private var statusText: String {
        switch status {
        case .notDownloaded:
            return bytesOnDisk > 0
                ? "Partly downloaded: \(Self.size(bytesOnDisk)) of about \(Self.size(downloadBytes)). Finished files are kept."
                : "Not downloaded. It downloads over Wi-Fi only."
        case .downloading(let progress):
            return "Downloading over Wi-Fi… \(Int(progress * 100))%. Keep the app open; finished files are kept if you leave."
        case .verifying:
            return "Checking the downloaded files…"
        case .ready:
            return "Ready."
        case .loading:
            return "Loading…"
        case .failed(let message):
            return message
        }
    }

    private static func size(_ bytes: Int64) -> String {
        ByteCountFormatter.string(fromByteCount: bytes, countStyle: .file)
    }
}
