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
/// iPhone is hot. Hidden in this version (`MiniCPMHost.accurateEligible`), so only files an earlier
/// build downloaded show, to be deleted.
struct AccurateOnDeviceSection: View {
    @Bindable var store: SettingsStore

    private var host: MiniCPMHost { .shared }

    var body: some View {
        if MiniCPMHost.accurateEligible() {
            Section {
                Toggle("Most accurate on device", isOn: $store.settings.accurateOnDevice)
                    // It needs the model that answers whenever the 35B can't; always switchable off.
                    .disabled(!store.settings.accurateOnDevice && !fallbackDownloaded)
                if showsDownload {
                    MiniCPMModelRows(model: .accurate, name: "MiniCPM-V 35B", downloadBytes: MiniCPM35BArtifacts.totalBytes)
                }
            } header: {
                Text("Most accurate on device")
            } footer: {
                Text(footer)
            }
        } else if showsDownload {
            // The mode isn't offered, but files from an earlier build must still be deletable.
            Section {
                MiniCPMModelRows(model: .accurate, name: "MiniCPM-V 35B", downloadBytes: MiniCPM35BArtifacts.totalBytes, allowsDownload: false)
            } header: {
                Text("MiniCPM-V 35B files")
            } footer: {
                Text("Most accurate on device isn't available in this version, so these files aren't used. Delete them to free the space.")
            }
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
        let fallback = LocalModelCatalog.option(for: store.settings.localModelID)?.displayName ?? "the on-device model"
        let size = ByteCountFormatter.string(fromByteCount: MiniCPM35BArtifacts.totalBytes, countStyle: .file)
        // Measured on an iPhone 17 Pro Max: speed in D22 and D23, the file check in D21 (architecture
        // plan §11). Heat and battery (D24) are still to be measured.
        let text = "MiniCPM-V 4.7 35B is a much larger model than \(fallback) and may handle harder questions better, but it's much slower. In a chat its first words take about 5 s, longer for a long message. In voice mode it starts speaking about 7–9 s after you finish, and up to about 12 s after you interrupt it. It then writes about 6 tokens a second. Its very first answer, and the first after you change your name or instructions, can take about 25 s. After a photo, and every few replies in a long conversation, it re-reads the conversation first, which can take a minute or more. It warms the iPhone, so \(fallback) answers instead when it's hot or in Low Power Mode, and always handles reminders, calendar and timers. Photos are still read by the photo reader. The download is about \(size), over Wi-Fi only; keep the app open while it downloads, and finished files are kept if you leave. Checking them afterwards takes about 10 s."
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
    /// False shows only the state and Delete, for files of a model this build doesn't offer.
    var allowsDownload = true

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
            if allowsDownload {
                Button(bytesOnDisk > 0 ? "Resume download" : "Download (\(Self.size(downloadBytes)))") { host.download(model) }
            }
            deleteButton
        case .failed:
            if allowsDownload { Button("Try again") { host.download(model) } }
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
