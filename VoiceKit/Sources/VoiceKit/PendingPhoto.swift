import Foundation
import ImageIO
import Observation
import SwiftUI
import UniformTypeIdentifiers

/// A photo attached to the user's next turn, and what the on-device vision model reads in it.
/// The read starts when the photo is attached, so it overlaps the user's typing or speaking; the
/// turn's request then waits for the note (`noteUnread`), which goes into the turn's context.
@MainActor
@Observable
public final class PendingPhoto {
    /// The note of a photo whose read failed. It stays in English whatever the language: it's what
    /// the model reads, and hosts match it to read such a photo again.
    public static let unreadable = "The photo couldn't be read on this iPhone."
    /// The longest edge a photo keeps: three 448-px slices of the reader's grid.
    nonisolated static let maxPixelSize = 1344

    /// The photo as stored and read: upright, at most `maxPixelSize`, JPEG.
    public let jpeg: Data
    /// What the reader has written so far.
    public private(set) var note = ""
    /// The read ended, with `note` or in failure.
    public private(set) var finished = false
    public private(set) var failed = false
    @ObservationIgnored private var task: Task<Void, Never>?
    /// Which read is current, so a cancelled one can't write over its restart.
    @ObservationIgnored private var run = 0

    init(jpeg: Data) {
        self.jpeg = jpeg
    }

    /// Camera or library image data (HEIC, JPEG, PNG…) made ready to read. Nil when it isn't an
    /// image. Decoding can take a moment for a large photo, so it may run off the main actor.
    public nonisolated init?(imageData: Data) {
        guard let image = Self.downscaled(imageData, maxPixelSize: Self.maxPixelSize) else { return nil }
        let out = NSMutableData()
        guard let destination = CGImageDestinationCreateWithData(out, UTType.jpeg.identifier as CFString, 1, nil) else { return nil }
        // Only the pixels are written: no location or other metadata.
        CGImageDestinationAddImage(destination, image, [kCGImageDestinationLossyCompressionQuality: 0.8] as CFDictionary)
        guard CGImageDestinationFinalize(destination) else { return nil }
        jpeg = out as Data
    }

    /// Starts reading with `read`, unless a read is running or finished. A read that was cancelled,
    /// by the caller or by the reader (the host stopping it), starts over.
    public func startReading(_ read: (Data) -> AsyncThrowingStream<String, Error>) {
        guard !finished, task?.isCancelled ?? true else { return }
        run += 1
        let run = run
        note = ""
        let pieces = read(jpeg)
        task = Task { [weak self] in
            var failed = false
            var cancelled = false
            do {
                for try await piece in pieces {
                    guard let self, self.run == run else { return }
                    self.note += piece
                }
            } catch is CancellationError {
                cancelled = true
            } catch {
                failed = true
            }
            // A cancelled read (Stop, a barge-in, the app leaving the foreground, a Woof load)
            // runs again at the next `startReading`.
            guard let self, self.run == run else { return }
            if cancelled || Task.isCancelled {
                self.task = nil
                return
            }
            self.failed = failed || self.note.trimmed.isEmpty
            self.finished = true
        }
    }

    /// The finished note, or `unreadable` when the read failed. Nil when the read was cancelled or
    /// never started. Cancelling the caller cancels the read.
    func finalNote() async -> String? {
        guard let task else { return nil }
        await withTaskCancellationHandler { await task.value } onCancel: { task.cancel() }
        guard finished else { return nil }
        return failed ? Self.unreadable : note.trimmed
    }

    public func cancel() {
        task?.cancel()
    }

    /// What a photo sent without words asks, in the language the user speaks: Traditional Chinese
    /// for Taiwan, Hong Kong, Macau and Cantonese.
    public static func defaultQuestion(localeIdentifier: String) -> String {
        let id = localeIdentifier.lowercased().replacingOccurrences(of: "_", with: "-")
        guard id.hasPrefix("zh") || id.hasPrefix("yue") else { return "What's in this photo?" }
        let traditional = !id.contains("hans") && (id.hasPrefix("yue") || ["hant", "-tw", "-hk", "-mo"].contains { id.contains($0) })
        return traditional ? "這張照片裡有什麼？" : "这张照片里有什么？"
    }

    /// Gives each message in `messages` that has a photo but no note its note, oldest first (only
    /// user messages carry photos): `attached` is the photo sent with its `user` message, already
    /// being read; any other photo's read was cut short (Stop, a barge-in, a quit) and runs again.
    /// `whileReading` runs once, when a note isn't ready yet. Throws `PhotoReadStopped` when the
    /// attached photo's read was cancelled; another photo's stopped read is left for a later turn
    /// (the host refuses reads in the background, where voice replies still run).
    public static func noteUnread(
        _ messages: [any VoiceMessage],
        attached: (user: any VoiceMessage, photo: PendingPhoto)?,
        read: (Data) -> AsyncThrowingStream<String, Error>,
        whileReading: () -> Void
    ) async throws {
        var waited = false
        for message in messages where message.photoNote == nil {
            guard let jpeg = message.photoData else { continue }
            let photo = if let attached, attached.user === message { attached.photo } else { PendingPhoto(jpeg: jpeg) }
            photo.startReading(read)
            if !photo.finished, !waited {
                waited = true
                whileReading()
            }
            guard let note = await photo.finalNote() else {
                if attached?.user === message { throw PhotoReadStopped() }
                continue
            }
            // The turn may have been taken back meanwhile, and its message deleted.
            try Task.checkCancellation()
            message.photoNote = note
        }
    }

    /// `data` as an upright image at most `maxPixelSize` on its long edge; nil when it isn't an image.
    nonisolated static func downscaled(_ data: Data, maxPixelSize: Int) -> CGImage? {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil) else { return nil }
        let options: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceThumbnailMaxPixelSize: maxPixelSize,
        ]
        return CGImageSourceCreateThumbnailAtIndex(source, 0, options as CFDictionary)
    }
}

/// A photo's read stopped before it finished, so its turn has no note yet.
struct PhotoReadStopped: LocalizedError {
    var errorDescription: String? { String(localized: "Reading the photo stopped before it finished. Try again.", bundle: .voiceKit) }
}

/// A stored photo, decoded once at a small size.
public struct PhotoThumbnail: View {
    let data: Data
    var contentMode: ContentMode
    @State private var image: CGImage?

    public init(data: Data, contentMode: ContentMode = .fit) {
        self.data = data
        self.contentMode = contentMode
    }

    public var body: some View {
        Group {
            if let image {
                Image(image, scale: 1, label: Text("Attached photo", bundle: .voiceKit))
                    .resizable()
                    .aspectRatio(contentMode: contentMode)
            } else {
                Color.secondary.opacity(0.15)
            }
        }
        // Keyed by the data, so the composer's thumbnail follows a newly attached photo.
        .task(id: data) {
            let data = data
            image = await Task.detached(priority: .userInitiated) { PendingPhoto.downscaled(data, maxPixelSize: 480) }.value
        }
    }
}
