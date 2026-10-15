// A runnable check of AlvinAssistant/Chat/PendingPhoto.swift (the app has no test target). From the
// repository root, on a Mac:
//
//   swiftc -parse-as-library AlvinAssistant/Chat/PendingPhoto.swift scripts/check_pending_photo.swift \
//     -o /tmp/check_pending_photo && /tmp/check_pending_photo
//
// It prints "ALL OK", or the first failing check and exits 1.
import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers

// Stand-ins for the app's types that PendingPhoto uses.
enum ChatRole { case user, assistant }
final class ChatMessage {
    var role: ChatRole
    var photoData: Data?
    var photoNote: String?
    init(role: ChatRole, photoData: Data? = nil, photoNote: String? = nil) {
        self.role = role
        self.photoData = photoData
        self.photoNote = photoNote
    }
}
extension String { var trimmed: String { trimmingCharacters(in: .whitespacesAndNewlines) } }

@MainActor
func check(_ ok: Bool, _ what: String, line: Int = #line) {
    guard ok else {
        print("FAIL line \(line): \(what)")
        exit(1)
    }
    print("ok  \(what)")
}

/// A w x h HEIC with EXIF `orientation` and a GPS position.
func heic(_ w: Int, _ h: Int, orientation: Int = 1) -> Data {
    let context = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: 0, space: CGColorSpaceCreateDeviceRGB(),
                            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
    context.setFillColor(red: 1, green: 0, blue: 0, alpha: 1)
    context.fill(CGRect(x: 0, y: 0, width: w, height: h))
    let out = NSMutableData()
    let destination = CGImageDestinationCreateWithData(out, UTType.heic.identifier as CFString, 1, nil)!
    let gps: [CFString: Any] = [kCGImagePropertyGPSLatitude: 51.5, kCGImagePropertyGPSLatitudeRef: "N",
                                kCGImagePropertyGPSLongitude: 0.12, kCGImagePropertyGPSLongitudeRef: "W"]
    CGImageDestinationAddImage(destination, context.makeImage()!,
                               [kCGImagePropertyOrientation: orientation, kCGImagePropertyGPSDictionary: gps] as CFDictionary)
    CGImageDestinationFinalize(destination)
    return out as Data
}

/// Width, height and whether GPS metadata is present.
func properties(_ data: Data) -> (Int, Int, Bool) {
    let source = CGImageSourceCreateWithData(data as CFData, nil)!
    let values = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as! [CFString: Any]
    return (values[kCGImagePropertyPixelWidth] as! Int, values[kCGImagePropertyPixelHeight] as! Int, values[kCGImagePropertyGPSDictionary] != nil)
}

/// A reader that yields `pieces` (each after `delay`), then ends with `fail` or normally.
func reader(_ pieces: [String], fail: Error? = nil, delay: UInt64 = 0) -> (Data) -> AsyncThrowingStream<String, Error> {
    { _ in
        AsyncThrowingStream { continuation in
            let task = Task {
                for piece in pieces {
                    if delay > 0 { try? await Task.sleep(nanoseconds: delay) }
                    if Task.isCancelled { return continuation.finish(throwing: CancellationError()) }
                    continuation.yield(piece)
                }
                continuation.finish(throwing: fail)
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }
}

struct Boom: Error {}

@main
struct CheckPendingPhoto {
    @MainActor
    static func main() async throws {
        // Image preparation: long edge 1344, orientation applied, JPEG, no metadata; not an image -> nil.
        check(properties(heic(30, 20, orientation: 6)).2, "the source photo carries GPS")
        let big = await Task.detached { PendingPhoto(imageData: heic(3000, 2000, orientation: 6)) }.value!
        let (width, height, gps) = properties(big.jpeg)
        check(width == 896 && height == 1344, "3000x2000 rotated 90° becomes 896x1344 (got \(width)x\(height))")
        check(!gps, "no GPS metadata kept")
        check(UTType(CGImageSourceGetType(CGImageSourceCreateWithData(big.jpeg as CFData, nil)!)! as String) == .jpeg, "stored as JPEG")
        let (smallWidth, smallHeight, _) = properties(PendingPhoto(imageData: heic(200, 100))!.jpeg)
        check(smallWidth == 200 && smallHeight == 100, "a small photo isn't upscaled")
        check(PendingPhoto(imageData: Data("not an image".utf8)) == nil, "non-image data is refused")

        // Reading.
        let read = PendingPhoto(jpeg: Data([1]))
        read.startReading(reader(["Text: ", "EXIT ", "\n"]))
        check(await read.finalNote() == "Text: EXIT", "pieces join into the trimmed note")
        read.startReading(reader(["other"]))
        check(await read.finalNote() == "Text: EXIT", "a finished read doesn't run again")

        let failed = PendingPhoto(jpeg: Data([1]))
        failed.startReading(reader(["half"], fail: Boom()))
        check(await failed.finalNote() == PendingPhoto.unreadable && failed.failed, "a failed read gives the unreadable note")

        let empty = PendingPhoto(jpeg: Data([1]))
        empty.startReading(reader([" \n"]))
        check(await empty.finalNote() == PendingPhoto.unreadable, "an empty reading counts as unreadable")

        let cancelled = PendingPhoto(jpeg: Data([1]))
        cancelled.startReading(reader(["a", "b", "c"], delay: 50_000_000))
        cancelled.cancel()
        check(await cancelled.finalNote() == nil && !cancelled.finished, "a cancelled read has no note")
        cancelled.startReading(reader(["again"]))
        check(await cancelled.finalNote() == "again", "a cancelled read starts over")

        // The host stops a read (a Woof load, the app resigning active): the stream ends with
        // CancellationError while the photo's own task wasn't cancelled. The next start reads again.
        let stoppedByHost = PendingPhoto(jpeg: Data([1]))
        stoppedByHost.startReading(reader(["Text: TO"], fail: CancellationError()))
        check(await stoppedByHost.finalNote() == nil && !stoppedByHost.failed, "a reader-side stop is retryable, not a failure")
        var restarts = 0
        stoppedByHost.startReading { jpeg in
            restarts += 1
            return reader(["Text: TOTAL 8.75"])(jpeg)
        }
        check(restarts == 1, "after a reader-side stop, the next start reads again")
        check(await stoppedByHost.finalNote() == "Text: TOTAL 8.75", "and gives the new note")

        let waited = PendingPhoto(jpeg: Data([1]))
        waited.startReading(reader(["a", "b", "c", "d"], delay: 100_000_000))
        let waiter = Task { await waited.finalNote() }
        try? await Task.sleep(nanoseconds: 30_000_000)
        waiter.cancel()
        check(await waiter.value == nil, "cancelling the waiter cancels the read")
        waited.startReading(reader(["restarted"]))
        check(await waited.finalNote() == "restarted", "a restart right after a cancel isn't dropped")

        // noteUnread: an old photo whose read was cut short is read again; the attached one is reused.
        let old = ChatMessage(role: .user, photoData: Data([9]))
        let answered = ChatMessage(role: .assistant)
        let noted = ChatMessage(role: .user, photoData: Data([8]), photoNote: "kept")
        let plain = ChatMessage(role: .user)
        let sent = ChatMessage(role: .user, photoData: Data([7]))
        let attached = PendingPhoto(jpeg: Data([7]))
        attached.startReading(reader(["attached note"], delay: 20_000_000))
        var waits = 0
        var reads: [Data] = []
        try await PendingPhoto.noteUnread([old, answered, noted, plain, sent], attached: (sent, attached), read: { jpeg in
            reads.append(jpeg)
            return reader(["fresh note"])(jpeg)
        }) { waits += 1 }
        check(old.photoNote == "fresh note" && sent.photoNote == "attached note" && noted.photoNote == "kept" && plain.photoNote == nil,
              "notes filled: old re-read, attached reused, others untouched")
        check(reads == [Data([9])], "only the cut-short photo is read again (reads: \(reads))")
        check(waits == 1, "whileReading runs once (got \(waits))")

        // A stopped read is fatal only for the photo sent with this turn.
        let stoppedOld = ChatMessage(role: .user, photoData: Data([6]))
        let fresh = ChatMessage(role: .user, photoData: Data([5]))
        let freshPhoto = PendingPhoto(jpeg: Data([5]))
        freshPhoto.startReading(reader(["fresh photo"]))
        try await PendingPhoto.noteUnread([stoppedOld, fresh], attached: (fresh, freshPhoto),
                                          read: reader(["x"], fail: CancellationError())) {}
        check(stoppedOld.photoNote == nil && fresh.photoNote == "fresh photo", "an older photo's stopped read is left for a later turn")
        let stoppedAttached = ChatMessage(role: .user, photoData: Data([4]))
        let stoppedPhoto = PendingPhoto(jpeg: Data([4]))
        do {
            try await PendingPhoto.noteUnread([stoppedAttached], attached: (stoppedAttached, stoppedPhoto),
                                              read: reader(["x"], fail: CancellationError())) {}
            check(false, "the attached photo's stopped read throws")
        } catch {
            check(error is PhotoReadStopped && stoppedAttached.photoNote == nil, "the attached photo's stopped read throws PhotoReadStopped")
        }

        let question = PendingPhoto.defaultQuestion(localeIdentifier:)
        check(question("zh-Hans-CN") == "这张照片里有什么？" && question("zh-CN") == "这张照片里有什么？", "Simplified Chinese")
        check(["zh-TW", "zh-HK", "zh_Hant_HK", "yue-CN"].allSatisfy { question($0) == "這張照片裡有什麼？" }, "Traditional Chinese and Cantonese")
        check(question("en-US") == "What's in this photo?", "English otherwise")
        print("ALL OK")
    }
}
