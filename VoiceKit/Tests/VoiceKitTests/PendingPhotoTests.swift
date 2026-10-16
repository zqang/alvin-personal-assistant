import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers
import XCTest
@testable import VoiceKit

/// A w x h photo with EXIF `orientation` and a GPS position: HEIC, or JPEG where this system can't
/// write HEIC.
private func photo(_ w: Int, _ h: Int, orientation: Int = 1) -> Data {
    let context = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: 0, space: CGColorSpaceCreateDeviceRGB(),
                            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
    context.setFillColor(red: 1, green: 0, blue: 0, alpha: 1)
    context.fill(CGRect(x: 0, y: 0, width: w, height: h))
    let gps: [CFString: Any] = [kCGImagePropertyGPSLatitude: 51.5, kCGImagePropertyGPSLatitudeRef: "N",
                                kCGImagePropertyGPSLongitude: 0.12, kCGImagePropertyGPSLongitudeRef: "W"]
    for type in [UTType.heic, .jpeg] {
        let out = NSMutableData()
        guard let destination = CGImageDestinationCreateWithData(out, type.identifier as CFString, 1, nil) else { continue }
        CGImageDestinationAddImage(destination, context.makeImage()!,
                                   [kCGImagePropertyOrientation: orientation, kCGImagePropertyGPSDictionary: gps] as CFDictionary)
        if CGImageDestinationFinalize(destination) { return out as Data }
    }
    fatalError("can't write a test photo")
}

/// Width, height and whether GPS metadata is present.
private func properties(_ data: Data) -> (width: Int, height: Int, gps: Bool) {
    let source = CGImageSourceCreateWithData(data as CFData, nil)!
    let values = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as! [CFString: Any]
    return (values[kCGImagePropertyPixelWidth] as! Int, values[kCGImagePropertyPixelHeight] as! Int, values[kCGImagePropertyGPSDictionary] != nil)
}

/// A reader that yields `pieces` (each after `delay`), then ends with `fail` or normally.
private func reader(_ pieces: [String], fail: Error? = nil, delay: UInt64 = 0) -> (Data) -> AsyncThrowingStream<String, Error> {
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

private struct Boom: Error {}

/// A user message, with a photo and its note.
@MainActor
private func user(photo: Data? = nil, note: String? = nil) -> Message {
    let message = Message(role: .user, text: "", isVoice: true, status: .complete)
    message.photoData = photo
    message.photoNote = note
    return message
}

@MainActor
final class PendingPhotoTests: XCTestCase {
    /// Long edge 1344, orientation applied, JPEG, no metadata; not an image gives nil.
    func testPreparingAPhoto() async throws {
        XCTAssertTrue(properties(photo(30, 20, orientation: 6)).gps, "the source photo carries GPS")
        let decoded = await Task.detached { PendingPhoto(imageData: photo(3000, 2000, orientation: 6)) }.value
        let big = try XCTUnwrap(decoded)
        let prepared = properties(big.jpeg)
        XCTAssertEqual([prepared.width, prepared.height], [896, 1344], "3000x2000 rotated 90° becomes 896x1344")
        XCTAssertFalse(prepared.gps, "no GPS metadata kept")
        let type = try XCTUnwrap(CGImageSourceGetType(CGImageSourceCreateWithData(big.jpeg as CFData, nil)!))
        XCTAssertEqual(UTType(type as String), .jpeg)
        let small = try properties(XCTUnwrap(PendingPhoto(imageData: photo(200, 100))).jpeg)
        XCTAssertEqual([small.width, small.height], [200, 100], "a small photo isn't upscaled")
        XCTAssertNil(PendingPhoto(imageData: Data("not an image".utf8)))
    }

    func testReading() async {
        let read = PendingPhoto(jpeg: Data([1]))
        read.startReading(reader(["Text: ", "EXIT ", "\n"]))
        let joined = await read.finalNote()
        XCTAssertEqual(joined, "Text: EXIT", "pieces join into the trimmed note")
        read.startReading(reader(["other"]))
        let again = await read.finalNote()
        XCTAssertEqual(again, "Text: EXIT", "a finished read doesn't run again")

        let failed = PendingPhoto(jpeg: Data([1]))
        failed.startReading(reader(["half"], fail: Boom()))
        let failedNote = await failed.finalNote()
        XCTAssertEqual(failedNote, PendingPhoto.unreadable)
        XCTAssertTrue(failed.failed)

        let empty = PendingPhoto(jpeg: Data([1]))
        empty.startReading(reader([" \n"]))
        let emptyNote = await empty.finalNote()
        XCTAssertEqual(emptyNote, PendingPhoto.unreadable, "an empty reading counts as unreadable")
    }

    func testACancelledReadStartsOver() async {
        let cancelled = PendingPhoto(jpeg: Data([1]))
        cancelled.startReading(reader(["a", "b", "c"], delay: 50_000_000))
        cancelled.cancel()
        let none = await cancelled.finalNote()
        XCTAssertNil(none)
        XCTAssertFalse(cancelled.finished)
        cancelled.startReading(reader(["again"]))
        let restarted = await cancelled.finalNote()
        XCTAssertEqual(restarted, "again")

        // Cancelling the waiter cancels the read, and a restart right after isn't dropped.
        let waited = PendingPhoto(jpeg: Data([1]))
        waited.startReading(reader(["a", "b", "c", "d"], delay: 100_000_000))
        let waiter = Task { await waited.finalNote() }
        try? await Task.sleep(nanoseconds: 30_000_000)
        waiter.cancel()
        let waiterNote = await waiter.value
        XCTAssertNil(waiterNote)
        waited.startReading(reader(["restarted"]))
        let afterWaiter = await waited.finalNote()
        XCTAssertEqual(afterWaiter, "restarted")
    }

    /// The host stops a read (a model load, the app resigning active): the stream ends with
    /// CancellationError while the photo's own task wasn't cancelled. The next start reads again.
    func testAReaderSideStopIsRetryable() async {
        let photo = PendingPhoto(jpeg: Data([1]))
        photo.startReading(reader(["Text: TO"], fail: CancellationError()))
        let stopped = await photo.finalNote()
        XCTAssertNil(stopped)
        XCTAssertFalse(photo.failed)
        var restarts = 0
        photo.startReading { jpeg in
            restarts += 1
            return reader(["Text: TOTAL 8.75"])(jpeg)
        }
        XCTAssertEqual(restarts, 1)
        let note = await photo.finalNote()
        XCTAssertEqual(note, "Text: TOTAL 8.75")
    }

    /// An old photo whose read was cut short is read again; the attached one is reused.
    func testNoteUnreadReadsOnlyCutShortPhotos() async throws {
        let old = user(photo: Data([9]))
        let answered = Message(role: .assistant, text: "", isVoice: true, status: .complete)
        let noted = user(photo: Data([8]), note: "kept")
        let plain = user()
        let sent = user(photo: Data([7]))
        let attached = PendingPhoto(jpeg: Data([7]))
        attached.startReading(reader(["attached note"], delay: 20_000_000))
        var waits = 0
        var reads: [Data] = []
        try await PendingPhoto.noteUnread([old, answered, noted, plain, sent], attached: (sent, attached), read: { jpeg in
            reads.append(jpeg)
            return reader(["fresh note"])(jpeg)
        }) { waits += 1 }
        XCTAssertEqual(old.photoNote, "fresh note")
        XCTAssertEqual(sent.photoNote, "attached note")
        XCTAssertEqual(noted.photoNote, "kept")
        XCTAssertNil(plain.photoNote)
        XCTAssertEqual(reads, [Data([9])], "only the cut-short photo is read again")
        XCTAssertEqual(waits, 1, "whileReading runs once")
    }

    /// A stopped read is fatal only for the photo sent with this turn.
    func testAStoppedReadThrowsOnlyForTheAttachedPhoto() async throws {
        let stoppedOld = user(photo: Data([6]))
        let fresh = user(photo: Data([5]))
        let freshPhoto = PendingPhoto(jpeg: Data([5]))
        freshPhoto.startReading(reader(["fresh photo"]))
        try await PendingPhoto.noteUnread([stoppedOld, fresh], attached: (fresh, freshPhoto),
                                          read: reader(["x"], fail: CancellationError())) {}
        XCTAssertNil(stoppedOld.photoNote, "an older photo's stopped read is left for a later turn")
        XCTAssertEqual(fresh.photoNote, "fresh photo")

        let stoppedAttached = user(photo: Data([4]))
        let stoppedPhoto = PendingPhoto(jpeg: Data([4]))
        do {
            try await PendingPhoto.noteUnread([stoppedAttached], attached: (stoppedAttached, stoppedPhoto),
                                              read: reader(["x"], fail: CancellationError())) {}
            XCTFail("the attached photo's stopped read throws")
        } catch {
            XCTAssertTrue(error is PhotoReadStopped)
            XCTAssertNil(stoppedAttached.photoNote)
        }
    }

    func testDefaultQuestionFollowsTheSpokenLanguage() {
        let question = PendingPhoto.defaultQuestion(localeIdentifier:)
        XCTAssertEqual(question("zh-Hans-CN"), "这张照片里有什么？")
        XCTAssertEqual(question("zh-CN"), "这张照片里有什么？")
        for id in ["zh-TW", "zh-HK", "zh_Hant_HK", "yue-CN"] {
            XCTAssertEqual(question(id), "這張照片裡有什麼？", id)
        }
        XCTAssertEqual(question("en-US"), "What's in this photo?")
    }
}
