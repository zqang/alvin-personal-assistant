@preconcurrency import AVFoundation
import Observation
import SwiftUI
import UIKit

/// The back camera, video only, for attaching a photo. It leaves the app's audio session alone
/// (no audio input, and the session doesn't configure it), so voice mode keeps listening and
/// talking while the preview runs.
@MainActor
@Observable
final class CameraFeed {
    /// Why there's no preview, if there isn't.
    private(set) var problem: String?
    private(set) var isRunning = false
    @ObservationIgnored let session = AVCaptureSession()
    @ObservationIgnored private let output = AVCapturePhotoOutput()
    /// Starting and stopping the session block, so they run here, in order.
    @ObservationIgnored private let queue = DispatchQueue(label: "CameraFeed")
    @ObservationIgnored private var wanted = false
    @ObservationIgnored private var configured = false
    /// The capture in flight: the photo output doesn't keep its delegate.
    @ObservationIgnored private var capture: PhotoCapture?

    init() {
        session.automaticallyConfiguresApplicationAudioSession = false
    }

    func start() async {
        wanted = true
        guard await AVCaptureDevice.requestAccess(for: .video) else {
            problem = "Camera access is off. Turn it on in Settings › Privacy & Security › Camera."
            return
        }
        guard wanted, !isRunning else { return }
        if !configured {
            guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
                  let input = try? AVCaptureDeviceInput(device: device),
                  session.canAddInput(input), session.canAddOutput(output)
            else {
                problem = "The camera isn't available."
                return
            }
            session.beginConfiguration()
            session.sessionPreset = .photo
            session.addInput(input)
            session.addOutput(output)
            session.commitConfiguration()
            configured = true
        }
        isRunning = true
        let session = session
        queue.async { session.startRunning() }
    }

    func stop() {
        wanted = false
        isRunning = false
        let session = session
        queue.async { session.stopRunning() }
    }

    /// Takes a photo and returns its file data (HEIC or JPEG, with its orientation), or nil when
    /// the camera isn't running yet or the capture failed.
    func takePhoto() async -> Data? {
        guard isRunning, capture == nil, let connection = output.connection(with: .video), connection.isActive, connection.isEnabled else { return nil }
        // Portrait, the app's only orientation.
        if connection.isVideoRotationAngleSupported(90) {
            connection.videoRotationAngle = 90
        }
        let data = await withCheckedContinuation { continuation in
            let capture = PhotoCapture { continuation.resume(returning: $0) }
            self.capture = capture
            output.capturePhoto(with: AVCapturePhotoSettings(), delegate: capture)
        }
        capture = nil
        return data
    }
}

/// One capture's delegate: hands the photo's file data, or nil, to `done` once the capture ends.
private final class PhotoCapture: NSObject, AVCapturePhotoCaptureDelegate {
    private let done: (Data?) -> Void
    private var data: Data?

    init(done: @escaping (Data?) -> Void) {
        self.done = done
    }

    func photoOutput(_ output: AVCapturePhotoOutput, didFinishProcessingPhoto photo: AVCapturePhoto, error: Error?) {
        if error == nil { data = photo.fileDataRepresentation() }
    }

    /// Always the capture's last callback, exactly once.
    func photoOutput(_ output: AVCapturePhotoOutput, didFinishCaptureFor resolvedSettings: AVCaptureResolvedPhotoSettings, error: Error?) {
        done(data)
    }
}

/// A live back-camera preview with a shutter. The camera runs while the view is on screen, and
/// each photo taken goes to `onPhoto`.
struct CameraFeedView: View {
    var onPhoto: (Data) -> Void
    /// Made on first appearance: a capture session is too heavy to build on every redraw.
    @State private var feed: CameraFeed?
    @State private var capturing = false
    /// The last shutter press gave no photo (the camera wasn't running yet, or was interrupted).
    @State private var shotFailed = false

    var body: some View {
        ZStack(alignment: .bottom) {
            Color.black
            if let problem = feed?.problem {
                VStack(spacing: 12) {
                    Text(problem)
                        .font(.footnote)
                        .foregroundStyle(.white.opacity(0.85))
                        .multilineTextAlignment(.center)
                    if AVCaptureDevice.authorizationStatus(for: .video) == .denied,
                       let settings = URL(string: UIApplication.openSettingsURLString) {
                        Link("Open Settings", destination: settings)
                            .font(.footnote.weight(.semibold))
                    }
                }
                .padding()
                .frame(maxHeight: .infinity)
            } else if let feed {
                CameraPreview(session: feed.session)
                    .accessibilityLabel("Camera preview")
                VStack(spacing: 12) {
                    if shotFailed {
                        Text("Couldn't take the photo. Try again.")
                            .font(.footnote)
                            .foregroundStyle(.white)
                            .padding(.horizontal, 12)
                            .padding(.vertical, 6)
                            .background(.black.opacity(0.6), in: Capsule())
                    }
                    Button {
                        shoot(feed)
                    } label: {
                        Circle()
                            .fill(.white.opacity(capturing ? 0.4 : 0.95))
                            .padding(6)
                            .overlay(Circle().strokeBorder(.white, lineWidth: 4))
                            .frame(width: 68, height: 68)
                    }
                    .disabled(!feed.isRunning || capturing)
                    .accessibilityLabel("Take photo")
                }
                .padding(.bottom, 16)
            }
        }
        .task {
            let feed = feed ?? CameraFeed()
            self.feed = feed
            await feed.start()
        }
        .onDisappear { feed?.stop() }
    }

    private func shoot(_ feed: CameraFeed) {
        capturing = true
        shotFailed = false
        Task {
            if let data = await feed.takePhoto() {
                onPhoto(data)
            } else {
                shotFailed = true
            }
            capturing = false
        }
    }
}

private struct CameraPreview: UIViewRepresentable {
    let session: AVCaptureSession

    func makeUIView(context: Context) -> PreviewView {
        let view = PreviewView()
        view.previewLayer.session = session
        view.previewLayer.videoGravity = .resizeAspectFill
        return view
    }

    func updateUIView(_ view: PreviewView, context: Context) {}

    final class PreviewView: UIView {
        override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
        var previewLayer: AVCaptureVideoPreviewLayer { layer as! AVCaptureVideoPreviewLayer }
    }
}
