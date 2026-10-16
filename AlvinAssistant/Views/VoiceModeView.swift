import SwiftUI

/// Full-screen, hands-free voice conversation.
struct VoiceModeView: View {
    let session: VoiceSession
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        ZStack {
            LinearGradient(
                colors: [Color(red: 0.04, green: 0.05, blue: 0.11), .black],
                startPoint: .top,
                endPoint: .bottom
            )
            .ignoresSafeArea()

            VStack(spacing: 0) {
                header
                Spacer(minLength: 24)
                if session.cameraOpen {
                    // The microphone keeps listening while the camera is open.
                    CameraFeedView { data in
                        Task { await session.attachPhoto(data) }
                    }
                    .aspectRatio(3 / 4, contentMode: .fit)
                    .frame(maxHeight: 340)
                    .clipShape(RoundedRectangle(cornerRadius: 24, style: .continuous))
                    orb
                        .frame(width: 96, height: 96)
                        .padding(.top, 16)
                } else {
                    orb
                        .frame(width: 250, height: 250)
                }
                Text(hintText ?? " ")
                    .font(.footnote)
                    .foregroundStyle(.white.opacity(0.45))
                    .padding(.top, 20)
                #if DEBUG
                if let readout = session.latencyReadout {
                    Text(readout)
                        .font(.caption2.monospacedDigit())
                        .foregroundStyle(.white.opacity(0.35))
                        .padding(.top, 6)
                }
                #endif
                Spacer(minLength: 24)
                captions
                    .frame(maxWidth: .infinity, minHeight: 150, alignment: .top)
                controls
            }
            .padding(.horizontal, 24)
            .padding(.bottom, 12)
        }
        .preferredColorScheme(.dark)
        .sensoryFeedback(.impact(weight: .light), trigger: session.phase)
        .task { await session.start() }
        .onDisappear { session.stop() }
    }

    private var orb: some View {
        OrbView(level: CGFloat(session.level), mood: OrbMood(session.phase))
            .contentShape(Circle())
            .onTapGesture { session.tapOrb() }
            .accessibilityElement()
            .accessibilityLabel(statusText)
            .accessibilityHint(hintText ?? "")
            .accessibilityAddTraits(.isButton)
    }

    private var header: some View {
        VStack(spacing: 4) {
            Text(statusText)
                .font(.headline)
                .foregroundStyle(.white)
                .contentTransition(.opacity)
                .animation(.easeInOut(duration: 0.2), value: statusText)
            Text(session.modelName)
                .font(.caption)
                .foregroundStyle(.white.opacity(0.5))
        }
        .frame(maxWidth: .infinity)
        .overlay(alignment: .trailing) {
            if session.canAttachPhoto, session.phase != .idle, session.phase != .starting, !isFailed {
                Button {
                    session.toggleCamera()
                } label: {
                    Image(systemName: session.cameraOpen ? "camera.fill" : "camera")
                        .font(.title3)
                        .frame(width: 44, height: 44)
                        .background(Circle().fill(session.cameraOpen ? Color.white : Color.white.opacity(0.14)))
                        .foregroundStyle(session.cameraOpen ? Color.black : Color.white)
                }
                .accessibilityLabel(session.cameraOpen ? "Close camera" : "Open camera")
                .accessibilityHint("Show the assistant a photo while you talk")
            }
        }
        .padding(.top, 16)
    }

    private var isFailed: Bool {
        if case .failed = session.phase { return true }
        return false
    }

    /// The photo going with the user's next turn, and how its read is going.
    private func pendingPhotoChip(_ photo: PendingPhoto) -> some View {
        HStack(spacing: 10) {
            PhotoThumbnail(data: photo.jpeg, contentMode: .fill)
                .frame(width: 40, height: 40)
                .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
            Text(photo.failed ? PendingPhoto.unreadable : photo.finished ? "Photo ready for your question" : VoiceSession.readingPhoto)
                .font(.footnote)
                .foregroundStyle(.white.opacity(0.75))
                .lineLimit(2)
            Spacer(minLength: 0)
            Button {
                session.removePhoto()
            } label: {
                Image(systemName: "xmark.circle.fill")
                    .font(.title3)
                    .foregroundStyle(.white.opacity(0.6))
            }
            .accessibilityLabel("Remove photo")
        }
        .padding(8)
        .background(RoundedRectangle(cornerRadius: 14, style: .continuous).fill(Color.white.opacity(0.08)))
    }

    @ViewBuilder
    private var captions: some View {
        VStack(spacing: 14) {
            if case .failed(let message) = session.phase {
                Text(message)
                    .foregroundStyle(.white.opacity(0.85))
                    .multilineTextAlignment(.center)
                Button {
                    Task { await session.restart() }
                } label: {
                    Label("Try again", systemImage: "arrow.clockwise")
                }
                .buttonStyle(.bordered)
                .tint(.white)
            } else {
                if let photo = session.pendingPhoto {
                    pendingPhotoChip(photo)
                }
                if let activity = session.activity {
                    Label(activity, systemImage: activity == VoiceSession.readingPhoto ? "text.viewfinder" : "globe")
                        .font(.subheadline)
                        .foregroundStyle(.white.opacity(0.7))
                        .symbolEffect(.pulse, options: .repeating)
                        .multilineTextAlignment(.center)
                }
                if session.phase == .speaking, !session.assistantCaption.isEmpty {
                    Text(session.assistantCaption)
                        .font(.title3)
                        .foregroundStyle(.white)
                        .multilineTextAlignment(.center)
                        .lineLimit(5)
                        .minimumScaleFactor(0.8)
                        .id(session.assistantCaption)
                        .transition(.opacity)
                } else if !session.userCaption.isEmpty {
                    Text(session.userCaption)
                        .font(.title3)
                        .foregroundStyle(.white.opacity(session.phase == .listening ? 0.9 : 0.55))
                        .multilineTextAlignment(.center)
                        .lineLimit(5)
                        .minimumScaleFactor(0.8)
                }
                if let notice = session.notice {
                    Label(notice, systemImage: "exclamationmark.triangle")
                        .font(.footnote)
                        .foregroundStyle(.yellow.opacity(0.85))
                        .multilineTextAlignment(.center)
                }
            }
        }
        .animation(.easeInOut(duration: 0.25), value: session.assistantCaption)
    }

    private var controls: some View {
        HStack {
            Button {
                session.toggleMute()
            } label: {
                Image(systemName: session.isMuted ? "mic.slash.fill" : "mic.fill")
                    .font(.title2)
                    .frame(width: 64, height: 64)
                    .background(Circle().fill(session.isMuted ? Color.white : Color.white.opacity(0.14)))
                    .foregroundStyle(session.isMuted ? Color.black : Color.white)
            }
            .accessibilityLabel(session.isMuted ? "Unmute microphone" : "Mute microphone")

            Spacer()

            if session.canGoDeep {
                Button {
                    session.requestDeepNextTurn(!session.deepNextTurn)
                } label: {
                    Label("Think deeper", systemImage: "brain.head.profile")
                        .font(.subheadline.weight(.semibold))
                        .lineLimit(1)
                        .minimumScaleFactor(0.8)
                        .padding(.horizontal, 16)
                        .frame(height: 44)
                        .background(Capsule().fill(session.deepNextTurn ? Color.white : Color.white.opacity(0.14)))
                        .foregroundStyle(session.deepNextTurn ? Color.black : Color.white)
                }
                .accessibilityLabel("Think deeper")
                .accessibilityValue(session.deepNextTurn ? "On for the next answer" : "Off")
                .accessibilityHint("Takes more time to answer your next question carefully")

                Spacer()
            }

            Button {
                dismiss()
            } label: {
                Image(systemName: "xmark")
                    .font(.title2.weight(.semibold))
                    .frame(width: 64, height: 64)
                    .background(Circle().fill(Color.red))
                    .foregroundStyle(.white)
            }
            .accessibilityLabel("End voice chat")
        }
        .padding(.horizontal, 12)
    }

    private var statusText: String {
        switch session.phase {
        case .idle, .starting: return "Connecting…"
        case .listening: return session.isMuted ? "Muted" : "Listening…"
        case .thinking: return "Thinking…"
        case .speaking: return "Speaking"
        case .failed: return "Voice chat stopped"
        }
    }

    private var hintText: String? {
        switch session.phase {
        case .listening where !session.userCaption.isEmpty: return "Tap the orb to send now"
        case .listening where session.pendingPhoto != nil: return "Ask about the photo, or tap the orb"
        case .listening: return "Go ahead, I'm listening"
        case .thinking, .speaking: return "Tap the orb or just start talking to interrupt"
        default: return nil
        }
    }
}
