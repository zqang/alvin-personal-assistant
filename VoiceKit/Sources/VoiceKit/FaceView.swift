import SwiftUI

/// A glossy, emoji-like face in the orb's place (`VoiceAvatar.face`): a soft ball whose body takes the mood's
/// colour, with big low-set eyes, blush and a mouth that follows the voice. Drawn from shapes and gradients, so it
/// is sharp at any size and needs no artwork. On a light page it goes pastel with an ink outline, like a sticker.
///
/// Colours come from the style's three per-mood colours: the glow behind, the body, and the accent of the
/// thought bubbles and listening arcs.
public struct FaceView: View {
    var level: CGFloat
    var mood: OrbMood
    @Environment(\.voiceModeStyle) private var style
    @Environment(\.colorScheme) private var colorScheme
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    public init(level: CGFloat, mood: OrbMood) {
        self.level = level
        self.mood = mood
    }

    private var onPaper: Bool { colorScheme == .light }
    private var failed: Bool { mood == .error }
    /// Features are always dark: the style's ink on paper (it is dark there), a fixed plum-black on a dark page.
    private var ink: Color { onPaper ? style.ink : Color(red: 0.14, green: 0.1, blue: 0.17) }
    private static let blush = Color(red: 1, green: 0.56, blue: 0.64)

    /// The voice level the mouth and arcs follow: smooth, or three steps under Reduce Motion.
    private var shownLevel: CGFloat {
        guard reduceMotion else { return level }
        return level < 0.2 ? 0.1 : level < 0.6 ? 0.5 : 0.9
    }

    public var body: some View {
        TimelineView(.animation(paused: reduceMotion)) { timeline in
            let time = reduceMotion ? 0 : timeline.date.timeIntervalSinceReferenceDate
            GeometryReader { proxy in
                let s = min(proxy.size.width, proxy.size.height)
                let colors = mood.palette(style)
                let glow = colors[0], body = colors[1], accent = colors[2]
                ZStack {
                    if onPaper {
                        // A glow reads as a stain on paper; a small shadow sits the face on the page instead.
                        Ellipse().fill(style.ink.opacity(failed ? 0.18 : 0.28)).frame(width: s * 0.66, height: s * 0.1).offset(y: s * 0.44).blur(radius: s * 0.025)
                    } else if !failed {
                        Circle().fill(glow.opacity(0.55)).frame(width: s).blur(radius: s * 0.11)
                        Ellipse().fill(.black.opacity(0.5)).frame(width: s * 0.58, height: s * 0.08).offset(y: s * 0.45).blur(radius: s * 0.025)
                    } else {
                        Ellipse().fill(.black.opacity(0.5)).frame(width: s * 0.58, height: s * 0.08).offset(y: s * 0.45).blur(radius: s * 0.025)
                    }
                    ball(s, body: body)
                        .overlay { features(s, accent: accent) }
                        .saturation(failed ? 0.1 : 1)
                        .frame(width: s * 0.84, height: s * 0.84)
                        .clipShape(Circle())
                        .overlay {
                            if onPaper {
                                Circle().strokeBorder(style.ink.opacity(0.55), lineWidth: s * 0.012).frame(width: s * 0.84, height: s * 0.84)
                            }
                        }
                        .scaleEffect(x: 1.05, y: 0.97)
                        .scaleEffect(breath(at: time))
                    if mood == .thinking { thoughtDots(s, accent: accent, time: time) }
                    if mood == .listening { soundMarks(s, accent: accent) }
                }
                .frame(width: proxy.size.width, height: proxy.size.height)
                .animation(.easeInOut(duration: 0.25), value: mood)
            }
        }
        .accessibilityHidden(true)
    }

    // MARK: - Body

    /// The shaded ball: the body colour lit from the top left, a darker rim, a bounce light below, two highlights.
    private func ball(_ s: CGFloat, body: Color) -> some View {
        ZStack {
            Circle().fill(body)
            // Light to shade, as white and black over the body colour, so any palette shades the same way.
            Circle().fill(RadialGradient(stops: onPaper ? [
                .init(color: .white.opacity(0.75), location: 0),
                .init(color: .white.opacity(0.35), location: 0.5),
                .init(color: .clear, location: 0.9),
                .init(color: .clear, location: 1),
            ] : [
                .init(color: .white.opacity(0.55), location: 0),
                .init(color: .white.opacity(0.08), location: 0.45),
                .init(color: .black.opacity(0.22), location: 0.82),
                .init(color: .black.opacity(0.5), location: 1),
            ], center: UnitPoint(x: 0.38, y: 0.3), startRadius: 0, endRadius: s * 0.5))
            Circle().strokeBorder(AngularGradient(colors: [.clear, .clear, .white.opacity(0.7), .clear, .clear], center: .center, startAngle: .degrees(20), endAngle: .degrees(160)), lineWidth: s * 0.05)
                .blur(radius: s * 0.018)
            if !onPaper {
                Circle().strokeBorder(.black.opacity(0.35), lineWidth: s * 0.02).blur(radius: s * 0.015)
            }
            Circle().fill(RadialGradient(colors: [.white.opacity(0.3), .clear], center: UnitPoint(x: 0.5, y: 0.62), startRadius: 0, endRadius: s * 0.3)).blendMode(.screen)
            if !failed {
                HStack(spacing: s * 0.3) {
                    Ellipse().fill(Self.blush.opacity(onPaper ? 0.7 : 0.55))
                    Ellipse().fill(Self.blush.opacity(onPaper ? 0.7 : 0.55))
                }
                .frame(width: s * 0.7, height: s * 0.12).offset(y: s * 0.13).blur(radius: s * 0.028)
            }
            Ellipse().fill(.white.opacity(onPaper ? 0.6 : 0.5)).frame(width: s * 0.36, height: s * 0.2).rotationEffect(.degrees(-30)).offset(x: -s * 0.16, y: -s * 0.27).blur(radius: s * 0.03)
            Capsule().fill(.white.opacity(0.95)).frame(width: s * 0.2, height: s * 0.07).rotationEffect(.degrees(-30)).offset(x: -s * 0.17, y: -s * 0.3)
            Circle().fill(.white.opacity(0.85)).frame(width: s * 0.045).offset(x: -s * 0.02, y: -s * 0.37)
        }
    }

    private func breath(at time: TimeInterval) -> CGFloat {
        guard !reduceMotion else { return 1 }
        switch mood {
        case .listening: return 1 + 0.015 * CGFloat(sin(time * 1.3)) + 0.06 * level
        case .error: return 1
        default: return 1 + 0.02 * CGFloat(sin(time * 1.3))
        }
    }

    // MARK: - Face

    @ViewBuilder private func features(_ s: CGFloat, accent: Color) -> some View {
        eyes(s, accent: accent)
        mouth(s)
    }

    @ViewBuilder private func eyes(_ s: CGFloat, accent: Color) -> some View {
        let gap = s * 0.25
        switch mood {
        case .idle:
            HStack(spacing: gap) { happyEye(s); happyEye(s) }.offset(y: -s * 0.02)
        case .listening:
            HStack(spacing: gap) { bigEye(s, accent: accent, scale: 1.15); bigEye(s, accent: accent, scale: 1.15) }.offset(y: -s * 0.02)
        case .speaking:
            HStack(spacing: gap) { bigEye(s, accent: accent, scale: 1); bigEye(s, accent: accent, scale: 1) }.offset(y: -s * 0.03)
        case .thinking:
            let look = CGPoint(x: 0.3, y: -0.5)
            HStack(spacing: gap) { bigEye(s, accent: accent, scale: 0.95, look: look); bigEye(s, accent: accent, scale: 0.95, look: look) }.offset(y: -s * 0.02)
        case .error:
            HStack(spacing: gap) { bigEye(s, accent: accent, scale: 0.9, shine: 0.5); bigEye(s, accent: accent, scale: 0.9, shine: 0.5) }.offset(y: -s * 0.02)
            HStack(spacing: s * 0.2) {
                Capsule().fill(ink.opacity(0.75)).frame(width: s * 0.12, height: s * 0.03).rotationEffect(.degrees(-16))
                Capsule().fill(ink.opacity(0.75)).frame(width: s * 0.12, height: s * 0.03).rotationEffect(.degrees(16))
            }.offset(y: -s * 0.19)
            ZStack {
                Ellipse().fill(LinearGradient(colors: [Color(red: 0.81, green: 0.92, blue: 0.98), Color(red: 0.45, green: 0.74, blue: 0.9)], startPoint: .top, endPoint: .bottom))
                Circle().fill(.white.opacity(0.8)).frame(width: s * 0.014).offset(x: -s * 0.01, y: -s * 0.015)
            }.frame(width: s * 0.05, height: s * 0.085).offset(x: s * 0.2, y: s * 0.1)
        }
    }

    /// An eye with depth: a soft socket, a glint of the accent low in the iris, a lid shadow and two catchlights.
    private func bigEye(_ s: CGFloat, accent: Color, scale: CGFloat, look: CGPoint = .zero, shine: Double = 1) -> some View {
        let w = s * 0.13 * scale, h = s * 0.17 * scale
        return ZStack {
            Ellipse().fill(.black.opacity(onPaper ? 0.12 : 0.3)).frame(width: w * 1.15, height: h * 1.12).offset(y: h * 0.04).blur(radius: s * 0.012)
            Ellipse().fill(LinearGradient(colors: [ink, ink.opacity(0.85)], startPoint: .top, endPoint: .bottom))
            Ellipse().fill(RadialGradient(colors: [accent.opacity(0.9), accent.opacity(0)], center: .center, startRadius: 0, endRadius: w * 0.45))
                .frame(width: w * 0.8, height: h * 0.55).offset(y: h * 0.18)
            Ellipse().fill(LinearGradient(colors: [.black.opacity(0.45), .clear], startPoint: .top, endPoint: .center))
            Ellipse().fill(.white.opacity(0.96 * shine)).frame(width: w * 0.42, height: h * 0.34).offset(x: -w * 0.16 + look.x * s * 0.012, y: -h * 0.26 + look.y * s * 0.012)
            Circle().fill(.white.opacity(0.7 * shine)).frame(width: w * 0.18).offset(x: w * 0.2, y: h * 0.2)
        }
        .frame(width: w, height: h)
        .offset(x: look.x * s * 0.02, y: look.y * s * 0.03)
    }

    /// The content squint: a closed, upturned arc.
    private func happyEye(_ s: CGFloat) -> some View {
        Path { p in
            p.move(to: CGPoint(x: 0, y: s * 0.06)); p.addQuadCurve(to: CGPoint(x: s * 0.13, y: s * 0.06), control: CGPoint(x: s * 0.065, y: -s * 0.025))
        }
        .stroke(ink, style: StrokeStyle(lineWidth: s * 0.03, lineCap: .round))
        .frame(width: s * 0.13, height: s * 0.07)
    }

    @ViewBuilder private func mouth(_ s: CGFloat) -> some View {
        switch mood {
        case .idle:
            Path { p in
                p.move(to: CGPoint(x: 0, y: 0)); p.addQuadCurve(to: CGPoint(x: s * 0.08, y: 0), control: CGPoint(x: s * 0.04, y: s * 0.06))
                p.addQuadCurve(to: CGPoint(x: s * 0.16, y: 0), control: CGPoint(x: s * 0.12, y: s * 0.06))
            }
            .stroke(ink, style: StrokeStyle(lineWidth: s * 0.026, lineCap: .round, lineJoin: .round))
            .frame(width: s * 0.16, height: s * 0.05).offset(y: s * 0.2)
        case .listening:
            openMouth(s, width: 0.08, height: 0.07).offset(y: s * 0.21)
        case .speaking:
            openMouth(s, width: 0.17, height: 0.05 + 0.11 * shownLevel).offset(y: s * 0.21)
        case .thinking:
            Capsule().fill(ink).frame(width: s * 0.08, height: s * 0.026).rotationEffect(.degrees(-10)).offset(x: s * 0.05, y: s * 0.21)
        case .error:
            Path { p in
                p.move(to: CGPoint(x: 0, y: s * 0.04)); p.addQuadCurve(to: CGPoint(x: s * 0.14, y: s * 0.04), control: CGPoint(x: s * 0.07, y: -s * 0.02))
            }
            .stroke(ink.opacity(0.85), style: StrokeStyle(lineWidth: s * 0.026, lineCap: .round))
            .frame(width: s * 0.14, height: s * 0.05).offset(y: s * 0.22)
        }
    }

    /// A mouth with an inside: a dark cavity and a tongue.
    private func openMouth(_ s: CGFloat, width: CGFloat, height: CGFloat) -> some View {
        let w = s * width, h = s * height
        return ZStack(alignment: .bottom) {
            Ellipse().fill(LinearGradient(colors: [Color(red: 0.23, green: 0.08, blue: 0.19), Color(red: 0.11, green: 0.04, blue: 0.09)], startPoint: .top, endPoint: .bottom))
            Ellipse().fill(LinearGradient(colors: [Color(red: 0.98, green: 0.55, blue: 0.65), Color(red: 0.89, green: 0.38, blue: 0.5)], startPoint: .top, endPoint: .bottom))
                .frame(width: w * 0.62, height: h * 0.48).offset(y: -h * 0.04)
        }
        .frame(width: w, height: h)
        .clipShape(Ellipse())
    }

    // MARK: - Around the face

    /// Three thought bubbles that appear one by one (all at once under Reduce Motion).
    private func thoughtDots(_ s: CGFloat, accent: Color, time: TimeInterval) -> some View {
        let shown = reduceMotion ? 3 : Int((time * 1.5).truncatingRemainder(dividingBy: 4))
        return ZStack {
            ForEach(0..<3, id: \.self) { i in
                let d = s * (0.045 + 0.022 * CGFloat(i))
                ZStack {
                    Circle().fill(RadialGradient(colors: [.white, accent.opacity(0.6)], center: UnitPoint(x: 0.35, y: 0.3), startRadius: 0, endRadius: d * 0.7))
                    Circle().strokeBorder((onPaper ? style.ink : accent).opacity(0.5), lineWidth: s * (onPaper ? 0.006 : 0.004))
                }
                .frame(width: d, height: d)
                .opacity(i < shown ? 1 : 0)
                .offset(x: s * (0.3 + 0.07 * CGFloat(i)), y: -s * (0.3 + 0.085 * CGFloat(i)))
            }
        }
    }

    /// Three arcs at its ear that lengthen with the voice.
    private func soundMarks(_ s: CGFloat, accent: Color) -> some View {
        let span = 0.05 + 0.05 * shownLevel
        return ZStack {
            ForEach(0..<3, id: \.self) { i in
                let r = s * (0.5 + 0.07 * CGFloat(i))
                Circle().trim(from: 1 - span, to: 1 + span)
                    .stroke(accent.opacity(0.85 - 0.22 * Double(i)), style: StrokeStyle(lineWidth: s * 0.022, lineCap: .round))
                    .frame(width: r, height: r)
            }
        }
        .offset(x: s * 0.22)
    }
}

#Preview("Face, dark") {
    HStack {
        ForEach([OrbMood.idle, .listening, .thinking, .speaking, .error], id: \.self) { mood in
            FaceView(level: 0.6, mood: mood).frame(width: 120, height: 120)
        }
    }
    .padding()
    .background(.black)
    .environment(\.colorScheme, .dark)
}
