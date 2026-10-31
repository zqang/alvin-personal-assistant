import SwiftUI

/// How a host colours the voice screen: the gradient behind it, the orb's three colours per mood, and the controls.
/// `.standard` is VoiceKit's own look; a host passes its own through `.environment(\.voiceModeStyle, …)`.
public struct VoiceModeStyle: Sendable {
    /// Top-to-bottom gradient behind everything.
    public var background: [Color]
    /// Three colours per mood: the glow, the core, and the third blob.
    public var orb: [OrbMood: [Color]]
    /// Filled controls when active (the mute button while muted, the camera while open), and the text on them.
    public var accent: Color
    public var onAccent: Color
    /// Text, and the translucent fill of resting controls.
    public var ink: Color
    /// The end-call button.
    public var stop: Color
    /// How strongly the orb's three colours add up (they blend with light, so 1 burns the centre white where they
    /// meet) and how bright its top-left highlight is; both 0…1, 1 is the standard look.
    public var glow: Double
    public var highlight: Double

    public init(background: [Color], orb: [OrbMood: [Color]], accent: Color, onAccent: Color, ink: Color, stop: Color, glow: Double = 1, highlight: Double = 1) {
        self.background = background; self.orb = orb; self.accent = accent; self.onAccent = onAccent; self.ink = ink; self.stop = stop
        self.glow = min(max(glow, 0), 1); self.highlight = min(max(highlight, 0), 1)
    }

    public static let standard = VoiceModeStyle(
        background: [Color(red: 0.04, green: 0.05, blue: 0.11), .black],
        orb: [
            .idle: [Color(red: 0.35, green: 0.45, blue: 0.75), Color(red: 0.25, green: 0.3, blue: 0.6), Color(red: 0.45, green: 0.55, blue: 0.85)],
            .listening: [Color(red: 0.2, green: 0.75, blue: 1.0), Color(red: 0.3, green: 0.4, blue: 1.0), Color(red: 0.55, green: 0.95, blue: 0.95)],
            .thinking: [Color(red: 0.6, green: 0.35, blue: 1.0), Color(red: 0.95, green: 0.4, blue: 0.8), Color(red: 0.35, green: 0.45, blue: 1.0)],
            .speaking: [Color(red: 0.35, green: 0.6, blue: 1.0), Color(red: 0.4, green: 0.95, blue: 0.85), Color(red: 0.85, green: 0.9, blue: 1.0)],
            .error: [Color(red: 1.0, green: 0.4, blue: 0.35), Color(red: 0.9, green: 0.25, blue: 0.4), Color(red: 1.0, green: 0.65, blue: 0.4)],
        ],
        accent: .white, onAccent: .black, ink: .white, stop: .red)
}

extension EnvironmentValues {
    @Entry public var voiceModeStyle = VoiceModeStyle.standard
}

public enum OrbMood: Hashable {
    case idle
    case listening
    case thinking
    case speaking
    case error

    init(_ phase: VoiceSession.Phase) {
        switch phase {
        case .idle, .starting: self = .idle
        case .listening: self = .listening
        case .thinking: self = .thinking
        case .speaking: self = .speaking
        case .failed: self = .error
        }
    }

    fileprivate func palette(_ style: VoiceModeStyle) -> [Color] {
        let colors = style.orb[self] ?? VoiceModeStyle.standard.orb[self] ?? []
        return colors.count >= 3 ? colors : VoiceModeStyle.standard.orb[self]!
    }

    fileprivate var speed: Double {
        switch self {
        case .idle: return 0.35
        case .listening: return 0.8
        case .thinking: return 1.6
        case .speaking: return 1.1
        case .error: return 0.25
        }
    }
}

/// An animated orb that breathes, swirls, and swells with the voice level.
public struct OrbView: View {
    var level: CGFloat
    var mood: OrbMood
    @Environment(\.voiceModeStyle) private var style

    public init(level: CGFloat, mood: OrbMood) {
        self.level = level
        self.mood = mood
    }

    public var body: some View {
        TimelineView(.animation) { timeline in
            let time = timeline.date.timeIntervalSinceReferenceDate
            GeometryReader { proxy in
                let size = min(proxy.size.width, proxy.size.height)
                let colors = mood.palette(style)
                ZStack {
                    Circle()
                        .fill(RadialGradient(
                            colors: [colors[1].opacity(0.9), (style.background.last ?? .black).opacity(0.9)],
                            center: .center,
                            startRadius: 0,
                            endRadius: size * 0.55
                        ))
                    ForEach(0..<3, id: \.self) { index in
                        blob(index: index, color: colors[index].opacity(style.glow), time: time, size: size)
                    }
                    Circle()
                        .fill(RadialGradient(
                            colors: [style.ink.opacity((0.35 + 0.3 * level) * style.highlight), .clear],
                            center: UnitPoint(x: 0.35, y: 0.3),
                            startRadius: 0,
                            endRadius: size * 0.45
                        ))
                        .blendMode(.plusLighter)
                }
                .frame(width: size, height: size)
                .drawingGroup()
                .clipShape(Circle())
                .overlay(Circle().stroke(style.ink.opacity(0.18), lineWidth: 1))
                .shadow(color: colors[0].opacity(0.55), radius: size * (0.12 + 0.18 * level))
                .scaleEffect(scale(at: time))
                .frame(width: proxy.size.width, height: proxy.size.height)
            }
        }
        .accessibilityHidden(true)
    }

    private func blob(index: Int, color: Color, time: TimeInterval, size: CGFloat) -> some View {
        let speed = mood.speed * [0.7, 1.0, 1.35][index]
        let angle = time * speed + Double(index) * 2.1
        let radius = size * (0.12 + 0.16 * level)
        return Circle()
            .fill(color)
            .frame(width: size * 0.62, height: size * 0.62)
            .offset(x: CGFloat(cos(angle)) * radius, y: CGFloat(sin(angle * 1.3)) * radius)
            .blur(radius: size * 0.1)
            .blendMode(.plusLighter)
    }

    private func scale(at time: TimeInterval) -> CGFloat {
        switch mood {
        case .thinking:
            return 0.9 + 0.04 * CGFloat(sin(time * 2.6))
        case .listening, .speaking:
            return 0.88 + 0.22 * level + 0.015 * CGFloat(sin(time * 1.7))
        case .idle, .error:
            return 0.86 + 0.02 * CGFloat(sin(time * 1.1))
        }
    }
}

#Preview {
    OrbView(level: 0.4, mood: .speaking)
        .frame(width: 240, height: 240)
        .padding()
        .background(.black)
}
