import SwiftUI

enum OrbMood: Equatable {
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

    fileprivate var palette: [Color] {
        switch self {
        case .idle:
            return [Color(red: 0.35, green: 0.45, blue: 0.75), Color(red: 0.25, green: 0.3, blue: 0.6), Color(red: 0.45, green: 0.55, blue: 0.85)]
        case .listening:
            return [Color(red: 0.2, green: 0.75, blue: 1.0), Color(red: 0.3, green: 0.4, blue: 1.0), Color(red: 0.55, green: 0.95, blue: 0.95)]
        case .thinking:
            return [Color(red: 0.6, green: 0.35, blue: 1.0), Color(red: 0.95, green: 0.4, blue: 0.8), Color(red: 0.35, green: 0.45, blue: 1.0)]
        case .speaking:
            return [Color(red: 0.35, green: 0.6, blue: 1.0), Color(red: 0.4, green: 0.95, blue: 0.85), Color(red: 0.85, green: 0.9, blue: 1.0)]
        case .error:
            return [Color(red: 1.0, green: 0.4, blue: 0.35), Color(red: 0.9, green: 0.25, blue: 0.4), Color(red: 1.0, green: 0.65, blue: 0.4)]
        }
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
struct OrbView: View {
    var level: CGFloat
    var mood: OrbMood

    var body: some View {
        TimelineView(.animation) { timeline in
            let time = timeline.date.timeIntervalSinceReferenceDate
            GeometryReader { proxy in
                let size = min(proxy.size.width, proxy.size.height)
                let colors = mood.palette
                ZStack {
                    Circle()
                        .fill(RadialGradient(
                            colors: [colors[1].opacity(0.9), Color.black.opacity(0.9)],
                            center: .center,
                            startRadius: 0,
                            endRadius: size * 0.55
                        ))
                    ForEach(0..<3, id: \.self) { index in
                        blob(index: index, color: colors[index], time: time, size: size)
                    }
                    Circle()
                        .fill(RadialGradient(
                            colors: [Color.white.opacity(0.35 + 0.3 * level), .clear],
                            center: UnitPoint(x: 0.35, y: 0.3),
                            startRadius: 0,
                            endRadius: size * 0.45
                        ))
                        .blendMode(.plusLighter)
                }
                .frame(width: size, height: size)
                .drawingGroup()
                .clipShape(Circle())
                .overlay(Circle().stroke(Color.white.opacity(0.18), lineWidth: 1))
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
