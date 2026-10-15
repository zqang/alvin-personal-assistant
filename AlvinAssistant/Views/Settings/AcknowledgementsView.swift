import SwiftUI

/// Settings › Acknowledgements: the open models and open-source projects the app uses, with their
/// licences, from `Acknowledgements.txt` in the app bundle.
struct AcknowledgementsView: View {
    private let text = Bundle.main.url(forResource: "Acknowledgements", withExtension: "txt")
        .flatMap { try? String(contentsOf: $0, encoding: .utf8) }
        ?? "The acknowledgements are missing from this build."

    var body: some View {
        ScrollView {
            Text(text)
                .font(.footnote)
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding()
        }
        .navigationTitle("Acknowledgements")
        .navigationBarTitleDisplayMode(.inline)
    }
}
