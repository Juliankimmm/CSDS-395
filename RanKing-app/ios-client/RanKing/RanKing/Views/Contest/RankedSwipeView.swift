import SwiftUI
import Combine

struct RankedSwipeView: View {
    let contestId: String
    private let networkManager = NetworkManager.getInstance()

    @State private var ranked: [Submission] = []
    @State private var currentIndex: Int = 0
    @State private var currentCard: SubmissionAndImage?
    @State private var scale: CGFloat = 1.0
    @State private var confettiCounter: Int = 0

    var body: some View {
        VStack(spacing: 12) {
            header
            ZStack {
                if let card = currentCard {
                    ZStack {
                        ObservableSubmissionView(
                            submissionAndImage: card,
                            scale: $scale
                        ) { _ in
                            advance()
                            withAnimation { scale = 1.0 }
                            confettiCounter += 1
                        }
                        .id(card.submission.submission_id)
                        .frame(maxWidth: .infinity, alignment: .center)
                    }
                } else if ranked.isEmpty {
                    ContentUnavailableView("No submissions", systemImage: "photo")
                } else {
                    Text("End of leaders")
                        .font(.headline)
                        .foregroundStyle(.secondary)
                }
            }
            .frame(maxWidth: .infinity)
            .overlay(
                ConfettiOverlay(trigger: confettiCounter)
            )
        }
        .task { await load() }
        .padding(.horizontal)
    }

    private var header: some View {
        HStack {
            Text("Top Submissions")
                .font(.headline)
            Spacer()
            if !ranked.isEmpty {
                Text("\(min(currentIndex+1, ranked.count)) of \(ranked.count)")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            }
        }
    }

    @MainActor
    private func load() async {
        // Fetch, sort by vote_count desc, and show first card
        let subs = (try? await networkManager.fetchSumbissions(contestId: contestId)) ?? []
        let sorted = subs.sorted { $0.vote_count < $1.vote_count }
        await MainActor.run {
            ranked = sorted
            // Prefetch first few images to warm the cache
            let prefetchCount = min(8, ranked.count)
            Task.detached(priority: .background) {
                for i in 0..<prefetchCount {
                    let id = ranked[i].submission_id
                    // Skip if already cached
                    if let _ = await ImageCache.shared.image(for: id) { continue }
                    if let data = try? await NetworkManager.getInstance().getSubmissionImage(submissionId: id) {
                        if let uiImg = UIImage(data: data) {
                            await ImageCache.shared.setImage(uiImg, for: id)
                        } else if let stringBody = String(data: data, encoding: .utf8) {
                            struct ImageEnvelope: Decodable { let image: String }
                            if let jsonData = stringBody.data(using: .utf8),
                               let envelope = try? JSONDecoder().decode(ImageEnvelope.self, from: jsonData),
                               let decoded = Data(base64Encoded: envelope.image, options: [.ignoreUnknownCharacters]),
                               let uiImg = UIImage(data: decoded) {
                                await ImageCache.shared.setImage(uiImg, for: id)
                            } else if let decoded = Data(base64Encoded: stringBody.trimmingCharacters(in: .whitespacesAndNewlines), options: [.ignoreUnknownCharacters]),
                                      let uiImg = UIImage(data: decoded) {
                                await ImageCache.shared.setImage(uiImg, for: id)
                            }
                        }
                    }
                }
            }
            currentIndex = 0
            if let first = ranked.first {
                currentCard = SubmissionAndImage(submission: first)
            } else {
                currentCard = nil
            }
            scale = 1.0
        }
    }

    private func advance() {
        guard !ranked.isEmpty else { return }
        let next = currentIndex + 1
        withAnimation(.bouncy(duration: 0.5)) {
            if ranked.indices.contains(next) {
                currentIndex = next
                currentCard = SubmissionAndImage(submission: ranked[next])
                scale = 0.1
            } else {
                currentCard = nil
            }
        }
    }
}

#Preview {
    RankedSwipeView(contestId: "1")
}

private struct ConfettiOverlay: View {
    let trigger: Int
    @State private var particles: [UUID] = []

    var body: some View {
        ZStack {
            ForEach(particles, id: \.self) { id in
                ConfettiParticle()
            }
        }
        .allowsHitTesting(false)
        .onChange(of: trigger) { _, _ in
            spawn()
        }
    }

    private func spawn() {
        particles = (0..<16).map { _ in UUID() }
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.8) {
            particles.removeAll()
        }
    }
}

private struct ConfettiParticle: View {
    @State private var offset: CGSize = .zero
    @State private var opacity: Double = 1
    private let symbol = ["🎉","✨","🏆","🎊","⭐️"].randomElement()!

    var body: some View {
        Text(symbol)
            .font(.system(size: 22))
            .opacity(opacity)
            .offset(offset)
            .onAppear {
                withAnimation(.easeOut(duration: 0.8)) {
                    offset = CGSize(width: Double.random(in: (-140)...140), height: Double.random(in: (-220)...(-80)))
                    opacity = 0
                }
            }
    }
}
