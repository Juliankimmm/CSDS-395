import SwiftUI

struct VoteViewData {
    var image1 : Image
    var image2 : Image
}

struct ObservableSubmissionView: View {
    // This tells SwiftUI: "When this object changes, redraw THIS view immediately"
    @ObservedObject var submissionAndImage: SubmissionAndImage
    @Binding var scale: CGFloat
    var onVote: (SwipeDirection) -> Void
    
    var body: some View {
        ZStack {
            VotableImageView(
                image: submissionAndImage.image ?? Image(systemName: "photo"),
                scale: $scale,
                onVote: onVote
            )
            .id(submissionAndImage.submission.submission_id)

            // Delayed loading indicator: shows only if image hasn't arrived quickly
            DelayedProgressOverlay(isLoading: submissionAndImage.image == nil, delay: 0.2)
        }
    }
}

private struct DelayedProgressOverlay: View {
    let isLoading: Bool
    let delay: Double
    @State private var show: Bool = false

    var body: some View {
        Group {
            if show && isLoading {
                ZStack {
                    Color.black.opacity(0.08)
                        .ignoresSafeArea()
                    ProgressView()
                        .progressViewStyle(.circular)
                        .tint(.secondary)
                        .scaleEffect(1.1)
                }
                .transition(.opacity)
            }
        }
        .onChange(of: isLoading) { _, newValue in
            if newValue {
                show = false
                Task { @MainActor in
                    try? await Task.sleep(nanoseconds: UInt64(delay * 1_000_000_000))
                    if isLoading { withAnimation(.easeIn(duration: 0.15)) { show = true } }
                }
            } else {
                withAnimation(.easeOut(duration: 0.1)) { show = false }
            }
        }
        .task {
            if isLoading {
                try? await Task.sleep(nanoseconds: UInt64(delay * 1_000_000_000))
                if isLoading { withAnimation(.easeIn(duration: 0.15)) { show = true } }
            }
        }
    }
}

@MainActor
class SubmissionAndImage : ObservableObject, Identifiable {
    var submission: Submission
    @Published var image: Image? = nil
    
    init(submission: Submission) {
        self.submission = submission
        loadImage()
    }
    
    private func loadImage() {
        Task {
            let id = submission.submission_id
            // 0) Cache first
            if let cached = await ImageCache.shared.image(for: id) {
                self.image = Image(uiImage: cached)
                return
            }
            // Fetch bytes from backend
            if let data = try? await NetworkManager.getInstance()
                .getSubmissionImage(submissionId: id) {
                // 1) Try direct bytes -> UIImage
                if let uiImg = UIImage(data: data) {
                    await ImageCache.shared.setImage(uiImg, for: id)
                    self.image = Image(uiImage: uiImg)
                    return
                }
                // 2) Try to interpret as UTF8 base64 string
                if let stringBody = String(data: data, encoding: .utf8) {
                    struct ImageEnvelope: Decodable { let image: String }
                    if let jsonData = stringBody.data(using: .utf8),
                       let envelope = try? JSONDecoder().decode(ImageEnvelope.self, from: jsonData),
                       let decoded = Data(base64Encoded: envelope.image, options: [.ignoreUnknownCharacters]),
                       let uiImg = UIImage(data: decoded) {
                        await ImageCache.shared.setImage(uiImg, for: id)
                        self.image = Image(uiImage: uiImg)
                        return
                    }
                    if let decoded = Data(base64Encoded: stringBody.trimmingCharacters(in: .whitespacesAndNewlines),
                                          options: [.ignoreUnknownCharacters]),
                       let uiImg = UIImage(data: decoded) {
                        await ImageCache.shared.setImage(uiImg, for: id)
                        self.image = Image(uiImage: uiImg)
                        return
                    }
                }
            }
        }
    }
}

struct VoteView: View {
    
    let networkManager : NetworkManager = NetworkManager.getInstance()
    let contestId: String
    @State var allSubmissions : [Submission] = []
    @State private var topSubmission: SubmissionAndImage?
    @State private var bottomSubmission: SubmissionAndImage?
    @State private var nextSubmissionIndex = 0
    @State private var popUpScale : CGFloat = 1.0
    
    var body: some View {
        VStack(spacing: 20) {
            Text("Vote on Outfit")
                .font(.title2)
                .bold()

            VStack(spacing: 20) {
                if let submission = topSubmission {
                    ObservableSubmissionView(
                        submissionAndImage: submission,
                        scale: $popUpScale
                    ) { voteDirection in
                        replaceSubmission(in: .top)
                        withAnimation {
                            popUpScale = 1.0
                        }
                        Task {
                            try? await networkManager.sendVote(submissionId: submission.submission.submission_id, userId: 1);
                        }
                    }
                }

                if let submission = bottomSubmission {
                    ObservableSubmissionView(
                        submissionAndImage: submission,
                        scale: $popUpScale
                    ) { voteDirection in
                        replaceSubmission(in: .bottom)
                        withAnimation {
                            popUpScale = 1.0
                        }
                    }
                }
            }
            
            if topSubmission == nil && bottomSubmission == nil && !allSubmissions.isEmpty {
                Text("No more submissions to vote on!")
                    .font(.headline)
            }
        }
        .onAppear(perform: setupInitialSubmissions)
        .padding()
    }
    
    func replaceSubmission(in position: ImagePosition) {
        print("Voted and replacing submission at position: \(position)")

        guard nextSubmissionIndex < allSubmissions.count else {
            if position == .top {
                topSubmission = nil
            } else {
                bottomSubmission = nil
            }
            return
        }
        
        let newSubmission = allSubmissions[nextSubmissionIndex]
        withAnimation(.bouncy(duration: 0.5)) {
            if position == .top {
                topSubmission = SubmissionAndImage(submission: newSubmission)
            } else {
                bottomSubmission = SubmissionAndImage(submission: newSubmission)
            }
            popUpScale = 0.1
        }
        
        nextSubmissionIndex += 1
    }

    func setupInitialSubmissions() {
        print("GET submissions/{contest_id}")
        Task {
            allSubmissions = try await networkManager.fetchSumbissions(contestId: contestId) ?? []
            print(allSubmissions)
            
            if allSubmissions.count >= 2 {
                topSubmission = SubmissionAndImage(submission: allSubmissions[0])
                bottomSubmission = SubmissionAndImage(submission: allSubmissions[1])
                nextSubmissionIndex = 2
            }
            else if allSubmissions.count == 1 {
                topSubmission = SubmissionAndImage(submission: allSubmissions[0])
                nextSubmissionIndex = 1
            }
        }
    }
    
    
}

#Preview {
    VoteView(contestId: "403")
}
