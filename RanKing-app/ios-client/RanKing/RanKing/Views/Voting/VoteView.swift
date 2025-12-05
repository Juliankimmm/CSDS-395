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
        print("Start downloading: \(submission.submission_id)")
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
    private let preloadBufferSize = 2
    @State var allSubmissions : [Submission] = []
    @State private var topSubmission: SubmissionAndImage?
    @State private var bottomSubmission: SubmissionAndImage?
    @State private var preloadedQueue: [SubmissionAndImage] = []
    @State private var nextBufferIndex = 0
    @State private var popUpScale : CGFloat = 1.0
    
    var body: some View {
        VStack(spacing: 20) {
            Text("Vote on Outfit")
                .font(.title2)
                .bold()
            
            VStack(spacing: 20) {
                // TOP IMAGE
                if let submission = topSubmission {
                    ObservableSubmissionView(
                        submissionAndImage: submission,
                        scale: $popUpScale
                    ) { voteDirection in
                        handleVote(for: submission, position: .top)
                    }
                    .allowsHitTesting(bottomSubmission != nil)
                    .opacity(bottomSubmission != nil ? 1.0 : 1.0)
                }
                
                // BOTTOM IMAGE
                if let submission = bottomSubmission {
                    ObservableSubmissionView(
                        submissionAndImage: submission,
                        scale: $popUpScale
                    ) { voteDirection in
                        handleVote(for: submission, position: .bottom)
                    }
                    .allowsHitTesting(topSubmission != nil)
                    .opacity(topSubmission != nil ? 1.0 : 1.0)
                }
            }
            
            // Text logic: Show "No more" if BOTH are gone,
            // OR show "Waiting for opponent" if only one is left.
            if topSubmission == nil && bottomSubmission == nil && !allSubmissions.isEmpty {
                Text("No more submissions to vote on!")
                    .font(.headline)
            } else if (topSubmission == nil || bottomSubmission == nil) && !allSubmissions.isEmpty {
                Text("This image was the best!!!")
                    .font(.headline)
                    .foregroundColor(.primary)
            }
        }
        .onAppear(perform: setupInitialSubmissions)
        .padding()
    }
    
    // Helper to clean up the body code
    func handleVote(for submission: SubmissionAndImage, position: ImagePosition) {
        replaceSubmission(in: position)
        withAnimation {
            popUpScale = 1.0
        }
        Task {
            if let userId = UserDefaults.standard.string(forKey: "user_id") {
                try? await networkManager.sendVote(submissionId: submission.submission.submission_id, userId: userId);
            }
            
        }
    }
    
    func replaceSubmission(in position: ImagePosition) {
        print("Replacing submission at position: \(position)")
        guard !preloadedQueue.isEmpty else {
            if nextBufferIndex < allSubmissions.count {
                let directLoad = SubmissionAndImage(submission: allSubmissions[nextBufferIndex])
                nextBufferIndex += 1
                assignNewSubmission(directLoad, to: position)
                return
            }
            if position == .top { topSubmission = nil }
            else { bottomSubmission = nil }
            return
        }
        let newSubmission = preloadedQueue.removeFirst()
        assignNewSubmission(newSubmission, to: position)
        fillBuffer()
    }
    
    func assignNewSubmission(_ newSubmission: SubmissionAndImage, to position: ImagePosition) {
        withAnimation(.bouncy(duration: 0.5)) {
            if position == .top {
                topSubmission = newSubmission
            } else {
                bottomSubmission = newSubmission
            }
            popUpScale = 0.1
        }
    }

    func fillBuffer() {
        while preloadedQueue.count < preloadBufferSize && nextBufferIndex < allSubmissions.count {
            print("Preloading index: \(nextBufferIndex)")
            let submissionMeta = allSubmissions[nextBufferIndex]
            let preloadedItem = SubmissionAndImage(submission: submissionMeta)
            preloadedQueue.append(preloadedItem)
            nextBufferIndex += 1
        }
    }

    func setupInitialSubmissions() {
        print("GET submissions/{contest_id}")
        Task {
            allSubmissions = try await networkManager.fetchSumbissions(contestId: contestId) ?? []
            if allSubmissions.count >= 1 {
                topSubmission = SubmissionAndImage(submission: allSubmissions[0])
                nextBufferIndex = 1
            }
            if allSubmissions.count >= 2 {
                bottomSubmission = SubmissionAndImage(submission: allSubmissions[1])
                nextBufferIndex = 2
            }
            fillBuffer()
        }
    }
}

#Preview {
    VoteView(contestId: "1")
}
