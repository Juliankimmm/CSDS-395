import SwiftUI

struct VoteViewData {
    var image1 : Image
    var image2 : Image
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
            if let data = try? await NetworkManager.getInstance()
                .getSubmissionImage(submissionId: submission.submission_id),
               let uiImg = UIImage(data: data) {
                self.image = Image(uiImage: uiImg)
                print("Finished downloading: \(submission.submission_id)")
            }
        }
    }
}

struct ObservableSubmissionView: View {
    @ObservedObject var submissionAndImage: SubmissionAndImage
    @Binding var scale: CGFloat
    var onVote: (SwipeDirection) -> Void
    
    var body: some View {
        VotableImageView(
            image: submissionAndImage.image ?? Image(systemName: "photo"),
            scale: $scale,
            onVote: onVote
        )
        .id(submissionAndImage.submission.submission_id)
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
                if let submission = topSubmission {
                    ObservableSubmissionView(
                        submissionAndImage: submission,
                        scale: $popUpScale
                    ) { voteDirection in
                        handleVote(for: submission, position: .top)
                    }
                }

                if let submission = bottomSubmission {
                    ObservableSubmissionView(
                        submissionAndImage: submission,
                        scale: $popUpScale
                    ) { voteDirection in
                        handleVote(for: submission, position: .bottom)
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
    VoteView(contestId: "403")
}
