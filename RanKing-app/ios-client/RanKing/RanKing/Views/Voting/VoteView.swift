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
        VotableImageView(
            image: submissionAndImage.image ?? Image(systemName: "photo"),
            scale: $scale,
            onVote: onVote
        )
        // This ensures SwiftUI sees it as a unique view when you swap submissions
        .id(submissionAndImage.submission.submission_id)
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
            if let data = try? await NetworkManager.getInstance()
                .getSubmissionImage(submissionId: submission.submission_id),
               let uiImg = UIImage(data: data) {
                self.image = Image(uiImage: uiImg)
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
                            if let userId = UserDefaults.standard.string(forKey: "user_id") {
                                try? await networkManager.sendVote(submissionId: submission.submission.submission_id, userId: userId);
                            }
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
