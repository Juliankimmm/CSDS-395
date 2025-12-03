import SwiftUI

struct VoteViewData {
    var image1 : Image
    var image2 : Image
}

struct VoteView: View {
    
    let networkManager : NetworkManager = NetworkManager.getInstance()
    
    let contestId: Int
    
    @State var allSubmissions : [SubmissionResponse] = []
    @State private var topSubmission: SubmissionResponse?
    @State private var bottomSubmission: SubmissionResponse?
    @State private var nextSubmissionIndex = 0
    
    @State private var popUpScale : CGFloat = 1.0
    
    var body: some View {
        VStack(spacing: 20) {
            Text("Vote on Outfit")
                .font(.title2)
                .bold()

            VStack(spacing: 20) {
                if let submission = topSubmission {
                    VotableImageView(
                        image: Image("jezthisguyishot"),
                        scale: $popUpScale
                    ) { voteDirection in
                        replaceSubmission(in: .top)
                        withAnimation {
                            popUpScale = 1.0
                        }
                    }
                    .id(submission.id)
                }

                if let submission = bottomSubmission {
                    VotableImageView(
                        image: Image("cutiepie"),
                        scale: $popUpScale
                    ) { voteDirection in
                        replaceSubmission(in: .bottom)
                        withAnimation {
                            popUpScale = 1.0
                        }
                    }
                    .id(submission.id)
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
                topSubmission = newSubmission
            } else {
                bottomSubmission = newSubmission
            }
            popUpScale = 0.1
        }
        
        nextSubmissionIndex += 1
    }

    func setupInitialSubmissions() {
        print("GET submissions/{contest_id}")
        Task {
//            allSubmissions = try await networkManager.fetchSumbissions(contestId: contestId) ?? []
            //TODO get the images
            allSubmissions = [SubmissionResponse(), SubmissionResponse()]
        }
        
        guard allSubmissions.count >= 2 else { return }
        
        topSubmission = allSubmissions[0]
        bottomSubmission = allSubmissions[1]
        nextSubmissionIndex = 2
        
        //TODO load Async ALL images
    }
    
    
}

#Preview {
    VoteView(contestId: 12)
}
