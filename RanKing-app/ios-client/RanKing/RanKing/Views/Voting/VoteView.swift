import SwiftUI

struct VoteView: View {
    @State private var voteMessage: String?

    var body: some View {
        VStack(spacing: 20) {
            Text("Vote on Outfit")
                .font(.title2)
                .bold()

            VStack(spacing: 20) {
                VotableImageView(
                    imageName: "jezthisguyishot"
                ) { voteDirection in
                    registerVote(swipeDir: voteDirection)
                }

                VotableImageView(
                    imageName: "cutiepie"
                ) { voteDirection in
                    registerVote(swipeDir: voteDirection)
                }
            }

            if let message = voteMessage {
                Text(message)
                    .font(.headline)
                    .foregroundColor(.green)
                    .transition(.opacity)
                    .padding(.top)
            }
        }
        .padding()
    }
    
    func registerVote(swipeDir : SwipeDirection) -> Void {
        if (swipeDir == SwipeDirection.up) {
            voteMessage = "👍 Voted up bottom outfit!"
        }
        else {
            voteMessage = "👎 Voted down top outfit!"
        }
    }
}



#Preview {
    VoteView()
}
