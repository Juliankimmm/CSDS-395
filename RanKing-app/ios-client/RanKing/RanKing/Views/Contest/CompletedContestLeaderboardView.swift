import SwiftUI

struct CompletedContestLeaderboardView: View {
    let contestId: String
    let contestName: String?

    var body: some View {
        VStack(spacing: 16) {
            if let name = contestName, !name.isEmpty {
                VStack(spacing: 6) {
                    HStack(spacing: 8) {
                        Image(systemName: "crown.fill")
                            .foregroundStyle(.yellow)
                            .imageScale(.large)
                        Text(name)
                            .font(.system(size: 26, weight: .bold))
                            .multilineTextAlignment(.center)
                    }
                    Text("Who took the crown?")
                        .font(.subheadline.weight(.semibold))
                        .foregroundStyle(.secondary)
                }
            } else {
                VStack(spacing: 6) {
                    HStack(spacing: 8) {
                        Image(systemName: "crown.fill")
                            .foregroundStyle(.yellow)
                            .imageScale(.large)
                        Text("Contest \(contestId)")
                            .font(.system(size: 26, weight: .bold))
                            .multilineTextAlignment(.center)
                    }
                    Text("Who took the crown?")
                        .font(.subheadline.weight(.semibold))
                        .foregroundStyle(.secondary)
                }
            }

            Capsule()
                .fill(Color.secondary.opacity(0.15))
                .frame(height: 1)
                .padding(.horizontal)

            RankedSwipeView(contestId: contestId)
                .padding(.top, 8)
                .frame(maxWidth: .infinity, alignment: .center)

            Spacer()
        }
        .padding()
        .navigationTitle("Leaderboard")
        .navigationBarTitleDisplayMode(.inline)
    }
}

#Preview {
    NavigationStack {
     
    }
}
