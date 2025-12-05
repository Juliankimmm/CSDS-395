import SwiftUI

struct ProfileView: View {
    @State private var selectedTab: Int = 0 // 0 = submissions, 1 = stats
    
    // Mock data, replace with backend data
    let profile = UserProfile(
        name: "Jane Doe",
        handle: "@janedoe",
        avatar: "avatar",
        submissions: [
            Submission(imageName: "look1", winRate: 0.945, votes: 1243, rank: 1),
            Submission(imageName: "look2", winRate: 0.873, votes: 956, rank: 4),
            Submission(imageName: "look3", winRate: 0.821, votes: 734, rank: 7),
            Submission(imageName: "look4", winRate: 0.798, votes: 612, rank: 12)
        ],
        totalVotesCast: 4200,
        avgPercentile: 0.87,
        avgWinRate: 0.67
    )
    
    var body: some View {
        ScrollView {
            VStack(spacing: 20) {
                
                ProfileHeader(profile: profile)
                
                ProfileStats(profile: profile)
                
                ProfileTabs(selectedTab: $selectedTab)
                
                if selectedTab == 0 {
                    SubmissionGrid(submissions: profile.submissions)
                } else {
                    StatsDetailView(profile: profile)
                }
            }
            .padding(.top, 10)
        }
        .navigationTitle("Profile")
    }
}

#Preview {
    ProfileView()
}
