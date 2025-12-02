import SwiftUI

struct ProfileStats: View {
    let profile: UserProfile
    
    var body: some View {
        HStack {
            StatCard(label: "Submissions", value: "\(profile.submissions.count)")
            StatCard(label: "Votes Cast", value: "\(profile.totalVotesCast)")
            StatCard(label: "Avg Percentile", value: "\(Int(profile.avgPercentile * 100))%")
        }
        .padding(.horizontal)
    }
}

struct StatCard: View {
    let label: String
    let value: String
    
    var body: some View {
        VStack {
            Text(value)
                .font(.title3)
                .fontWeight(.semibold)
            Text(label)
                .font(.caption)
                .foregroundColor(.gray)
        }
        .frame(maxWidth: .infinity)
        .padding(12)
        .background(Color(.systemGray6))
        .cornerRadius(12)
    }
}

