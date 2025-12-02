//
//  StatsDetailView.swift
//  RanKing
//
//  Created by Damario Hamilton on 11/29/25.
//
import SwiftUI

struct StatsDetailView: View {
    let profile: UserProfile
    
    var body: some View {
        VStack(spacing: 20) {
            
            // Performance Card
            VStack(alignment: .leading, spacing: 12) {
                Text("Performance")
                    .font(.headline)
                
                Divider()
                
                HStack {
                    Text("Best Submission Rank")
                    Spacer()
                    Text("#\(bestSubmissionRank())")
                }
                
                HStack {
                    Text("Avg Win Rate")
                    Spacer()
                    Text("\(Int(profile.avgWinRate * 100))%")
                }
                
                HStack {
                    Text("Avg Percentile")
                    Spacer()
                    Text("\(Int(profile.avgPercentile * 100))%")
                }
                
                HStack {
                    Text("Total Votes Received")
                    Spacer()
                    Text("\(totalVotesReceived())")
                }
            }
            .padding()
            .background(Color(.systemGray6))
            .cornerRadius(16)
            .padding(.horizontal)
            
            // Recent Performance
            VStack(alignment: .leading, spacing: 12) {
                Text("Recent Performance")
                    .font(.headline)
                
                Divider()
                
                HStack {
                    Text("Last 10 Votes Win Rate")
                    Spacer()
                    Text("72%")
                }
                
                HStack {
                    Text("Trend")
                    Spacer()
                    Text("↑ Improving")
                        .foregroundColor(.green)
                }
                
                HStack {
                    Text("Engagement Score")
                    Spacer()
                    Text("High")
                        .fontWeight(.semibold)
                }
            }
            .padding()
            .background(Color(.systemGray6))
            .cornerRadius(16)
            .padding(.horizontal)
        }
    }
    
    func bestSubmissionRank() -> Int {
        profile.submissions.map { $0.rank }.min() ?? 0
    }
    
    func totalVotesReceived() -> Int {
        profile.submissions.reduce(0) { $0 + $1.votes }
    }
}
