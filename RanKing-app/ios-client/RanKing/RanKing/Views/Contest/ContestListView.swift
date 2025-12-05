//
//  MainView.swift
//  RanKing
//
//  Created by Jonathan Da Silva on 10/4/25.
//

import SwiftUI

private func imageName(for contest: Contest) -> String {
    let lower = contest.name.lowercased()
    if lower.contains("autumn") { return "autumn" }
    if lower.contains ("summer"){ return "summer"}
    if lower.contains("spring"){
        return "spring"
    }
    return ["fashion1", "fashion2", "fashion3"].randomElement() ?? "fashion1"
}

private struct ContestCard: View {
    let title: String
    let subtitle: String
    let imageName: String

    var body: some View {
        ZStack(alignment: .bottomLeading) {
            Image(imageName)
                .resizable()
                .scaledToFill()
                .frame(height: 180)
                .frame(maxWidth: .infinity)
                .clipped()
                .overlay(
                    LinearGradient(
                        colors: [Color.black.opacity(0.0), Color.black.opacity(0.6)],
                        startPoint: .center,
                        endPoint: .bottom
                    )
                )
                .clipShape(RoundedRectangle(cornerRadius: 18, style: .continuous))

            VStack(alignment: .leading, spacing: 6) {
                Text(title)
                    .font(.headline)
                    .foregroundColor(.white)
                    .lineLimit(2)
                if !subtitle.isEmpty {
                    Text(subtitle)
                        .font(.subheadline)
                        .foregroundColor(.white.opacity(0.85))
                        .lineLimit(2)
                }
            }
            .padding(14)
        }
        .background(
            RoundedRectangle(cornerRadius: 18, style: .continuous)
                .fill(Color(.systemBackground))
        )
        .shadow(color: Color.black.opacity(0.12), radius: 10, y: 6)
        .contentShape(RoundedRectangle(cornerRadius: 18, style: .continuous))
    }
}

struct ContestListView: View {
    
    let networkManager  = NetworkManager.getInstance()
    
    @State var contests: [Contest] = []
    
    @State var votingContests: [Contest] = []
    
    var body: some View {
        TabView {
            Tab("Active Contests", systemImage: "list.bullet") {
                NavigationStack {
                    ActiveContestsListView(contests: $contests)
                        .navigationTitle("Active Contests")
                }
            }

            Tab("Voting Contests", systemImage: "list.bullet") {
                NavigationStack {
                    VotingContestsListView(votingContests: $votingContests)
                        .navigationTitle("Voting Contests")
                }
            }
        }
        .onAppear(perform: fetchData)

    }
    
    func fetchData() {
        Task {
            contests = try await networkManager.fetchContests() ?? []
        }
    }
}

struct VotingContestsListView: View {
    
    @Binding var votingContests : [Contest]
    
    var body: some View {
        ScrollView {
            LazyVStack(spacing: 16) {
                ForEach(votingContests, id: \.contest_id) { votingContest in
                    NavigationLink(destination: VoteView(contestId: votingContest.contest_id)) {
                        ContestCard(title: votingContest.name,
                                    subtitle: votingContest.description,
                                    imageName: imageName(for: votingContest))
                            .padding(.horizontal)
                    }
                    .buttonStyle(.plain)
                }
                Spacer(minLength: 12)
            }
            .padding(.top)
        }
    }
}

struct ActiveContestsListView: View {
    
    @Binding var contests : [Contest]
    
    var body: some View {
        ScrollView {
            LazyVStack(spacing: 16) {
                ForEach(contests, id: \.contest_id) { contest in
                    let data = ContestViewData(contestTitle: contest.name,
                                               contestDescription: contest.description,
                                               constantId: contest.contest_id,
                                               votingPeriod: .init(start: Date(), duration: 120))
                    NavigationLink(destination: ContestView(contestData: data)) {
                        ContestCard(title: data.contestTitle,
                                    subtitle: data.contestDescription,
                                    imageName: imageName(for: contest))
                            .padding(.horizontal)
                    }
                    .buttonStyle(.plain)
                }
                Spacer(minLength: 12)
            }
            .padding(.top)
        }
    }
}


#Preview {
    ContestListView()
}
