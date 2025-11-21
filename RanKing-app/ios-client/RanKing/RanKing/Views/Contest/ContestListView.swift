//
//  MainView.swift
//  RanKing
//
//  Created by Jonathan Da Silva on 10/4/25.
//

import SwiftUI

struct ContestListView: View {
    
    let networkManager  = NetworkManager.getInstance()
    
    @State var contests: [Contest] = []
    
    @State var votingContests: [Contest] = []
    
    var body: some View {
        TabView {
            Tab("Active Contests", systemImage: "list.bullet") {
                ActiveContestsListView(contests: $contests)
            }

            Tab("Voting Contests", systemImage: "list.bullet") {
                VotingContestsListView(votingContests: $votingContests)
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
        VStack {
            Text("Voting Contests")
                .font(.title)
                .fontWeight(.bold)
            
            Spacer()
            NavigationStack {
                List {
                    ForEach(votingContests, id: \.contest_id) { votingContest in
                        NavigationLink(destination: VoteView(contestId: votingContest.contest_id)) {
                            Text("Voting Contest")
                        }
                    }
                }
                .listStyle(.plain)
                .padding(.top)
            }
        }
    }
}

struct ActiveContestsListView: View {
    
    @Binding var contests : [Contest]
    
    var body: some View {
        VStack {
            Text("Contests")
                .font(.title)
                .fontWeight(.bold)
            
            Spacer()
            NavigationStack {
                List {
                    ForEach(contests, id: \.contest_id) { contest in
                        let contestViewData = ContestViewData(contestTitle: contest.name,
                                                              contestDescription:contest.description,
                                                              constantId: contest.contest_id,
                                                              votingPeriod: .init(start: Date(), duration: 120))
                            NavigationLink(destination: ContestView(contestData: contestViewData)) {
                                Text(contestViewData.contestTitle)
                        }
                    }
                }
                .listStyle(.plain)
                .padding(.top)
            }
        }
    }
}


#Preview {
    ContestListView()
}
