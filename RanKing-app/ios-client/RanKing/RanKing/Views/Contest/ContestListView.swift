//
//  MainView.swift
//  RanKing
//
//  Created by Jonathan Da Silva on 10/4/25.
//

import SwiftUI
import SwiftData

struct ContestListView: View {
    
    @Query var contests: [Contest] = []
    
    var body: some View {
        TabView {
            Tab("Active Contests", systemImage: "list.bullet") {
                ActiveContestsListView(contests: contests)
            }

            Tab("Voting Contests", systemImage: "list.bullet") {
                VotingContestsListView(votingContests: contests)
            }
        }
        .onAppear(perform: fetchData)

    }
    
    func fetchData() {
        // TODO fetch all contests GET contests/
        print("GET contests/")
    }
}

struct VotingContestsListView: View {
    
    let votingContests : [Contest]
    
    var body: some View {
        VStack {
            Text("Voting Contests")
                .font(.title)
                .fontWeight(.bold)
            
            Spacer()
            NavigationStack {
                List {
                    ForEach(votingContests) { votingContests in
                        if (votingContests.contestPhase == ContestPhase.VOTING) {
                            let contestId : UUID = votingContests.id
                            NavigationLink(destination: VoteView(contestId: contestId)) {
                                Text("Voting Contest")
                            }
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
    
    let contests : [Contest]
    
    var body: some View {
        VStack {
            Text("Contests")
                .font(.title)
                .fontWeight(.bold)
            
            Spacer()
            NavigationStack {
                List {
                    ForEach(contests) { contest in
                        let contestViewData = ContestViewData(contestTitle: contest.name,
                                                              contestDescription:contest.contestDescription,
                                                              submissionNumber: contest.numSumbissions,
                                                              votingPeriod: .init(start: Date(), duration: 120))
                        if (contest.contestPhase == ContestPhase.SUBMISSION) {
                            NavigationLink(destination: ContestView(contestData: contestViewData)) {
                                Text(contestViewData.contestTitle)
                            }
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
        .modelContainer(SampleData.shared.modelContainer)
}
