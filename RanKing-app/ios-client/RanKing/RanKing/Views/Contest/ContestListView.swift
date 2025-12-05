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
    if lower.contains("winter"){
        return "winter"
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

private struct ContestFilterChips: View {
    @Binding var selected: String
    let options: [String]

    var body: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 10) {
                ForEach(options, id: \.self) { option in
                    Button {
                        selected = option
                    } label: {
                        Text(option)
                            .font(.subheadline.weight(.semibold))
                            .padding(.horizontal, 12)
                            .padding(.vertical, 8)
                            .background(
                                Capsule()
                                    .fill(selected == option ? Color.primary.opacity(0.12) : Color.secondary.opacity(0.08))
                            )
                    }
                    .buttonStyle(.plain)
                }
            }
            .padding(.horizontal)
        }
    }
}

struct ContestListView: View {
    
    let networkManager  = NetworkManager.getInstance()
    
    enum SortOption: String, CaseIterable, Identifiable {
        case newest = "Newest"
        case endingSoon = "Ending Soon"
        case popular = "Most Popular"
        var id: String { rawValue }
    }
    
    @State var contests: [Contest] = []
    @State var votingContests: [Contest] = []
    @State private var searchQuery: String = ""
    @State private var sortOption: SortOption = .newest
    
    var body: some View {
        TabView {
            Tab("Active Contests", systemImage: "bolt.circle") {
                NavigationStack {
                    ActiveContestsListView(contests: $contests, searchQuery: $searchQuery, sortOption: $sortOption)
                        .navigationBarTitleDisplayMode(.large)
                        .toolbar {
                            ToolbarItem(placement: .principal) {
                                HStack(spacing: 10) {
                              
                                    Image(systemName: "crown.fill")
                                        .foregroundStyle(.primary)
                                        .imageScale(.large)
                                        .symbolRenderingMode(.hierarchical)
                                    Text("Active Contests")
                                     
                                        .font(.system(size: 34, weight: .semibold, design: .serif))
                                        .kerning(0.5)
                                        .foregroundStyle(.primary)
                                        .lineLimit(1)
                                        .minimumScaleFactor(0.85)
                                }
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .padding(.bottom, 6)
                            }
                        }
                }
                .searchable(text: $searchQuery, placement: .navigationBarDrawer(displayMode: .automatic))
                .toolbar {
                    Menu {
                        ForEach(SortOption.allCases) { option in
                            Button(option.rawValue) { sortOption = option }
                        }
                    } label: {
                        Label("Sort", systemImage: "arrow.up.arrow.down")
                    }
                }
            }

            Tab("Voting Contests", systemImage: "checklist") {
                NavigationStack {
                    VotingContestsListView(votingContests: $votingContests)
                        .navigationTitle("Voting Contests")
                }
            }
            Tab("Completed Contests", systemImage: "checkmark.seal") {
                NavigationStack {
                    VStack(spacing: 16) {
                        Image(systemName: "checkmark.seal.fill")
                            .font(.system(size: 48))
                            .foregroundStyle(.secondary)
                        Text("Completed Contests")
                            .font(.title2.weight(.semibold))
                        Text("Nothing to show yet.")
                            .font(.subheadline)
                            .foregroundStyle(.secondary)
                    }
                    .padding()
                    .navigationTitle("Completed Contests")
                }
            }
        }
        .onAppear(perform: fetchData)

    }
    
    func fetchData() {
        Task {
            contests = try await networkManager.fetchContests() ?? []
            votingContests = Array(contests)
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
    @Binding var searchQuery: String
    @Binding var sortOption: ContestListView.SortOption
    
    private var filteredAndSorted: [Contest] {
        var items = contests
        // Filter by search query (name or description)
        let q = searchQuery.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        if !q.isEmpty {
            items = items.filter { c in
                c.name.lowercased().contains(q) || c.description.lowercased().contains(q)
            }
        }
        // Sort options – placeholder logic (adjust when you have dates/popularity)
        switch sortOption {
        case .newest:
            items = items.sorted { $0.contest_id > $1.contest_id }
        case .endingSoon:
            items = items.sorted { $0.contest_id < $1.contest_id }
        case .popular:
            break // no-op until popularity is available
        }
        return items
    }
    
    var body: some View {
        ScrollView {
            LazyVStack(spacing: 16) {
                HStack {
                    Text("\(filteredAndSorted.count) contests")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                    Spacer()
                    Button {
                        withAnimation(.snappy) {
                            switch sortOption {
                            case .newest:
                                sortOption = .endingSoon
                            case .endingSoon:
                                sortOption = .newest
                            case .popular:
                                // If currently popular, default to newest on tap
                                sortOption = .newest
                            }
                        }
                    } label: {
                        HStack(spacing: 8) {
                            Image(systemName: sortOption == .endingSoon ? "arrow.down" : "arrow.up")
                            Text(sortOption.rawValue)
                            Image(systemName: "chevron.up.chevron.down")
                                .font(.caption2)
                                .opacity(0.7)
                        }
                        .padding(.horizontal, 10)
                        .padding(.vertical, 6)
                        .background(
                            Capsule().fill(Color.secondary.opacity(0.12))
                        )
                        .overlay(
                            Capsule().strokeBorder(Color.secondary.opacity(0.2), lineWidth: 0.5)
                        )
                        .contentShape(Capsule())
                    }
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                    .buttonStyle(.plain)
                    .accessibilityLabel("Toggle sort: currently sorted by \(sortOption.rawValue)")
                    .hoverEffect(.highlight)
                    .scaleEffect(1.0)
                    .animation(.easeInOut(duration: 0.15), value: sortOption)
                }
                .padding(.horizontal)
                
                ForEach(filteredAndSorted, id: \.contest_id) { contest in
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

