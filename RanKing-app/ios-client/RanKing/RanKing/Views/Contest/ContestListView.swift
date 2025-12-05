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
    
    @State var contests: [Contest] = []
    @State var votingContests: [Contest] = []
    @State private var searchQuery: String = ""
    
    var body: some View {
        TabView {
            Tab("Active Contests", systemImage: "bolt.circle") {
                NavigationStack {
                    ActiveContestsListView(contests: $contests, searchQuery: $searchQuery)
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
                                .padding(.bottom, 0)
                            }
                        }
                }
                .searchable(text: $searchQuery, placement: .navigationBarDrawer(displayMode: .automatic))
            }

            Tab("Voting Contests", systemImage: "checklist") {
                NavigationStack {
                    VotingContestsListView(votingContests: $votingContests)
                        .navigationBarTitleDisplayMode(.large)
                        .toolbar {
                            ToolbarItem(placement: .principal) {
                                HStack(spacing: 10) {
                                    Image(systemName: "crown.fill")
                                        .foregroundStyle(.primary)
                                        .imageScale(.large)
                                        .symbolRenderingMode(.hierarchical)
                                    Text("Voting Contests")
                                        .font(.system(size: 34, weight: .semibold, design: .serif))
                                        .kerning(0.5)
                                        .foregroundStyle(.primary)
                                        .lineLimit(1)
                                        .minimumScaleFactor(0.85)
                                }
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .padding(.bottom, 0)
                            }
                        }
                }
            }
            Tab("Completed Contests", systemImage: "checkmark.seal") {
                NavigationStack {
                    CompletedContestsListView(contests: $contests)
                        .navigationBarTitleDisplayMode(.large)
                        .toolbar {
                            ToolbarItem(placement: .principal) {
                                HStack(spacing: 10) {
                                    Image(systemName: "crown.fill")
                                        .foregroundStyle(.primary)
                                        .imageScale(.large)
                                        .symbolRenderingMode(.hierarchical)
                                    Text("Completed Contests")
                                        .font(.system(size: 34, weight: .semibold, design: .serif))
                                        .kerning(0.5)
                                        .foregroundStyle(.primary)
                                        .lineLimit(1)
                                        .minimumScaleFactor(0.85)
                                }
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .padding(.bottom, 0)
                            }
                        }
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
    
    @State private var searchQuery: String = ""
    
    enum SortOption: String, CaseIterable, Identifiable {
        case newest = "Newest"
        case endingSoon = "Ending Soon"
        var id: String { rawValue }
    }
    @State private var sortOption: SortOption = .newest
    
    private var filteredAndSorted: [Contest] {
        var items = votingContests
        let q = searchQuery.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        if !q.isEmpty {
            items = items.filter { c in
                c.name.lowercased().contains(q) || c.description.lowercased().contains(q)
            }
        }
        switch sortOption {
        case .newest:
            items = items.sorted { $0.contest_id > $1.contest_id }
        case .endingSoon:
            items = items.sorted { $0.contest_id < $1.contest_id }
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
                            sortOption = (sortOption == .newest) ? .endingSoon : .newest
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
                        .background(Capsule().fill(Color.secondary.opacity(0.12)))
                        .overlay(Capsule().strokeBorder(Color.secondary.opacity(0.2), lineWidth: 0.5))
                        .contentShape(Capsule())
                    }
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                    .buttonStyle(.plain)
                    .accessibilityLabel("Toggle sort: currently sorted by \(sortOption.rawValue)")
                    .hoverEffect(.highlight)
                    .animation(.easeInOut(duration: 0.15), value: sortOption)
                }
                .padding(.horizontal)
                
                ForEach(filteredAndSorted, id: \.contest_id) { votingContest in
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
        .searchable(text: $searchQuery, placement: .navigationBarDrawer(displayMode: .automatic))
    }
}

struct ActiveContestsListView: View {
    
    @Binding var contests : [Contest]
    @Binding var searchQuery: String
    
    enum SortOption: String, CaseIterable, Identifiable {
        case newest = "Newest"
        case endingSoon = "Ending Soon"
        var id: String { rawValue }
    }
    @State private var sortOption: SortOption = .newest
    
    private var filteredAndSorted: [Contest] {
        var items = contests
        let q = searchQuery.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        if !q.isEmpty {
            items = items.filter { c in
                c.name.lowercased().contains(q) || c.description.lowercased().contains(q)
            }
        }
        switch sortOption {
        case .newest:
            items = items.sorted { $0.contest_id > $1.contest_id }
        case .endingSoon:
            items = items.sorted { $0.contest_id < $1.contest_id }
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
                            sortOption = (sortOption == .newest) ? .endingSoon : .newest
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
                        .background(Capsule().fill(Color.secondary.opacity(0.12)))
                        .overlay(Capsule().strokeBorder(Color.secondary.opacity(0.2), lineWidth: 0.5))
                        .contentShape(Capsule())
                    }
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                    .buttonStyle(.plain)
                    .accessibilityLabel("Toggle sort: currently sorted by \(sortOption.rawValue)")
                    .hoverEffect(.highlight)
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

struct CompletedContestsListView: View {
    @Binding var contests: [Contest]

    @State private var searchQuery: String = ""
    
    enum SortOption: String, CaseIterable, Identifiable {
        case newest = "Newest"
        case endingSoon = "Oldest"
        var id: String { rawValue }
    }
    @State private var sortOption: SortOption = .newest

    private var filteredAndSorted: [Contest] {
        let now = Date()
        var items = contests.filter { $0.voting_end_date < now }
        let q = searchQuery.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        if !q.isEmpty {
            items = items.filter { c in
                c.name.lowercased().contains(q) || c.description.lowercased().contains(q)
            }
        }
        switch sortOption {
        case .newest:
            items = items.sorted { $0.voting_end_date > $1.voting_end_date }
        case .endingSoon:
            items = items.sorted { $0.voting_end_date < $1.voting_end_date }
        }
        return items
    }

    var body: some View {
        ScrollView {
            LazyVStack(spacing: 16) {
                
                HStack {
                    Text("\(filteredAndSorted.count) completed")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                    Spacer()
                    Button {
                        withAnimation(.snappy) {
                            sortOption = (sortOption == .newest) ? .endingSoon : .newest
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
                        .background(Capsule().fill(Color.secondary.opacity(0.12)))
                        .overlay(Capsule().strokeBorder(Color.secondary.opacity(0.2), lineWidth: 0.5))
                        .contentShape(Capsule())
                    }
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                    .buttonStyle(.plain)
                    .accessibilityLabel("Toggle sort: currently sorted by \(sortOption.rawValue)")
                    .hoverEffect(.highlight)
                    .animation(.easeInOut(duration: 0.15), value: sortOption)
                }
                .padding(.horizontal)

                ForEach(filteredAndSorted, id: \.contest_id) { contest in
                    let data = ContestViewData(
                        contestTitle: contest.name,
                        contestDescription: contest.description,
                        constantId: contest.contest_id,
                        votingPeriod: .init(start: contest.submission_start_date, end: contest.voting_end_date)
                    )
                    NavigationLink(destination: ContestView(contestData: data)) {
                        ContestCard(
                            title: contest.name,
                            subtitle: contest.description,
                            imageName: imageName(for: contest)
                        )
                        .padding(.horizontal)
                    }
                    .buttonStyle(.plain)
                }

                Spacer(minLength: 12)
            }
            .padding(.top)
        }
        .searchable(text: $searchQuery, placement: .navigationBarDrawer(displayMode: .automatic))
    }
}


#Preview {
    ContestListView()
}

