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
    return "spring"
}

private struct ContestCard: View {
    let title: String
    let subtitle: String
    let imageName: String

    var body: some View {
        ZStack(alignment: .bottomLeading) {
            Group {
                if imageName == "waiting" {
                    ZStack {
                        RoundedRectangle(cornerRadius: 18, style: .continuous)
                            .fill(Color(.systemGray5))
                            .frame(height: 180)
                        ProgressView()
                            .progressViewStyle(.circular)
                            .tint(.secondary)
                    }
                    .frame(maxWidth: .infinity)
                    .overlay(
                        LinearGradient(
                            colors: [Color.black.opacity(0.0), Color.black.opacity(0.6)],
                            startPoint: .center,
                            endPoint: .bottom
                        )
                        .clipShape(RoundedRectangle(cornerRadius: 18, style: .continuous))
                    )
                } else {
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
                }
            }

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
    @State private var allContests: [Contest] = []
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
                    CompletedContestsListView(contests: $allContests)
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
            let all = try await networkManager.fetchContests() ?? []
            let now = Date()
            allContests = all
            contests = all.filter { $0.submission_end_date > now }
            votingContests = all.filter { $0.submission_end_date <= now && $0.voting_end_date >= now }
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
    
    @State private var prefetchedContestIds: Set<String> = []
    
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
    
    private func prefetchTopImages(for contestId: String, limit: Int = 6) {
        // Prevent duplicate prefetch for the same contest
        if prefetchedContestIds.contains(contestId) { return }
        prefetchedContestIds.insert(contestId)
        Task(priority: .background) {
            // Hop to the main actor only to get the instance
            let manager: NetworkManager = await MainActor.run { NetworkManager.getInstance() }
            // Fetch submissions for this contest
            let subs = (try? await manager.fetchSumbissions(contestId: contestId)) ?? []
            // Sort by vote_count desc to prefetch likely leaders
            let ranked = subs.sorted { $0.vote_count > $1.vote_count }
            let count = min(limit, ranked.count)
            for i in 0..<count {
                let id = ranked[i].submission_id
                if let _ = await ImageCache.shared.image(for: id) { continue }
                if let data = try? await manager.getSubmissionImage(submissionId: id) {
                    if let uiImg = UIImage(data: data) {
                        await ImageCache.shared.setImage(uiImg, for: id)
                    } else if let stringBody = String(data: data, encoding: .utf8) {
                        struct ImageEnvelope: Decodable { let image: String }
                        if let jsonData = stringBody.data(using: .utf8),
                           let envelope = try? JSONDecoder().decode(ImageEnvelope.self, from: jsonData),
                           let decoded = Data(base64Encoded: envelope.image, options: [.ignoreUnknownCharacters]),
                           let uiImg = UIImage(data: decoded) {
                            await ImageCache.shared.setImage(uiImg, for: id)
                        } else if let decoded = Data(base64Encoded: stringBody.trimmingCharacters(in: .whitespacesAndNewlines), options: [.ignoreUnknownCharacters]),
                                  let uiImg = UIImage(data: decoded) {
                            await ImageCache.shared.setImage(uiImg, for: id)
                        }
                    }
                }
            }
        }
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
                    .onAppear { prefetchTopImages(for: votingContest.contest_id) }
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
                            Image(systemName: sortOption == .endingSoon ? "arrow-down" : "arrow-up")
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
                                               contestId: contest.contest_id,
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
    
    @State private var prefetchedContestIds: Set<String> = []

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
    
    private func prefetchTopImages(for contestId: String, limit: Int = 6) {
        // Prevent duplicate prefetch for the same contest
        if prefetchedContestIds.contains(contestId) { return }
        prefetchedContestIds.insert(contestId)
        Task(priority: .background) {
            // Hop to the main actor only to get the instance
            let manager: NetworkManager = await MainActor.run { NetworkManager.getInstance() }
            // Fetch submissions for this contest
            let subs = (try? await manager.fetchSumbissions(contestId: contestId)) ?? []
            // Sort by vote_count desc to prefetch likely leaders
            let ranked = subs.sorted { $0.vote_count > $1.vote_count }
            let count = min(limit, ranked.count)
            for i in 0..<count {
                let id = ranked[i].submission_id
                if let _ = await ImageCache.shared.image(for: id) { continue }
                if let data = try? await manager.getSubmissionImage(submissionId: id) {
                    if let uiImg = UIImage(data: data) {
                        await ImageCache.shared.setImage(uiImg, for: id)
                    } else if let stringBody = String(data: data, encoding: .utf8) {
                        struct ImageEnvelope: Decodable { let image: String }
                        if let jsonData = stringBody.data(using: .utf8),
                           let envelope = try? JSONDecoder().decode(ImageEnvelope.self, from: jsonData),
                           let decoded = Data(base64Encoded: envelope.image, options: [.ignoreUnknownCharacters]),
                           let uiImg = UIImage(data: decoded) {
                            await ImageCache.shared.setImage(uiImg, for: id)
                        } else if let decoded = Data(base64Encoded: stringBody.trimmingCharacters(in: .whitespacesAndNewlines), options: [.ignoreUnknownCharacters]),
                                  let uiImg = UIImage(data: decoded) {
                            await ImageCache.shared.setImage(uiImg, for: id)
                        }
                    }
                }
            }
        }
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
                            Image(systemName: sortOption == .endingSoon ? "arrow-down" : "arrow-up")
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
                        contestId: contest.contest_id,
                        votingPeriod: .init(start: contest.submission_start_date, end: contest.voting_end_date)
                    )
                    NavigationLink(destination: CompletedContestLeaderboardView(contestId: contest.contest_id, contestName: contest.name)) {
                        ContestCard(
                            title: contest.name,
                            subtitle: contest.description,
                            imageName: imageName(for: contest)
                        )
                        .padding(.horizontal)
                    }
                    .onAppear { prefetchTopImages(for: contest.contest_id) }
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

