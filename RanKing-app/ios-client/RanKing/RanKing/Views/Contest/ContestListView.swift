//
//  MainView.swift
//  RanKing
//
//  Created by Jonathan Da Silva on 10/4/25.
//

import SwiftUI

struct ContestListView: View {
    var body: some View {
        TabView {
            Tab("Contests", systemImage: "list.bullet") {
                MainView(title: "Contests")
            }
            .badge(2)


            Tab("Active Contests", systemImage: "list.bullet") {
                MainView(title: "Active Contests")
            }
        }

    }
}

struct MainView: View {
    
    let title: String
    
    var body: some View {
        VStack {
            Text(title)
                .font(.title)
            
            Spacer()
            NavigationStack {
                List {
                    // TODO this are placeholders
                    NavigationLink(destination: ContestView()) {
                        Text("Contests1")
                    }
                    NavigationLink(destination: ContestView()) {
                        Text("Contests2")
                    }
                    NavigationLink(destination: ContestView()) {
                        Text("Contests3")
                    }
                    NavigationLink(destination: ContestView()) {
                        Text("Contests3")
                    }
                }
            }
        }
    }
}


#Preview {
    ContestListView()
}
